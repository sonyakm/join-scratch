# %% [markdown]
# # TMS — Virtual IceChunk Store (All Granules)
#
# Builds an IceChunk store on S3 with virtual references to **all** available
# TMS granules in `s3://airborne-smce-prod-user-bucket/JOIN/TMS/{SAT_PRODUCT}`.
#
# **Store location:**
# `s3://airborne-smce-prod-user-bucket/JOIN/icechunk-stores/TMS/{SAT_PRODUCT}`
#
# No science data is downloaded — only chunk byte-range metadata is written.

# %% [markdown]
# 
import io
import h5py
import re
import numpy as np
import pandas as pd
import xarray as xr
from virtualizarr.parsers import HDFParser
from virtualizarr import open_virtual_dataset
from virtualizarr.writers.icechunk import virtual_dataset_to_icechunk
import icechunk
from icechunk import Repository, s3_storage
import boto3
import s3fs
import obstore
from obstore.store import S3Store
from obspec_utils.registry import ObjectStoreRegistry


S3_BUCKET = "airborne-smce-prod-user-bucket"
SAT_PREFIX = "TMS"
SAT_PRODUCT = "TS02_L1B-TB"
S3_PREFIX = f'JOIN/{SAT_PREFIX}/{SAT_PRODUCT}/'
S3_ICECHUNK = f'JOIN/icechunk-stores/{SAT_PREFIX}/{SAT_PRODUCT}/'
S3_REGION  = "us-west-2"

s3 = s3fs.S3FileSystem(anon=False, region=S3_REGION)
#Use boto3 to grab your working credentials (handles profiles, SSO, ~/.aws, etc.)
session = boto3.Session(region_name=S3_REGION)
creds = session.get_credentials()
#If boto3 found credentials, extract the raw strings
if creds:
    frozen = creds.get_frozen_credentials()
    ACCESS_KEY = frozen.access_key
    SECRET_KEY = frozen.secret_key
    if frozen.token:
        TOKEN = frozen.token

s3_store = S3Store.from_url(f"s3://{S3_BUCKET}/", region=S3_REGION, skip_signature=False,
            access_key_id=ACCESS_KEY,
            secret_access_key=SECRET_KEY,
            token=TOKEN  )
registry  = ObjectStoreRegistry({f"s3://{S3_BUCKET}/": s3_store})

def read_tms_coords(key: str) -> dict:
    """Eagerly read time, latitude, longitude from a TMS L1C-TC granule."""
    if key.startswith("s3://"):
        relative_key = key.split("/", 3)[-1]
    elif key.startswith(S3_BUCKET):
        relative_key = key.split("/", 1)[-1]
    else:
        relative_key = key

    buf = io.BytesIO(obstore.get(s3_store, relative_key).bytes())
    with h5py.File(buf, "r") as f:
        # 1. Read time components
        try:
            year = f["Year"][:]
        except KeyError:
            # Fallback: parse year from the 'start_time' global attribute
            start_time = f.attrs.get("start_time", b"2026").decode("utf-8")
            year_val = int(start_time[:4])
            year = np.full(f["Month"].shape, year_val)
            
        month = f["Month"][:]
        day = f["Day"][:]
        hour = f["Hour"][:]
        minute = f["Minute"][:]
        second = f["Second"][:]
        ms = f["Millisecond"][:]
        
        # 2. Read spatial coordinates
        lat = f["latitude"][:]
        lon = f["longitude"][:]

    # 3. Assemble datetime64 array using pandas
    time_df = pd.DataFrame({
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute,
        "second": second,
        "ms": ms
    })
    time_dt = pd.to_datetime(time_df).values.astype("datetime64[ns]")

    return {"time": time_dt, "latitude": lat, "longitude": lon}


def make_tms_vds(key: str) -> xr.Dataset:
    url = f"s3://{S3_BUCKET}/{key}"
    parser = HDFParser()
    
    with open_virtual_dataset(key, parser=parser, registry=registry, decode_times=False) as vds:
        # Extract actual coordinate values eagerly
        coords = read_tms_coords(key)
        
        # Determine correct dimension names based on the 3D shape 
        
        # Let's check what the dataset calls these axes natively
        dim_names = vds["latitude"].dims if "latitude" in vds else ("spots", "along_track_grid", "channels")
        
        if "scans" in vds.dims:
            vds = vds.rename_dims({"scans": "along_track_grid"})
            dim_names = tuple(d.replace("scans", "along_track_grid") for d in dim_names)
            
        if "time" in vds.variables: #fix time units
            vds["time"].attrs["units"] = "seconds since 2000-01-01 00:00:00"
            vds = vds.rename_vars({"time": "time_radiometric"})
            
        # Assign the eagerly read coordinates to the virtual dataset
        # We use the dynamic 'dim_names' tuple for lat and lon
        vds = vds.assign_coords(
            time=(
                "along_track_grid", 
                coords["time"],
                {"long_name": "observation time", "timezone": "UTC"}
            ),
            latitude=(
                dim_names, 
                coords["latitude"],
                {"long_name": "latitude", "units": "degrees_north"}
            ),
            longitude=(
                dim_names, 
                coords["longitude"],
                {"long_name": "longitude", "units": "degrees_east"}
            ),
        )
    
    return vds
# Return file keys (without reading)
def list_s3_netcdf_keys(bucket_name: str,
                        prefix: str) -> List[str]:
    """
    Lists all netCDF file keys in the specified S3 path without reading them.
    
    Parameters:
    -----------
    bucket_name : str
        Name of the S3 bucket
    prefix : str
        S3 prefix/path to search for netCDF files
        
    Returns:
    --------
    list
        List of S3 keys for netCDF files
    """
    s3_client = boto3.client('s3')
    netcdf_keys = []
    
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.lower().endswith(('.nc', '.nc4', '.netcdf','.h5')):
                    netcdf_keys.append(f's3://{bucket_name}/{key}')
    
    return netcdf_keys

keys = list_s3_netcdf_keys(bucket_name = S3_BUCKET, prefix = S3_PREFIX)
print(f"Found {len(keys)} netCDF files:")
vds_list = [make_tms_vds(k) for k in keys]

def extract_granule_id(key: str) -> str:
    """Extracts the start time (e.g., ST20260217-014358) to use as a group name."""
    m = re.search(r"(ST\d{8}-\d{6})", key)
    return m.group(1) if m else f"granule_{hash(key)}"

print(f"Writing tms ({len(keys)} granules)…")
config = icechunk.RepositoryConfig.default()
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(
        f"s3://{S3_BUCKET}/",
        icechunk.s3_store(region=S3_REGION),
    )
)
storage = s3_storage(
    bucket=S3_BUCKET,
    prefix=S3_ICECHUNK,
    region=S3_REGION,
    from_env=True
)


print("Writing virtual references iteratively to Icechunk...")
repository = Repository.open_or_create(storage, config=config)
for i, key in enumerate(keys):
    # 1. Create the virtual dataset
    vds = make_tms_vds(key)
    
    # 2. Open a writable session
    session = repository.writable_session("main")
    
    # 3. Put each granule in its own isolated sub-group
    gran_id = extract_granule_id(key)
    group_name = f"tms_l1c_tc/{gran_id}"
    
    # 4. Write directly to Icechunk (no append_dim needed!)
    virtual_dataset_to_icechunk(vds, session.store, group=group_name)
    
    # 5. Commit the session
    session.commit(f"Virtualize TMS {gran_id} ({i+1}/{len(keys)})")
    
    print(f"  {i+1}/{len(keys)} committed")

print("TMS virtualization done.")
