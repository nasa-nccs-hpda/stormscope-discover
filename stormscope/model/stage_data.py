#!/usr/bin/env python3
"""Stage StormScope example inputs from public NOAA S3 buckets into local storage.

This script is designed for workflows like:
1) run this on a machine that can reach public cloud object storage
2) copy the staged `/data` tree to an offline HPC environment
3) run inference using local files

Default settings match the Earth2Studio StormScope GOES+MRMS example timestamp:
`2023-12-05T12:00:00Z` with 60-minute model inputs (lead 0 only).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Iterable

import boto3
from botocore import UNSIGNED
from botocore.config import Config


GOES_CUTOVER_UTC = dt.datetime(2025, 4, 7, tzinfo=dt.timezone.utc)
MRMS_PRODUCT_REFLECTIVITY = "MergedReflectivityQCComposite_00.50"

# Minimal default mapping needed for the example. Extend as needed.
GFS_VAR_TO_IDX_PATTERN = {
    "z500": "HGT::500 mb",
}


def parse_utc_timestamp(value: str) -> dt.datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    ts = dt.datetime.fromisoformat(value)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


def parse_lead_minutes(csv_text: str) -> list[int]:
    out = []
    for token in csv_text.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("No lead minutes were provided.")
    return sorted(set(out))


def choose_goes_satellite(init_time_utc: dt.datetime, mode: str) -> str:
    if mode != "auto":
        return mode
    return "goes16" if init_time_utc < GOES_CUTOVER_UTC else "goes19"


def s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_full_object(client, bucket: str, key: str, out_path: Path) -> None:
    ensure_parent(out_path)
    client.download_file(bucket, key, str(out_path))


def download_range_object(
    client, bucket: str, key: str, start: int, length: int, out_path: Path
) -> None:
    ensure_parent(out_path)
    end = start + length - 1
    resp = client.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")
    out_path.write_bytes(resp["Body"].read())


def iter_s3_keys(client, bucket: str, prefix: str) -> Iterable[str]:
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            yield obj["Key"]
        if not resp.get("IsTruncated"):
            return
        token = resp.get("NextContinuationToken")


def resolve_goes_key(
    client, satellite: str, scan_mode: str, target_time_utc: dt.datetime
) -> str:
    bucket = f"noaa-{satellite}"
    day_of_year = target_time_utc.timetuple().tm_yday
    prefix = (
        f"ABI-L2-MCMIP{scan_mode}/{target_time_utc.year:04d}/{day_of_year:03d}/"
        f"{target_time_utc.hour:02d}/"
    )
    keys = list(iter_s3_keys(client, bucket, prefix))
    wanted = f"OR_ABI-L2-MCMIP{scan_mode}"
    candidates = [k for k in keys if wanted in k]
    if not candidates:
        raise FileNotFoundError(f"No GOES objects for prefix s3://{bucket}/{prefix}")

    def start_time_from_key(key: str) -> dt.datetime:
        token = key.split("/")[-1].split("_")[-3]  # like sYYYYJJJHHMMSSx
        stamp = token[1:-1]
        ts = dt.datetime.strptime(stamp, "%Y%j%H%M%S")
        return ts.replace(tzinfo=dt.timezone.utc)

    return min(candidates, key=lambda k: abs(start_time_from_key(k) - target_time_utc))


def resolve_mrms_key(
    client,
    product: str,
    target_time_utc: dt.datetime,
    tolerance_minutes: float,
) -> str:
    bucket = "noaa-mrms-pds"
    t_min = target_time_utc - dt.timedelta(minutes=tolerance_minutes)
    t_max = target_time_utc + dt.timedelta(minutes=tolerance_minutes)
    day_set = {
        t_min.strftime("%Y%m%d"),
        target_time_utc.strftime("%Y%m%d"),
        t_max.strftime("%Y%m%d"),
    }
    pattern = re.compile(rf"^MRMS_{re.escape(product)}_(\d{{8}})-(\d{{6}})\.grib2\.gz$")

    best = None
    for ymd in sorted(day_set):
        prefix = f"CONUS/{product}/{ymd}/"
        for key in iter_s3_keys(client, bucket, prefix):
            filename = key.split("/")[-1]
            m = pattern.match(filename)
            if not m:
                continue
            ymd_part, hms_part = m.groups()
            ts = dt.datetime.strptime(ymd_part + hms_part, "%Y%m%d%H%M%S").replace(
                tzinfo=dt.timezone.utc
            )
            delta = abs((ts - target_time_utc).total_seconds())
            if delta > tolerance_minutes * 60:
                continue
            rank = (delta, ts, key)
            if best is None or rank < best:
                best = rank

    if best is None:
        raise FileNotFoundError(
            "No MRMS object found within "
            f"+/-{tolerance_minutes} minutes for {target_time_utc.isoformat()}"
        )
    return best[2]


def gfs_file_keys(cycle_utc: dt.datetime, lead_hour: int) -> tuple[str, str]:
    day_dir = f"gfs.{cycle_utc.year:04d}{cycle_utc.month:02d}{cycle_utc.day:02d}"
    hh_dir = f"{cycle_utc.hour:02d}"
    base = (
        f"{day_dir}/{hh_dir}/atmos/"
        f"gfs.t{cycle_utc.hour:02d}z.pgrb2.0p25.f{lead_hour:03d}"
    )
    return f"{base}.idx", base


def parse_gfs_idx(idx_text: str) -> dict[str, tuple[int, int]]:
    lines = [ln.strip() for ln in idx_text.splitlines() if ln.strip()]
    table: dict[str, tuple[int, int]] = {}
    for i in range(len(lines) - 1):
        left = lines[i].split(":")
        right = lines[i + 1].split(":")
        if len(left) < 7 or len(right) < 2:
            continue
        start = int(left[1])
        end = int(right[1])
        length = end - start
        key = f"{left[0]}::{left[3]}::{left[4]}"
        table[key] = (start, length)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init-time",
        default="2023-12-05T12:00:00Z",
        help="Initialization time in ISO-8601 (default: %(default)s)",
    )
    parser.add_argument(
        "--data-root",
        default="/data",
        help="Root output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--goes-satellite",
        default="auto",
        choices=["auto", "goes16", "goes17", "goes18", "goes19"],
        help="GOES platform (default: %(default)s)",
    )
    parser.add_argument(
        "--scan-mode",
        default="C",
        choices=["C", "F"],
        help="GOES scan mode (default: %(default)s)",
    )
    parser.add_argument(
        "--lead-minutes",
        default="0",
        help="Comma-separated lead minutes to stage (default: %(default)s)",
    )
    parser.add_argument(
        "--mrms-product",
        default=MRMS_PRODUCT_REFLECTIVITY,
        help="MRMS product key (default: %(default)s)",
    )
    parser.add_argument(
        "--mrms-tolerance-minutes",
        type=float,
        default=10.0,
        help="Nearest-file tolerance for MRMS matching (default: %(default)s)",
    )
    parser.add_argument(
        "--gfs-vars",
        default="z500",
        help="Comma-separated GFS variables to stage (default: %(default)s)",
    )
    parser.add_argument(
        "--manifest",
        default="stormscope_stage_manifest.json",
        help="Manifest filename under data root (default: %(default)s)",
    )
    args = parser.parse_args()

    init_time = parse_utc_timestamp(args.init_time)
    lead_minutes = parse_lead_minutes(args.lead_minutes)
    gfs_vars = [v.strip() for v in args.gfs_vars.split(",") if v.strip()]
    unknown = [v for v in gfs_vars if v not in GFS_VAR_TO_IDX_PATTERN]
    if unknown:
        raise ValueError(
            f"Unsupported gfs vars: {unknown}. Extend GFS_VAR_TO_IDX_PATTERN in stage_data.py."
        )

    data_root = Path(args.data_root)
    satellite = choose_goes_satellite(init_time, args.goes_satellite)
    client = s3_client()
    manifest: dict = {
        "init_time_utc": init_time.isoformat(),
        "lead_minutes": lead_minutes,
        "goes_satellite": satellite,
        "scan_mode": args.scan_mode,
        "files": [],
    }

    print(f"Staging to: {data_root}")
    print(f"Init time (UTC): {init_time.isoformat()}")
    print(f"GOES satellite: {satellite}")
    print(f"Lead minutes: {lead_minutes}")

    for lead_min in lead_minutes:
        target_time = init_time + dt.timedelta(minutes=lead_min)
        print(f"\n[lead {lead_min:+d} min] target={target_time.isoformat()}")

        # GOES
        goes_bucket = f"noaa-{satellite}"
        goes_key = resolve_goes_key(client, satellite, args.scan_mode, target_time)
        goes_local = data_root / goes_bucket / goes_key
        print(f"  GOES : s3://{goes_bucket}/{goes_key}")
        download_full_object(client, goes_bucket, goes_key, goes_local)
        manifest["files"].append(
            {
                "source": "goes",
                "bucket": goes_bucket,
                "key": goes_key,
                "local_path": str(goes_local),
            }
        )

        # MRMS
        mrms_bucket = "noaa-mrms-pds"
        mrms_key = resolve_mrms_key(
            client,
            args.mrms_product,
            target_time,
            tolerance_minutes=args.mrms_tolerance_minutes,
        )
        mrms_local = data_root / mrms_bucket / mrms_key
        print(f"  MRMS : s3://{mrms_bucket}/{mrms_key}")
        download_full_object(client, mrms_bucket, mrms_key, mrms_local)
        manifest["files"].append(
            {
                "source": "mrms",
                "bucket": mrms_bucket,
                "key": mrms_key,
                "local_path": str(mrms_local),
            }
        )

        # GFS byte-ranges from pgrb2 + idx
        if init_time.minute != 0 or init_time.second != 0:
            raise ValueError("GFS cycle must be on the hour. Use an exact cycle init time.")
        if init_time.hour % 6 != 0:
            raise ValueError("GFS cycle must be a 6-hour cycle: 00, 06, 12, or 18 UTC.")

        gfs_bucket = "noaa-gfs-bdp-pds"
        lead_hour = int(lead_min // 60)
        idx_key, grib_key = gfs_file_keys(init_time, lead_hour)
        idx_local = data_root / gfs_bucket / idx_key
        print(f"  GFS  : s3://{gfs_bucket}/{idx_key}")
        download_full_object(client, gfs_bucket, idx_key, idx_local)

        idx_text = idx_local.read_text(encoding="utf-8")
        idx_table = parse_gfs_idx(idx_text)
        for var in gfs_vars:
            pattern = GFS_VAR_TO_IDX_PATTERN[var]
            match = next((k for k in idx_table if pattern in k), None)
            if match is None:
                raise KeyError(
                    f"GFS variable '{var}' pattern '{pattern}' not found in {idx_key}"
                )
            start, length = idx_table[match]
            var_suffix = var.replace("/", "_")
            out_key = f"{grib_key}.{var_suffix}.grib2"
            out_local = data_root / gfs_bucket / out_key
            print(
                f"    - {var}: s3://{gfs_bucket}/{grib_key} bytes={start}-{start + length - 1}"
            )
            download_range_object(client, gfs_bucket, grib_key, start, length, out_local)
            manifest["files"].append(
                {
                    "source": "gfs",
                    "bucket": gfs_bucket,
                    "key": grib_key,
                    "idx_key": idx_key,
                    "var": var,
                    "idx_match": match,
                    "byte_start": start,
                    "byte_length": length,
                    "local_path": str(out_local),
                }
            )

    manifest_path = data_root / args.manifest
    ensure_parent(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote manifest: {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()
