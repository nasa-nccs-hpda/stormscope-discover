"""simple_inference.py
Run StormScope GOES-only iterative inference using a local GOES NetCDF file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from earth2studio.data import DataArrayFile, GFS_FX, fetch_data
from earth2studio.models.px.stormscope import StormScopeBase, StormScopeGOES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run StormScope GOES-only inference from local NetCDF input."
    )
    parser.add_argument(
        "--goes-input",
        type=str,
        required=True,
        help="Path to local GOES input NetCDF file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stormscope_goes_forecast.nc",
        help="Output NetCDF file.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="6km_60min_natten_cos_zenith_input_eoe_v2",
        help="StormScope GOES model name.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2,
        help="Number of autoregressive forecast steps.",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="Which time entry from the local file to use as initialization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    goes_path = Path(args.goes_input)
    if not goes_path.exists():
        raise FileNotFoundError(f"GOES input file not found: {goes_path}")

    # Load GOES model exactly in the same spirit as the Earth2Studio example:
    # GOES model + GFS_FX conditioning for the 1-hour GOES model.
    package = StormScopeBase.load_default_package()
    model = StormScopeGOES.load_model(
        package=package,
        conditioning_data_source=GFS_FX(),
        model_name=args.model_name,
    ).to(device)
    model.eval()

    # Local GOES file as Earth2Studio data source
    goes_local = DataArrayFile(str(goes_path))

    # Model-required coordinates
    in_coords = model.input_coords()
    variables = np.array(in_coords["variable"])

    # Pick init time from local file
    available_times = goes_local.get_times()
    if len(available_times) == 0:
        raise ValueError("No time coordinate found in local GOES file.")

    if args.time_index < 0 or args.time_index >= len(available_times):
        raise IndexError(
            f"time-index {args.time_index} out of range; file has {len(available_times)} time values."
        )

    start_time = [available_times[args.time_index]]
    print(f"Initialization time: {start_time[0]}")

    # Build interpolators.
    # For a local GOES file, we use its lat/lon grid as the input grid.
    sample_da = goes_local(
        time=start_time,
        variable=np.array([variables[0]]),
        lead_time=in_coords["lead_time"],
    )

    input_lat = sample_da.coords["lat"].values
    input_lon = sample_da.coords["lon"].values

    model.build_input_interpolator(input_lat, input_lon)
    model.build_conditioning_interpolator(GFS_FX.GFS_LAT, GFS_FX.GFS_LON)

    # Fetch local GOES data through Earth2Studio helper so shapes/coords match model workflow
    x, x_coords = fetch_data(
        goes_local,
        time=start_time,
        variable=variables,
        lead_time=in_coords["lead_time"],
        device=device,
    )

    # Add batch dimension: expected shape [B, T, L, C, H, W]
    if x.dim() == 5:
        x = x.unsqueeze(0)
        x_coords["batch"] = np.arange(1)
        x_coords.move_to_end("batch", last=False)

    x = x.to(dtype=torch.float32)

    print(f"Input tensor shape: {tuple(x.shape)}")
    print(f"Variables: {list(variables)}")
    print(f"Lead times: {x_coords['lead_time']}")

    # Iterative GOES-only inference loop
    y, y_coords = x, x_coords
    forecast_frames = []
    forecast_coords = []

    with torch.no_grad():
        for step_idx in range(args.n_steps):
            print(f"Running forecast step {step_idx + 1}/{args.n_steps}")

            # One model step
            y_pred, y_pred_coords = model(y, y_coords)

            # Save raw prediction from this step
            forecast_frames.append(y_pred.detach().cpu())
            forecast_coords.append(y_pred_coords.copy())

            # Advance sliding input window for next step
            y, y_coords = model.next_input(y_pred, y_pred_coords, y, y_coords)

    # Concatenate along lead_time if possible
    # Each y_pred should contain one future lead block produced by the model.
    pred_xr_list = []
    for i, (pred_torch, coords) in enumerate(zip(forecast_frames, forecast_coords)):
        pred_np = pred_torch.numpy()

        dims = list(coords.keys())
        pred_da = xr.DataArray(pred_np, dims=dims, coords=coords, name="stormscope_goes")
        pred_da = pred_da.assign_coords(forecast_step=i + 1).expand_dims("forecast_step")
        pred_xr_list.append(pred_da)

    out_da = xr.concat(pred_xr_list, dim="forecast_step")

    # Mask invalid grid cells if available
    if hasattr(model, "valid_mask") and model.valid_mask is not None:
        valid_mask = model.valid_mask.detach().cpu().numpy()
        # Broadcast mask over non-spatial dims
        out_da = out_da.where(valid_mask)

    out_da.to_netcdf(args.output)
    print(f"Saved forecast to: {args.output}")


if __name__ == "__main__":
    main()