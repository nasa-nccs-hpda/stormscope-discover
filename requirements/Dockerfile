FROM nvcr.io/nvidia/pytorch:25.02-py3

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
      "earth2studio @ git+https://github.com/NVIDIA/earth2studio.git" \
      matplotlib cartopy xarray netcdf4 h5netcdf

WORKDIR /workspace
CMD ["python", "--version"]
