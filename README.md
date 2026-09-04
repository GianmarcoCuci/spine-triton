# ICARUS SPINE on Triton

ICARUS SPINE 0.16 served through the Triton 23.07 Python backend.
Run the commands below in a Bash shell from the repository root.

## Requirements

- Docker with BuildKit.
- NVIDIA Container Toolkit.
- Python 3 for the Python client, or a C++17 compiler for the C++ client.

## Build

```bash
docker build -t spine-triton:23.07-spine0.16.0-full-chain .
docker volume create spine-weights-v016
```

## Start the server

Use `full` for inputs with both cryostats. For cryoE-only simulation, replace
`full` with `single` in both the server and export commands.

```bash
docker run --rm \
  --gpus all \
  --shm-size=4g \
  --name spine-triton \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -e SPINE_ICARUS_PROFILE=full \
  -v "$PWD/model_repository:/models:ro" \
  -v spine-weights-v016:/model-cache \
  spine-triton:23.07-spine0.16.0-full-chain \
  --model-repository=/models \
  --model-control-mode=explicit \
  --load-model=spine_icarus_full_chain \
  --strict-readiness=true
```

The checkpoint is downloaded on first use and kept in `spine-weights-v016`.
Initialization can take several minutes. Check readiness from another shell:

```bash
docker logs spine-triton
curl --fail http://localhost:8000/v2/health/ready
```

Stop the server with `docker stop spine-triton`.


## Prepare an input event

ROOT input files are not included. Set `input_file` to an existing LArCV file:

```bash
input_file=/absolute/path/to/icarus.root

docker run --rm \
  -v "$PWD:/workspace/project" \
  -v "$input_file:/data/input.root:ro" \
  --entrypoint python3 \
  spine-triton:23.07-spine0.16.0-full-chain \
  /workspace/project/export_larcv_to_npz.py \
  /data/input.root \
  /workspace/project/data/icarus_event0.npz \
  --entry 0 --profile full
```

If the file has no optical flash products, add `--no-flashes` to the export
command and use `--skip-flash-match` with either client.

## Python client

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy h5py "tritonclient[http]"
python client.py data/icarus_event0.npz
```

The output is `data/icarus_event0_triton.h5`. Use `--url HOST:8000` for a remote
server, `-o PATH.h5` to choose the output, and `--overwrite` to replace an
existing file. All options are listed by `python client.py --help`.

## C++ client

`client.cpp` is an alternative to the Python client and also writes HDF5.
It requires the Triton 23.07 C++ HTTP client, libzip, libcurl, and the HDF5 C++
development libraries.

Example build for Debian/Ubuntu on x86-64. Set `TRITON_CLIENT_ROOT` to the
installation containing `include/http_client.h` and `lib/libhttpclient.so`;
adjust the HDF5 paths on other systems.

```bash
TRITON_CLIENT_ROOT=/path/to/triton-client-install

g++ -std=c++17 -O2 client.cpp -o client_cpp \
  -I"$TRITON_CLIENT_ROOT/include" \
  -I/usr/include/hdf5/serial \
  -L"$TRITON_CLIENT_ROOT/lib" \
  -L/usr/lib/x86_64-linux-gnu/hdf5/serial \
  -Wl,-rpath,"$TRITON_CLIENT_ROOT/lib" \
  -lhttpclient -lzip -lcurl -lhdf5_cpp -lhdf5 -pthread

./client_cpp data/icarus_event0.npz -o data/icarus_event0_cpp.h5
```

The C++ client accepts the same connection, FlashMatch, and output options as
the Python client. Run `./client_cpp --help` for the full list.
