#!/usr/bin/env python3
"""HTTP client for the ICARUS SPINE full-chain Triton model.

The client reads an event exported to a numeric NPZ archive, checks that the
Triton server and model are ready, sends one inference request, validates all
18 response tensors, prints a compact result summary, and saves the complete
response as an HDF5 file.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# Defaults used by the validated local deployment.
DEFAULT_URL = "localhost:8000"
DEFAULT_MODEL_NAME = "spine_icarus_full_chain"
DEFAULT_MODEL_VERSION = "1"
HDF5_SUFFIXES = {".h5", ".hdf5"}


@dataclass(frozen=True)
class InputSpec:
    """Description of one array in the NPZ-to-Triton mapping."""

    archive_name: str
    triton_name: str
    numpy_dtype: np.dtype[Any]
    triton_dtype: str
    rank: int
    columns: int | None = None


# Required charge and event-description tensors.
INPUT_SPECS = (
    InputSpec("coordinates", "COORDINATES", np.dtype(np.int32), "INT32", 2, 4),
    InputSpec("features", "FEATURES", np.dtype(np.float32), "FP32", 2, 8),
    InputSpec("sources", "SOURCES", np.dtype(np.int32), "INT32", 2, 2),
    InputSpec("counts", "COUNTS", np.dtype(np.int32), "INT32", 1),
    InputSpec("meta", "META", np.dtype(np.float32), "FP32", 2, 12),
    InputSpec("run_info", "RUN_INFO", np.dtype(np.int64), "INT64", 2, 3),
)

# Optical information is optional.
FLASH_INPUT_SPECS = (
    InputSpec("flash_data", "FLASH_DATA", np.dtype(np.float32), "FP32", 2, 16),
    InputSpec("flash_pe", "FLASH_PE", np.dtype(np.float32), "FP32", 2, 180),
    InputSpec("flash_counts", "FLASH_COUNTS", np.dtype(np.int32), "INT32", 1),
)

# Request outputs explicitly. This makes a server-side contract change fail
# visibly instead of producing an incomplete result file.
OUTPUT_NAMES = (
    "EVENT_ID",
    "GHOST_PRED",
    "GHOST_COUNTS",
    "SEGMENTATION",
    "SEGMENTATION_COUNTS",
    "ORIG_INDEX",
    "POINTS",
    "DEPOSITIONS",
    "DEGHOSTED_SOURCES",
    "PARTICLES",
    "PARTICLE_COUNTS",
    "PARTICLE_VOXELS",
    "PARTICLE_VOXEL_COUNTS",
    "INTERACTIONS",
    "INTERACTION_COUNTS",
    "INTERACTION_FLASHES",
    "INTERACTION_FLASH_COUNTS",
    "FLASH_MATCH_RAN",
)

# output name -> (dtype, rank, fixed number of columns for rank-2 arrays)
OUTPUT_SPECS: dict[str, tuple[np.dtype[Any], int, int | None]] = {
    "EVENT_ID": (np.dtype(np.int64), 2, 3),
    "GHOST_PRED": (np.dtype(np.int64), 1, None),
    "GHOST_COUNTS": (np.dtype(np.int32), 1, None),
    "SEGMENTATION": (np.dtype(np.int64), 1, None),
    "SEGMENTATION_COUNTS": (np.dtype(np.int32), 1, None),
    "ORIG_INDEX": (np.dtype(np.int64), 1, None),
    "POINTS": (np.dtype(np.float32), 2, 3),
    "DEPOSITIONS": (np.dtype(np.float32), 1, None),
    "DEGHOSTED_SOURCES": (np.dtype(np.int32), 2, 2),
    "PARTICLES": (np.dtype(np.float32), 2, 41),
    "PARTICLE_COUNTS": (np.dtype(np.int32), 1, None),
    "PARTICLE_VOXELS": (np.dtype(np.int64), 2, 3),
    "PARTICLE_VOXEL_COUNTS": (np.dtype(np.int32), 1, None),
    "INTERACTIONS": (np.dtype(np.float32), 2, 29),
    "INTERACTION_COUNTS": (np.dtype(np.int32), 1, None),
    "INTERACTION_FLASHES": (np.dtype(np.float32), 2, 6),
    "INTERACTION_FLASH_COUNTS": (np.dtype(np.int32), 1, None),
    "FLASH_MATCH_RAN": (np.dtype(np.int32), 1, None),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Query the ICARUS SPINE full-chain model through Triton HTTP and "
            "save all returned tensors in an HDF5 file."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input NPZ produced by export_larcv_to_npz.py.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Output HDF5 path. By default, INPUT.npz becomes "
            "INPUT_triton.h5."
        ),
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Triton HTTP endpoint (default: {DEFAULT_URL}).",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Triton model name (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--model-version",
        default=DEFAULT_MODEL_VERSION,
        help=f"Triton model version (default: {DEFAULT_MODEL_VERSION}).",
    )
    parser.add_argument(
        "--skip-flash-match",
        action="store_true",
        help="Omit FLASH_DATA, FLASH_PE, and FLASH_COUNTS from the request.",
    )
    parser.add_argument(
        "--connection-timeout",
        type=float,
        default=10.0,
        help="HTTP connection timeout in seconds (default: 10).",
    )
    parser.add_argument(
        "--network-timeout",
        type=float,
        default=3600.0,
        help="HTTP network timeout in seconds (default: 3600).",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Show up to N values/rows from every output tensor; use 0 to "
            "disable previews (default: 2)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow an existing output HDF5 file to be replaced.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose Triton HTTP client logging.",
    )
    args = parser.parse_args()

    if args.connection_timeout <= 0:
        parser.error("--connection-timeout must be greater than zero")
    if args.network_timeout <= 0:
        parser.error("--network-timeout must be greater than zero")
    if args.preview < 0:
        parser.error("--preview cannot be negative")

    return args


def default_output_path(input_path: Path) -> Path:
    """Place the result next to the input without overwriting the request."""

    if input_path.suffix.lower() == ".npz":
        return input_path.with_name(f"{input_path.stem}_triton.h5")
    return input_path.with_name(f"{input_path.name}_triton.h5")


def normalize_output_path(path: Path) -> Path:
    """Add the default HDF5 suffix or validate the supplied suffix."""

    if not path.suffix:
        return Path(f"{path}.h5")
    if path.suffix.lower() not in HDF5_SUFFIXES:
        raise ValueError("The output file must use the .h5 or .hdf5 extension.")
    return path


def validate_tensor(name: str, array: np.ndarray, spec: InputSpec) -> None:
    """Check the fixed part of an input tensor contract."""

    if array.dtype != spec.numpy_dtype:
        raise ValueError(
            f"Input '{name}' has dtype {array.dtype}; expected {spec.numpy_dtype}."
        )
    if array.ndim != spec.rank:
        raise ValueError(
            f"Input '{name}' has rank {array.ndim}; expected rank {spec.rank}."
        )
    if spec.columns is not None and array.shape[1] != spec.columns:
        raise ValueError(
            f"Input '{name}' has shape {array.shape}; expected "
            f"(*, {spec.columns})."
        )


def load_inputs(
    input_path: Path,
    skip_flash_match: bool,
) -> tuple[dict[str, np.ndarray], bool]:
    """Load the request archive and decide whether to send optical data."""

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    required_names = {spec.archive_name for spec in INPUT_SPECS}
    flash_names = {spec.archive_name for spec in FLASH_INPUT_SPECS}

    # Triton only receives numeric tensors, so object/pickle loading is disabled.
    with np.load(input_path, allow_pickle=False) as archive:
        available = set(archive.files)
        missing = sorted(required_names - available)
        if missing:
            raise ValueError(
                "Input NPZ is missing required arrays: " + ", ".join(missing)
            )

        present_flash_names = flash_names & available
        if not skip_flash_match and present_flash_names not in (set(), flash_names):
            missing_flash = sorted(flash_names - present_flash_names)
            raise ValueError(
                "The optical inputs are an all-or-none group. Missing: "
                + ", ".join(missing_flash)
            )

        use_flash_match = not skip_flash_match and present_flash_names == flash_names
        active_specs = INPUT_SPECS + (FLASH_INPUT_SPECS if use_flash_match else ())

        arrays: dict[str, np.ndarray] = {}
        for spec in active_specs:
            array = np.asarray(archive[spec.archive_name])
            validate_tensor(spec.archive_name, array, spec)
            # Binary HTTP transport expects a flat, contiguous memory buffer.
            arrays[spec.archive_name] = np.ascontiguousarray(array)

    validate_input_relationships(arrays, use_flash_match)
    return arrays, use_flash_match


def validate_input_relationships(
    arrays: dict[str, np.ndarray],
    use_flash_match: bool,
) -> None:
    """Check row counts that depend on more than one input array."""

    coordinates = arrays["coordinates"]
    features = arrays["features"]
    sources = arrays["sources"]
    counts = arrays["counts"]
    meta = arrays["meta"]
    run_info = arrays["run_info"]

    if np.any(counts < 0):
        raise ValueError("Input 'counts' cannot contain negative values.")

    voxel_count = coordinates.shape[0]
    event_count = counts.shape[0]
    if features.shape[0] != voxel_count or sources.shape[0] != voxel_count:
        raise ValueError(
            "coordinates, features, and sources must have the same row count."
        )
    if int(counts.sum(dtype=np.int64)) != voxel_count:
        raise ValueError(
            f"sum(counts) is {int(counts.sum(dtype=np.int64))}, but there are "
            f"{voxel_count} coordinate rows."
        )
    if meta.shape[0] != event_count or run_info.shape[0] != event_count:
        raise ValueError("counts, meta, and run_info must describe the same events.")

    # The coordinate event ID is local to this request. Run/subrun/event values
    # are carried separately in run_info.
    expected_event_ids = np.repeat(
        np.arange(event_count, dtype=np.int32),
        counts.astype(np.int64, copy=False),
    )
    if not np.array_equal(coordinates[:, 0], expected_event_ids):
        raise ValueError(
            "The first COORDINATES column must contain contiguous request-local "
            "event IDs 0, 1, ... according to counts."
        )

    if use_flash_match:
        flash_data = arrays["flash_data"]
        flash_pe = arrays["flash_pe"]
        flash_counts = arrays["flash_counts"]
        flash_count = flash_data.shape[0]
        if flash_pe.shape[0] != flash_count:
            raise ValueError("flash_data and flash_pe must have the same row count.")
        if flash_counts.shape != counts.shape:
            raise ValueError("flash_counts must contain one value per event.")
        if np.any(flash_counts < 0):
            raise ValueError("Input 'flash_counts' cannot contain negative values.")
        if int(flash_counts.sum(dtype=np.int64)) != flash_count:
            raise ValueError(
                "sum(flash_counts) must equal the number of optical-flash rows."
            )


def build_request_inputs(
    httpclient: Any,
    arrays: dict[str, np.ndarray],
    use_flash_match: bool,
) -> list[Any]:
    """Wrap NumPy arrays in Triton HTTP input objects."""

    active_specs = INPUT_SPECS + (FLASH_INPUT_SPECS if use_flash_match else ())
    request_inputs = []
    for spec in active_specs:
        array = arrays[spec.archive_name]
        infer_input = httpclient.InferInput(
            spec.triton_name,
            list(array.shape),
            spec.triton_dtype,
        )
        infer_input.set_data_from_numpy(array, binary_data=True)
        request_inputs.append(infer_input)
    return request_inputs


def collect_outputs(result: Any) -> dict[str, np.ndarray]:
    """Extract every requested response tensor as a NumPy array."""

    arrays: dict[str, np.ndarray] = {}
    for name in OUTPUT_NAMES:
        array = result.as_numpy(name)
        if array is None:
            raise RuntimeError(f"Triton response is missing output '{name}'.")
        arrays[name] = np.ascontiguousarray(array)
    return arrays


def validate_output_tensor(name: str, array: np.ndarray) -> None:
    """Check the dtype and fixed dimensions of one response tensor."""

    expected_dtype, expected_rank, expected_columns = OUTPUT_SPECS[name]
    if array.dtype != expected_dtype:
        raise RuntimeError(
            f"Output '{name}' has dtype {array.dtype}; expected {expected_dtype}."
        )
    if array.ndim != expected_rank:
        raise RuntimeError(
            f"Output '{name}' has rank {array.ndim}; expected {expected_rank}."
        )
    if expected_columns is not None and array.shape[1] != expected_columns:
        raise RuntimeError(
            f"Output '{name}' has shape {array.shape}; expected "
            f"(*, {expected_columns})."
        )


def require_equal(actual: int, expected: int, description: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{description}: got {actual}, expected {expected}.")


def validate_outputs(
    outputs: dict[str, np.ndarray],
    inputs: dict[str, np.ndarray],
    use_flash_match: bool,
) -> None:
    """Validate relationships between the variable-length response arrays."""

    for name, array in outputs.items():
        validate_output_tensor(name, array)

    event_count = inputs["counts"].shape[0]
    input_voxel_count = inputs["coordinates"].shape[0]
    deghosted_count = outputs["SEGMENTATION"].shape[0]
    particle_count = outputs["PARTICLES"].shape[0]
    membership_count = outputs["PARTICLE_VOXELS"].shape[0]
    interaction_count = outputs["INTERACTIONS"].shape[0]
    flash_association_count = outputs["INTERACTION_FLASHES"].shape[0]

    # Event identity and the original-voxel bookkeeping should be unchanged by
    # inference.
    if outputs["EVENT_ID"].shape != (event_count, 3):
        raise RuntimeError("EVENT_ID has an invalid shape.")
    if not np.array_equal(outputs["EVENT_ID"], inputs["run_info"]):
        raise RuntimeError("EVENT_ID does not match the request RUN_INFO.")

    require_equal(outputs["GHOST_PRED"].shape[0], input_voxel_count, "GHOST_PRED rows")
    if outputs["GHOST_COUNTS"].shape != (event_count,):
        raise RuntimeError("GHOST_COUNTS must contain one value per event.")
    if not np.array_equal(outputs["GHOST_COUNTS"], inputs["counts"]):
        raise RuntimeError("GHOST_COUNTS does not match the request COUNTS.")

    # All deghosted arrays must refer to the same M retained voxels.
    if outputs["SEGMENTATION_COUNTS"].shape != (event_count,):
        raise RuntimeError("SEGMENTATION_COUNTS must contain one value per event.")
    require_equal(
        int(outputs["SEGMENTATION_COUNTS"].sum(dtype=np.int64)),
        deghosted_count,
        "sum(SEGMENTATION_COUNTS)",
    )
    require_equal(outputs["ORIG_INDEX"].shape[0], deghosted_count, "ORIG_INDEX rows")
    require_equal(outputs["POINTS"].shape[0], deghosted_count, "POINTS rows")
    require_equal(outputs["DEPOSITIONS"].shape[0], deghosted_count, "DEPOSITIONS rows")
    require_equal(
        outputs["DEGHOSTED_SOURCES"].shape[0],
        deghosted_count,
        "DEGHOSTED_SOURCES rows",
    )

    original_indices = outputs["ORIG_INDEX"]
    if np.any(original_indices < 0) or np.any(original_indices >= input_voxel_count):
        raise RuntimeError("ORIG_INDEX contains values outside the request row range.")
    if np.any(outputs["GHOST_PRED"][original_indices] != 0):
        raise RuntimeError("ORIG_INDEX points to one or more predicted ghost voxels.")

    # Particle summary rows and voxel-membership rows use separate offsets.
    if outputs["PARTICLE_COUNTS"].shape != (event_count,):
        raise RuntimeError("PARTICLE_COUNTS must contain one value per event.")
    require_equal(
        int(outputs["PARTICLE_COUNTS"].sum(dtype=np.int64)),
        particle_count,
        "sum(PARTICLE_COUNTS)",
    )
    require_equal(
        outputs["PARTICLE_VOXEL_COUNTS"].shape[0],
        particle_count,
        "PARTICLE_VOXEL_COUNTS rows",
    )
    require_equal(
        int(outputs["PARTICLE_VOXEL_COUNTS"].sum(dtype=np.int64)),
        membership_count,
        "sum(PARTICLE_VOXEL_COUNTS)",
    )

    # Flash associations are grouped by reconstructed interaction.
    if outputs["INTERACTION_COUNTS"].shape != (event_count,):
        raise RuntimeError("INTERACTION_COUNTS must contain one value per event.")
    require_equal(
        int(outputs["INTERACTION_COUNTS"].sum(dtype=np.int64)),
        interaction_count,
        "sum(INTERACTION_COUNTS)",
    )
    require_equal(
        outputs["INTERACTION_FLASH_COUNTS"].shape[0],
        interaction_count,
        "INTERACTION_FLASH_COUNTS rows",
    )
    require_equal(
        int(outputs["INTERACTION_FLASH_COUNTS"].sum(dtype=np.int64)),
        flash_association_count,
        "sum(INTERACTION_FLASH_COUNTS)",
    )

    if outputs["FLASH_MATCH_RAN"].shape != (event_count,):
        raise RuntimeError("FLASH_MATCH_RAN must contain one value per event.")
    expected_flash_flag = 1 if use_flash_match else 0
    if np.any(outputs["FLASH_MATCH_RAN"] != expected_flash_flag):
        raise RuntimeError(
            "FLASH_MATCH_RAN is inconsistent with the optical inputs sent."
        )


def compact_preview(array: np.ndarray, row_count: int) -> str:
    """Return a small preview without dumping wide particle rows."""

    if row_count == 0:
        return ""
    if array.ndim == 1:
        sample = array[:row_count]
    else:
        sample = array[:row_count, : min(array.shape[1], 8)]
    return np.array2string(sample, threshold=32, edgeitems=4)


def print_summary(
    outputs: dict[str, np.ndarray],
    elapsed_seconds: float,
    preview_rows: int,
) -> None:
    """Print the main reconstruction counts and the tensor inventory."""

    print("\nInference result")
    print(f"  Event IDs:           {outputs['EVENT_ID'].tolist()}")
    print(f"  Input voxels:        {int(outputs['GHOST_COUNTS'].sum())}")
    print(f"  Deghosted voxels:    {outputs['SEGMENTATION'].shape[0]}")
    print(f"  Particles:           {outputs['PARTICLES'].shape[0]}")
    print(f"  Interactions:        {outputs['INTERACTIONS'].shape[0]}")
    print(f"  FlashMatch ran:      {outputs['FLASH_MATCH_RAN'].tolist()}")
    print(f"  Flash associations: {outputs['INTERACTION_FLASHES'].shape[0]}")
    print(f"  Inference time:      {elapsed_seconds:.3f} s")

    print("\nReturned arrays")
    for name in OUTPUT_NAMES:
        array = outputs[name]
        print(f"  {name:27} shape={str(array.shape):14} dtype={array.dtype}")
        preview = compact_preview(array, preview_rows)
        if preview:
            print(f"    first {min(preview_rows, array.shape[0])}: {preview}")


def save_outputs(
    h5py: Any,
    output_path: Path,
    outputs: dict[str, np.ndarray],
    overwrite: bool,
) -> None:
    """Write each returned tensor as a dataset in one HDF5 file."""

    output_path = normalize_output_path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as output_file:
        for name in OUTPUT_NAMES:
            array = outputs[name]
            options = {}
            if array.size:
                options = {
                    "compression": "gzip",
                    "compression_opts": 4,
                    "shuffle": True,
                }
            output_file.create_dataset(name, data=array, **options)


def run() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    requested_output = args.output or default_output_path(input_path)
    output_path = normalize_output_path(requested_output.expanduser()).resolve()

    if output_path == input_path:
        raise ValueError("The output path must be different from the input path.")

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "Missing HDF5 support. Install it with: pip install h5py"
        ) from exc

    inputs, use_flash_match = load_inputs(input_path, args.skip_flash_match)

    # Import after parsing so that `client.py --help` still works before the
    # Triton client package is installed.
    try:
        import tritonclient.http as httpclient
        from tritonclient.utils import InferenceServerException
    except ImportError as exc:
        raise RuntimeError(
            'Missing Triton client. Install it with: pip install "tritonclient[http]"'
        ) from exc

    client = httpclient.InferenceServerClient(
        url=args.url,
        verbose=args.verbose,
        concurrency=1,
        connection_timeout=args.connection_timeout,
        network_timeout=args.network_timeout,
    )
    try:
        # Check readiness before preparing and transferring the request.
        if not client.is_server_live():
            raise RuntimeError(f"Triton server at {args.url} is not live.")
        if not client.is_server_ready():
            raise RuntimeError(f"Triton server at {args.url} is not ready.")
        if not client.is_model_ready(args.model_name, args.model_version):
            raise RuntimeError(
                f"Model {args.model_name!r}, version {args.model_version!r}, "
                "is not ready."
            )

        print(f"Connected: {args.url}")
        print(f"Model ready: {args.model_name}, version {args.model_version}")
        print(f"Input: {input_path}")
        print(f"Optical inputs: {'enabled' if use_flash_match else 'omitted'}")

        request_inputs = build_request_inputs(httpclient, inputs, use_flash_match)
        requested_outputs = [
            httpclient.InferRequestedOutput(name, binary_data=True)
            for name in OUTPUT_NAMES
        ]

        # Include HTTP transfer and server execution in the elapsed time.
        start = time.perf_counter()
        result = client.infer(
            model_name=args.model_name,
            model_version=args.model_version,
            inputs=request_inputs,
            outputs=requested_outputs,
        )
        elapsed = time.perf_counter() - start
    except InferenceServerException as exc:
        raise RuntimeError(f"Triton request failed: {exc}") from exc
    finally:
        client.close()

    outputs = collect_outputs(result)
    validate_outputs(outputs, inputs, use_flash_match)
    print_summary(outputs, elapsed, args.preview)
    save_outputs(h5py, output_path, outputs, args.overwrite)

    print(f"\nSaved: {output_path}")
    print("ICARUS SPINE Triton output contract: PASS")
    return 0


def main() -> int:
    try:
        return run()
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
