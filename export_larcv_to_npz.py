#!/usr/bin/env python3
"""Export one or more ICARUS LArCV entries to the Triton wire contract.

Run this script inside the SPINE runtime image, where both SPINE and LArCV are
available.  It deliberately uses the data/sources/meta/run_info/flash parsers
from the exact production configuration bundled with the Triton model.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from spine.config import load_config_file
from spine.geo import GeoManager
from spine.io.factories import dataset_factory


FLASH_WIDTH = 16
FLASH_PE_WIDTH = 180
CONFIG_PROFILES = {
    "full": Path("full_chain_co_250625.yaml"),
    "single": Path("legacy/icarus_full_chain_single_co_250625.yaml"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input ICARUS LArCV ROOT file")
    parser.add_argument("output", type=Path, help="Destination .npz file")
    parser.add_argument(
        "--entry",
        type=int,
        action="append",
        dest="entries",
        help="Entry index to export; repeat for a multi-event request (default: 0)",
    )
    parser.add_argument(
        "--no-flashes",
        action="store_true",
        help="Do not require or export opflash products",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(CONFIG_PROFILES),
        default="full",
        help=(
            "ICARUS input profile: full requires cryoE+cryoW; single uses the "
            "official cryoE-only simulation modifier (default: full)"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    entries = args.entries or [0]
    if any(entry < 0 for entry in entries):
        raise ValueError("Entry indexes must be non-negative")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    project_root = Path(__file__).resolve().parent
    config_root = (
        project_root
        / "model_repository"
        / "spine_icarus_full_chain"
        / "config"
    )
    config_path = config_root / CONFIG_PROFILES[args.profile]
    config = load_config_file(str(config_path), download=False)
    GeoManager.initialize_or_get(**config["geo"])

    dataset_config = deepcopy(config["io"]["loader"]["dataset"])
    wanted = ["data", "sources", "meta", "run_info"]
    if not args.no_flashes:
        wanted.append("flashes")
    dataset_config["schema"] = {
        key: dataset_config["schema"][key] for key in wanted
    }
    dataset_config["file_keys"] = [str(args.input.resolve())]
    dataset = dataset_factory(dataset_config, dtype="float32")

    coordinate_blocks = []
    feature_blocks = []
    source_blocks = []
    meta_rows = []
    run_rows = []
    counts = []
    flash_rows = []
    flash_pe_rows = []
    flash_counts = []

    for event_id, entry in enumerate(entries):
        if entry >= len(dataset):
            raise IndexError(
                f"Entry {entry} is outside the dataset range [0, {len(dataset) - 1}]"
            )
        sample = dataset[entry]
        data = sample["data"]
        sources = sample["sources"]

        coords = np.asarray(data.coords, dtype=np.int32)
        features = np.asarray(data.features, dtype=np.float32)
        source_features = np.asarray(sources.features, dtype=np.int32)
        source_coords = np.asarray(sources.coords, dtype=np.int32)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Parsed data coordinates have shape {coords.shape}")
        if features.shape != (len(coords), 8):
            raise ValueError(f"Parsed data features have shape {features.shape}")
        if source_features.shape != (len(coords), 2):
            raise ValueError(
                "Parsed sources are not aligned with data: "
                f"{source_features.shape} versus {len(coords)} voxels"
            )
        if not np.array_equal(source_coords, coords):
            raise ValueError(
                "Parsed source coordinates do not exactly match data coordinates"
            )

        coordinate_blocks.append(
            np.column_stack(
                [np.full(len(coords), event_id, dtype=np.int32), coords]
            )
        )
        feature_blocks.append(features)
        source_blocks.append(source_features)
        counts.append(len(coords))

        meta = sample["meta"]
        meta_rows.append(
            np.concatenate(
                [
                    np.asarray(meta.lower, dtype=np.float32),
                    np.asarray(meta.upper, dtype=np.float32),
                    np.asarray(meta.size, dtype=np.float32),
                    np.asarray(meta.count, dtype=np.float32),
                ]
            )
        )
        info = sample["run_info"]
        run_rows.append([info.run, info.subrun, info.event])

        flashes = [] if args.no_flashes else sample["flashes"]
        flash_counts.append(len(flashes))
        for flash in flashes:
            pe_per_ch = np.asarray(flash.pe_per_ch, dtype=np.float32)
            if pe_per_ch.shape != (FLASH_PE_WIDTH,):
                raise ValueError(
                    f"Flash {flash.id} has {len(pe_per_ch)} PE channels; "
                    f"expected {FLASH_PE_WIDTH}"
                )
            flash_rows.append(
                [
                    flash.id,
                    flash.volume_id,
                    flash.frame,
                    flash.on_beam_time,
                    int(flash.in_beam_frame),
                    flash.time,
                    flash.time_width,
                    flash.time_abs,
                    flash.total_pe,
                    flash.fast_to_total,
                    *np.asarray(flash.center, dtype=np.float32),
                    *np.asarray(flash.width, dtype=np.float32),
                ]
            )
            flash_pe_rows.append(pe_per_ch)

    arrays = {
        "coordinates": np.ascontiguousarray(
            np.vstack(coordinate_blocks), dtype=np.int32
        ),
        "features": np.ascontiguousarray(np.vstack(feature_blocks), dtype=np.float32),
        "sources": np.ascontiguousarray(np.vstack(source_blocks), dtype=np.int32),
        "counts": np.asarray(counts, dtype=np.int32),
        "meta": np.asarray(meta_rows, dtype=np.float32).reshape(-1, 12),
        "run_info": np.asarray(run_rows, dtype=np.int64).reshape(-1, 3),
        "flash_data": np.asarray(flash_rows, dtype=np.float32).reshape(
            -1, FLASH_WIDTH
        ),
        "flash_pe": np.asarray(flash_pe_rows, dtype=np.float32).reshape(
            -1, FLASH_PE_WIDTH
        ),
        "flash_counts": np.asarray(flash_counts, dtype=np.int32),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"Profile: {args.profile}")
    print(f"Exported entries: {entries}")
    print(f"Voxels: {len(arrays['coordinates'])}")
    print(f"Flashes: {len(arrays['flash_data'])}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
