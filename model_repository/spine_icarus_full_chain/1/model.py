"""Triton Python backend for the ICARUS SPINE full reconstruction chain."""

from __future__ import annotations

import hashlib
import os
import sys
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import triton_python_backend_utils as pb_utils

MODEL_VERSION_DIRECTORY = Path(__file__).resolve().parent
if str(MODEL_VERSION_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODEL_VERSION_DIRECTORY))

from adapter_contract import (
    FLASH_PE_WIDTH,
    serialize_outputs,
    split_serving_post_config,
    validate_request_arrays,
)
from spine.config import load_config_file
from spine.construct import BuildManager
from spine.data import Flash, Meta, ObjectList, RunInfo, TensorBatch
from spine.geo import GeoManager
from spine.io.unwrap import Unwrapper
from spine.model.manager import ModelManager
from spine.post import PostManager
from spine.utils import jit as spine_jit


CONFIG_PROFILES = {
    "full": Path("full_chain_co_250625.yaml"),
    "single": Path("legacy/icarus_full_chain_single_co_250625.yaml"),
}


class TritonPythonModel:
    """Run production ICARUS inference, object building, and post-processing."""

    def initialize(self, args):
        self.model_name = args["model_name"]
        self.model_version = args["model_version"]
        self.device_id = int(args["model_instance_device_id"])

        if not torch.cuda.is_available():
            raise RuntimeError(
                "spine_icarus_full_chain requires a CUDA-enabled PyTorch runtime"
            )
        torch.cuda.set_device(self.device_id)

        self.model_directory = Path(__file__).resolve().parent.parent
        self.profile = os.environ.get("SPINE_ICARUS_PROFILE", "full").strip().lower()
        if self.profile not in CONFIG_PROFILES:
            raise ValueError(
                "SPINE_ICARUS_PROFILE must be one of "
                f"{sorted(CONFIG_PROFILES)}, got {self.profile!r}"
            )
        self.config_root = self.model_directory / "config"
        self.config_path = self.config_root / CONFIG_PROFILES[self.profile]
        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"Production SPINE configuration not found: {self.config_path}"
            )

        self._validate_config_snapshot()
        config = load_config_file(str(self.config_path), download=True)
        self._resolve_profile_paths(config)
        self.geometry = GeoManager.initialize_or_get(**config["geo"])
        self.num_modules = self.geometry.tpc.num_modules

        collate_config = config["io"]["loader"].get("collate_fn", {})
        self.split_by_module = (
            isinstance(collate_config, dict)
            and bool(collate_config.get("split", False))
        )
        expected_split = self.profile == "full"
        if self.split_by_module != expected_split:
            raise RuntimeError(
                f"Configuration profile {self.profile!r} resolved to an "
                f"unexpected collate mode: {collate_config!r}"
            )

        model_config = deepcopy(config["model"])
        model_config["network_input"] = {
            "data": "data",
            "sources": "sources",
            "meta": "meta",
            "run_info": "run_info",
        }
        model_config.pop("loss_input", None)
        model_config["to_numpy"] = True
        model_config["train"] = None
        model_config["distributed"] = False
        model_config["rank"] = 0
        model_config["dtype"] = config.get("base", {}).get("dtype", "float32")

        self.model_manager = ModelManager(**model_config)
        self.model_manager.net.eval()

        build_config = deepcopy(config["build"])
        build_config["mode"] = "reco"
        self.builder = BuildManager(**build_config)

        core_post_config, flash_post_config = split_serving_post_config(
            config["post"]
        )
        config_parent = str(self.config_path.parent)
        self.post_manager = PostManager(
            core_post_config,
            parent_path=config_parent,
        )
        self.flash_post_manager = PostManager(
            flash_post_config,
            parent_path=config_parent,
        )
        self.unwrapper = Unwrapper()

        optical = self.geometry.optical
        if optical is None:
            raise RuntimeError("ICARUS optical geometry is not available")
        if optical.num_channels_per_volume != FLASH_PE_WIDTH:
            raise RuntimeError(
                "The Triton FLASH_PE contract expects "
                f"{FLASH_PE_WIDTH} channels per volume, but geometry reports "
                f"{optical.num_channels_per_volume}"
            )

        numpy_probe = torch.zeros(1, dtype=torch.float32).numpy()
        if (
            np.ndarray is not spine_jit.np.ndarray
            or not isinstance(numpy_probe, spine_jit.np.ndarray)
        ):
            raise RuntimeError(
                "Incoherent NumPy/PyTorch runtime detected; use the clean "
                "SPINE 0.16/Triton 23.07 image"
            )

        print(
            f"Initialized {self.model_name} version {self.model_version} on "
            f"CUDA device {self.device_id}; profile={self.profile}; "
            f"split_by_module={self.split_by_module}; "
            f"modules={self.num_modules}; "
            f"core_post={list(self.post_manager.modules)}; "
            f"flash_post={list(self.flash_post_manager.modules)}",
            flush=True,
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                responses.append(self._execute_one(request))
            except Exception:
                details = traceback.format_exc()
                print(details, flush=True)
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(details)
                    )
                )
        return responses

    def _execute_one(self, request):
        coordinates = self._required_input(request, "COORDINATES")
        features = self._required_input(request, "FEATURES")
        sources = self._required_input(request, "SOURCES")
        counts = self._required_input(request, "COUNTS")
        meta_array = self._required_input(request, "META")
        run_info_array = self._required_input(request, "RUN_INFO")

        flash_data = self._optional_input(request, "FLASH_DATA")
        flash_pe = self._optional_input(request, "FLASH_PE")
        flash_counts = self._optional_input(request, "FLASH_COUNTS")

        validate_request_arrays(
            coordinates,
            features,
            sources,
            counts,
            meta_array,
            run_info_array,
            flash_data,
            flash_pe,
            flash_counts,
        )
        self._validate_source_ids(sources)
        if flash_data is not None:
            assert flash_counts is not None
            self._validate_flash_ids(flash_data, flash_counts)

        coordinates = np.ascontiguousarray(coordinates, dtype=np.int32)
        features = np.ascontiguousarray(features, dtype=np.float32)
        sources = np.ascontiguousarray(sources, dtype=np.int32)
        counts = np.ascontiguousarray(counts, dtype=np.int32)
        meta_array = np.ascontiguousarray(meta_array, dtype=np.float32)
        run_info_array = np.ascontiguousarray(run_info_array, dtype=np.int64)

        metas = self._make_meta(meta_array)
        run_infos = self._make_run_info(run_info_array)
        if self.split_by_module:
            data_batch, sources_batch, split_to_event_local = self._split_inputs(
                coordinates,
                features,
                sources,
                counts,
                metas,
            )
        else:
            data_batch, sources_batch, split_to_event_local = self._batch_inputs(
                coordinates,
                features,
                sources,
                counts,
            )

        data = {
            "index": list(range(len(counts))),
            "meta": metas,
            "run_info": run_infos,
            "data": data_batch,
            "sources": sources_batch,
        }

        with torch.inference_mode():
            result = self.model_manager(data)
        data.update(result)
        data = self.unwrapper(data)

        self.builder(data)
        self.post_manager(data)

        flash_match_ran = flash_data is not None
        if flash_match_ran:
            assert flash_pe is not None and flash_counts is not None
            data["flashes"] = self._make_flashes(
                flash_data,
                flash_pe,
                flash_counts,
            )
            self.flash_post_manager(data)

        output_arrays = serialize_outputs(
            data,
            split_to_event_local,
            counts,
            run_info_array,
            flash_match_ran,
        )
        output_tensors = [
            self._output_from_array(name, value)
            for name, value in output_arrays.items()
        ]
        return pb_utils.InferenceResponse(output_tensors=output_tensors)

    @staticmethod
    def _batch_inputs(coordinates, features, sources, counts):
        """Reproduce production CollateAll(split=False)."""

        data_array = np.ascontiguousarray(
            np.hstack([coordinates, features]), dtype=np.float32
        )
        sources_array = np.ascontiguousarray(sources, dtype=np.float32)
        data_batch = TensorBatch(
            data_array,
            counts=counts,
            has_batch_col=True,
            coord_cols=[1, 2, 3],
        )
        sources_batch = TensorBatch(sources_array, counts=counts)
        split_to_event_local = [
            np.arange(int(count), dtype=np.int64) for count in counts
        ]
        return data_batch, sources_batch, split_to_event_local

    def _split_inputs(
        self,
        coordinates,
        features,
        sources,
        counts,
        metas,
    ):
        """Reproduce production CollateAll(split=True, target_id=0)."""

        event_edges = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)]
        )
        data_blocks = []
        source_blocks = []
        split_counts = np.zeros(
            len(counts) * self.num_modules, dtype=np.int64
        )
        split_to_event_local = []

        for event_id in range(len(counts)):
            lower, upper = event_edges[event_id : event_id + 2]
            coords_event = coordinates[lower:upper, 1:4]
            features_event = features[lower:upper]
            sources_event = sources[lower:upper]

            wrapped, module_indexes = self.geometry.split(
                coords_event,
                0,
                meta=metas[event_id],
            )
            event_order = []
            for module_id, module_index_raw in enumerate(module_indexes):
                module_index = np.asarray(module_index_raw, dtype=np.int64)
                source_index = np.where(sources_event[:, 0] == module_id)[0]
                if not np.array_equal(np.sort(module_index), source_index):
                    raise ValueError(
                        "Geometry-derived module assignment disagrees with "
                        f"SOURCES for event {event_id}, module {module_id}"
                    )

                internal_batch = event_id * self.num_modules + module_id
                split_counts[internal_batch] = len(module_index)
                event_order.append(module_index)

                data_block = np.empty(
                    (len(module_index), 1 + 3 + features.shape[1]),
                    dtype=np.float32,
                )
                data_block[:, 0] = internal_batch
                data_block[:, 1:4] = wrapped[module_index]
                data_block[:, 4:] = features_event[module_index]
                data_blocks.append(data_block)
                source_blocks.append(
                    sources_event[module_index].astype(np.float32, copy=False)
                )

            split_to_event_local.append(
                np.concatenate(event_order).astype(np.int64, copy=False)
            )

        data_array = np.ascontiguousarray(np.vstack(data_blocks), dtype=np.float32)
        sources_array = np.ascontiguousarray(
            np.vstack(source_blocks), dtype=np.float32
        )
        data_batch = TensorBatch(
            data_array,
            counts=split_counts,
            has_batch_col=True,
            coord_cols=[1, 2, 3],
        )
        sources_batch = TensorBatch(sources_array, counts=split_counts)
        return data_batch, sources_batch, split_to_event_local

    @staticmethod
    def _make_meta(meta_array):
        metas = []
        for row in meta_array:
            metas.append(
                Meta(
                    lower=np.array(row[0:3], dtype=np.float32, copy=True),
                    upper=np.array(row[3:6], dtype=np.float32, copy=True),
                    size=np.array(row[6:9], dtype=np.float32, copy=True),
                    count=np.rint(row[9:12]).astype(np.int64),
                )
            )
        return metas

    @staticmethod
    def _make_run_info(run_info_array):
        return [
            RunInfo(run=int(row[0]), subrun=int(row[1]), event=int(row[2]))
            for row in run_info_array
        ]

    @staticmethod
    def _make_flashes(flash_data, flash_pe, flash_counts):
        flashes_by_event = []
        edges = np.concatenate(
            [
                np.zeros(1, dtype=np.int64),
                np.cumsum(flash_counts, dtype=np.int64),
            ]
        )
        for event_id in range(len(flash_counts)):
            lower, upper = edges[event_id : event_id + 2]
            flashes = []
            for row, pe_per_ch in zip(
                flash_data[lower:upper], flash_pe[lower:upper]
            ):
                flashes.append(
                    Flash(
                        id=int(round(float(row[0]))),
                        volume_id=int(round(float(row[1]))),
                        frame=int(round(float(row[2]))),
                        on_beam_time=int(round(float(row[3]))),
                        in_beam_frame=bool(round(float(row[4]))),
                        time=float(row[5]),
                        time_width=float(row[6]),
                        time_abs=float(row[7]),
                        total_pe=float(row[8]),
                        fast_to_total=float(row[9]),
                        center=np.array(row[10:13], dtype=np.float32, copy=True),
                        width=np.array(row[13:16], dtype=np.float32, copy=True),
                        pe_per_ch=np.array(
                            pe_per_ch, dtype=np.float32, copy=True
                        ),
                    )
                )
            flashes_by_event.append(ObjectList(flashes, Flash()))
        return flashes_by_event

    def _validate_source_ids(self, sources):
        if self.profile == "single" and np.any(sources[:, 0] != 0):
            raise ValueError(
                "The single-cryostat profile only accepts cryoE source module ID 0"
            )
        if np.any(sources[:, 0] >= self.num_modules):
            raise ValueError(
                f"SOURCES module IDs must be in [0, {self.num_modules - 1}]"
            )
        # The second LArCV source column contains readout TPC IDs.  ICARUS
        # single-cryostat files use IDs 0--3, which are not the same indexing
        # domain as GeoManager's geometry chambers.  Non-negativity is already
        # enforced by validate_request_arrays; preserve these producer IDs.

    def _validate_flash_ids(self, flash_data, flash_counts):
        volume_ids = np.rint(flash_data[:, 1]).astype(np.int64, copy=False)
        if self.profile == "single" and np.any(volume_ids != 0):
            raise ValueError(
                "The single-cryostat profile only accepts cryoE flash volume ID 0"
            )
        if np.any(volume_ids >= self.num_modules):
            raise ValueError(
                f"FLASH_DATA volume IDs must be in [0, {self.num_modules - 1}]"
            )
        edges = np.concatenate(
            [
                np.zeros(1, dtype=np.int64),
                np.cumsum(flash_counts, dtype=np.int64),
            ]
        )
        for event_id in range(len(flash_counts)):
            lower, upper = edges[event_id : event_id + 2]
            ids = np.rint(flash_data[lower:upper, 0]).astype(
                np.int64, copy=False
            )
            if len(np.unique(ids)) != len(ids):
                raise ValueError(
                    f"FLASH_DATA IDs must be unique within event {event_id}"
                )

    def _resolve_profile_paths(self, config):
        """Resolve the single-profile FlashMatch path from the frozen snapshot."""

        if self.profile != "single":
            return
        flash_cfg = (
            self.config_root
            / "modifier"
            / "numi"
            / "flashmatch"
            / "flashmatch_numi_250214.cfg"
        )
        if not flash_cfg.is_file():
            raise FileNotFoundError(
                f"Single-cryostat FlashMatch configuration not found: {flash_cfg}"
            )
        config["post"]["flash_match"]["cfg"] = str(flash_cfg)

    def _validate_config_snapshot(self):
        manifest = self.model_directory / "CONFIG_SHA256SUMS"
        if not manifest.is_file():
            raise FileNotFoundError(f"Configuration manifest not found: {manifest}")

        config_root = self.model_directory / "config"
        config_root_resolved = config_root.resolve()
        actual_files = {
            path.relative_to(config_root).as_posix()
            for path in config_root.rglob("*")
            if path.is_file()
        }
        if len(actual_files) != 114:
            raise RuntimeError(
                "The production configuration snapshot must contain exactly "
                f"114 files, but found {len(actual_files)}"
            )

        checked_files = set()
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, recorded_path = line.split(maxsplit=1)
            if len(expected) != 64 or any(
                char not in "0123456789abcdef" for char in expected
            ):
                raise RuntimeError(f"Invalid SHA-256 manifest line: {line}")
            marker = "model_repository/spine_icarus_full_chain/config/"
            relative = recorded_path.strip()
            if marker in relative:
                relative = relative.split(marker, 1)[1]
            elif relative.startswith("config/"):
                relative = relative[len("config/") :]
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise RuntimeError(
                    f"Unsafe configuration manifest path: {recorded_path}"
                )
            relative = relative_path.as_posix()
            if relative in checked_files:
                raise RuntimeError(
                    f"Duplicate configuration manifest entry: {relative}"
                )

            path = (config_root / relative_path).resolve()
            if not path.is_relative_to(config_root_resolved):
                raise RuntimeError(
                    f"Configuration manifest path escapes snapshot: {relative}"
                )
            if not path.is_file():
                raise FileNotFoundError(f"Manifest entry is missing: {path}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                raise RuntimeError(f"Configuration checksum mismatch: {path}")
            checked_files.add(relative)

        if checked_files != actual_files:
            missing = sorted(actual_files - checked_files)
            extra = sorted(checked_files - actual_files)
            raise RuntimeError(
                "Configuration manifest does not exactly match the snapshot; "
                f"unlisted={missing}, nonexistent={extra}"
            )

    @classmethod
    def _required_input(cls, request, name):
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        if tensor is None:
            raise ValueError(f"Missing {name} input")
        return cls._input_as_numpy(tensor)

    @classmethod
    def _optional_input(cls, request, name):
        tensor = pb_utils.get_input_tensor_by_name(request, name)
        return None if tensor is None else cls._input_as_numpy(tensor)

    @staticmethod
    def _input_as_numpy(tensor):
        # Avoid the Triton 23.07 NumPy bridge with the NumPy 2.x runtime.
        torch_tensor = torch.utils.dlpack.from_dlpack(tensor.to_dlpack())
        return torch_tensor.cpu().numpy()

    @staticmethod
    def _output_from_array(name, value):
        # The same compatibility constraint applies to output construction.
        array = np.ascontiguousarray(value)
        torch_tensor = torch.from_numpy(array).contiguous()
        return pb_utils.Tensor.from_dlpack(
            name,
            torch.utils.dlpack.to_dlpack(torch_tensor),
        )

    def finalize(self):
        self.flash_post_manager = None
        self.post_manager = None
        self.builder = None
        self.unwrapper = None
        self.model_manager = None
        self.geometry = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"Finalized {self.model_name} version {self.model_version}",
            flush=True,
        )
