"""Pure NumPy validation and serialization for the ICARUS Triton adapter.

This module deliberately has no Triton, PyTorch, or SPINE imports.  Keeping the
wire contract isolated makes it possible to test request validation and output
indexing on a machine without a GPU or a SPINE runtime.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


FEATURE_WIDTH = 8
META_WIDTH = 12
FLASH_WIDTH = 16
FLASH_PE_WIDTH = 180


PARTICLE_COLUMNS = (
    "event_id",
    "id",
    "interaction_id",
    "shape",
    "pid",
    "is_primary",
    "is_valid",
    "size",
    "calo_ke",
    "csda_ke",
    "mcs_ke",
    "ke",
    "length",
    "start_x",
    "start_y",
    "start_z",
    "end_x",
    "end_y",
    "end_z",
    "start_dir_x",
    "start_dir_y",
    "start_dir_z",
    "end_dir_x",
    "end_dir_y",
    "end_dir_z",
    "start_dedx",
    "end_dedx",
    "vertex_distance",
    "start_straightness",
    "directional_spread",
    "axial_spread",
    "is_contained",
    "is_time_contained",
    "pid_score_0",
    "pid_score_1",
    "pid_score_2",
    "pid_score_3",
    "pid_score_4",
    "pid_score_5",
    "primary_score_0",
    "primary_score_1",
)


INTERACTION_COLUMNS = (
    "event_id",
    "id",
    "size",
    "num_particles",
    "num_primary_particles",
    "is_fiducial",
    "is_flash_matched",
    "flash_total_pe",
    "flash_hypo_pe",
    "vertex_x",
    "vertex_y",
    "vertex_z",
    "dir_x",
    "dir_y",
    "dir_z",
    "is_contained",
    "is_time_contained",
    "particle_count_0",
    "particle_count_1",
    "particle_count_2",
    "particle_count_3",
    "particle_count_4",
    "particle_count_5",
    "primary_particle_count_0",
    "primary_particle_count_1",
    "primary_particle_count_2",
    "primary_particle_count_3",
    "primary_particle_count_4",
    "primary_particle_count_5",
)


FLASH_DATA_COLUMNS = (
    "id",
    "volume_id",
    "frame",
    "on_beam_time",
    "in_beam_frame",
    "time_us",
    "time_width_us",
    "time_abs",
    "total_pe",
    "fast_to_total",
    "center_x",
    "center_y",
    "center_z",
    "width_x",
    "width_y",
    "width_z",
)


INTERACTION_FLASH_COLUMNS = (
    "interaction_row",
    "flash_id",
    "volume_id",
    "time_us",
    "score",
    "total_pe",
)


def split_serving_post_config(
    post_config: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build reconstruction-only core and opt-in FlashMatch configurations.

    The production configuration is also used for MC analysis and therefore
    enables truth processing on several modules.  Triton serves reconstruction
    inputs only.  ``time_containment`` is an alias of ``containment`` but does
    not declare ``run_mode`` in the frozen ICARUS configuration, so it must be
    forced to reconstruction mode explicitly rather than inheriting the
    processor's ``both`` default.
    """

    core = deepcopy(dict(post_config))
    flash = {"flash_match": core.pop("flash_match")}
    core.pop("match", None)
    core.pop("children_count", None)
    for module_name, module_config in core.items():
        if not isinstance(module_config, dict):
            continue
        if module_name == "time_containment" or "run_mode" in module_config:
            module_config["run_mode"] = "reco"
        module_config.pop("truth_point_mode", None)
        module_config.pop("truth_dep_mode", None)
        module_config.pop("truth_ke_mode", None)
    return core, flash


def validate_request_arrays(
    coordinates: np.ndarray,
    features: np.ndarray,
    sources: np.ndarray,
    counts: np.ndarray,
    meta: np.ndarray,
    run_info: np.ndarray,
    flash_data: np.ndarray | None = None,
    flash_pe: np.ndarray | None = None,
    flash_counts: np.ndarray | None = None,
) -> None:
    """Validate one complete Triton request without changing its arrays."""

    _require_dtype(coordinates, np.int32, "COORDINATES")
    _require_dtype(features, np.float32, "FEATURES")
    _require_dtype(sources, np.int32, "SOURCES")
    _require_dtype(counts, np.int32, "COUNTS")
    _require_dtype(meta, np.float32, "META")
    _require_dtype(run_info, np.int64, "RUN_INFO")

    if coordinates.ndim != 2 or coordinates.shape[1] != 4:
        raise ValueError("COORDINATES must have shape [N, 4]")
    if features.ndim != 2 or features.shape[1] != FEATURE_WIDTH:
        raise ValueError(f"FEATURES must have shape [N, {FEATURE_WIDTH}]")
    if sources.ndim != 2 or sources.shape[1] != 2:
        raise ValueError("SOURCES must have shape [N, 2]")
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError("COUNTS must have shape [B], with B >= 1")
    if meta.shape != (counts.size, META_WIDTH):
        raise ValueError(f"META must have shape [B, {META_WIDTH}]")
    if run_info.shape != (counts.size, 3):
        raise ValueError("RUN_INFO must have shape [B, 3]")

    voxel_count = coordinates.shape[0]
    if voxel_count == 0:
        raise ValueError("At least one voxel is required")
    if features.shape[0] != voxel_count or sources.shape[0] != voxel_count:
        raise ValueError("COORDINATES, FEATURES, and SOURCES must share N")
    if np.any(counts <= 0):
        raise ValueError("Every event must contain at least one voxel")

    counts64 = counts.astype(np.int64, copy=False)
    declared_count = int(counts64.sum(dtype=np.int64))
    if declared_count != voxel_count:
        raise ValueError(
            f"COUNTS sums to {declared_count}, but N is {voxel_count}"
        )

    expected_batch_ids = np.repeat(
        np.arange(counts.size, dtype=np.int32), counts64
    )
    if not np.array_equal(coordinates[:, 0], expected_batch_ids):
        raise ValueError(
            "The COORDINATES batch-ID column does not agree with COUNTS"
        )

    if not np.isfinite(features).all():
        raise ValueError("FEATURES contains NaN or infinity")
    if np.any(sources < 0):
        raise ValueError("SOURCES contains a negative module or TPC ID")
    if not np.isfinite(meta).all():
        raise ValueError("META contains NaN or infinity")

    lower = meta[:, 0:3]
    upper = meta[:, 3:6]
    size = meta[:, 6:9]
    count = meta[:, 9:12]
    if np.any(size <= 0) or np.any(count <= 0):
        raise ValueError("META voxel sizes and counts must be positive")
    if not np.allclose(count, np.rint(count), rtol=0.0, atol=0.0):
        raise ValueError("META count columns must contain integral values")
    expected_upper = lower + size * count
    if not np.allclose(upper, expected_upper, rtol=1e-5, atol=1e-4):
        raise ValueError("META upper must equal lower + size * count")

    event_offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(counts64)]
    )
    for event_id in range(counts.size):
        begin, end = event_offsets[event_id : event_id + 2]
        event_coords = coordinates[begin:end, 1:4]
        event_count = np.rint(count[event_id]).astype(np.int64)
        if np.any(event_coords < 0) or np.any(event_coords >= event_count):
            raise ValueError(
                f"COORDINATES for event {event_id} fall outside META count"
            )

    supplied = (
        flash_data is not None,
        flash_pe is not None,
        flash_counts is not None,
    )
    if any(supplied) and not all(supplied):
        raise ValueError(
            "FLASH_DATA, FLASH_PE, and FLASH_COUNTS must be supplied together"
        )
    if not all(supplied):
        return

    assert flash_data is not None
    assert flash_pe is not None
    assert flash_counts is not None
    _require_dtype(flash_data, np.float32, "FLASH_DATA")
    _require_dtype(flash_pe, np.float32, "FLASH_PE")
    _require_dtype(flash_counts, np.int32, "FLASH_COUNTS")
    if flash_data.ndim != 2 or flash_data.shape[1] != FLASH_WIDTH:
        raise ValueError(f"FLASH_DATA must have shape [F, {FLASH_WIDTH}]")
    if flash_pe.ndim != 2 or flash_pe.shape[1] != FLASH_PE_WIDTH:
        raise ValueError(f"FLASH_PE must have shape [F, {FLASH_PE_WIDTH}]")
    if flash_counts.shape != counts.shape:
        raise ValueError("FLASH_COUNTS must have shape [B]")
    if np.any(flash_counts < 0):
        raise ValueError("FLASH_COUNTS cannot contain negative values")
    flash_count = int(flash_counts.astype(np.int64).sum(dtype=np.int64))
    if flash_count != flash_data.shape[0] or flash_pe.shape[0] != flash_count:
        raise ValueError("FLASH_COUNTS does not agree with FLASH_DATA/FLASH_PE")
    if flash_count:
        required_finite = flash_data[:, [0, 1, 2, 3, 4, 5, 6, 8]]
        if not np.isfinite(required_finite).all():
            raise ValueError("Required FLASH_DATA fields contain NaN or infinity")
        integer_fields = flash_data[:, 0:5]
        if not np.allclose(
            integer_fields, np.rint(integer_fields), rtol=0.0, atol=0.0
        ):
            raise ValueError("FLASH_DATA ID/frame/flag fields must be integral")
        if np.any(flash_data[:, 0:2] < 0):
            raise ValueError("FLASH_DATA IDs and volume IDs must be non-negative")
        if np.any((flash_data[:, 4] != 0) & (flash_data[:, 4] != 1)):
            raise ValueError("FLASH_DATA in_beam_frame must be 0 or 1")
        if np.any(flash_data[:, 6] < 0) or np.any(flash_data[:, 8] < 0):
            raise ValueError("FLASH_DATA time width and total PE must be non-negative")
        if not np.isfinite(flash_pe).all() or np.any(flash_pe < 0):
            raise ValueError("FLASH_PE must be finite and non-negative")


def serialize_outputs(
    data: Mapping[str, Any],
    split_to_event_local: Sequence[np.ndarray],
    raw_counts: np.ndarray,
    run_info: np.ndarray,
    flash_match_ran: bool,
) -> dict[str, np.ndarray]:
    """Flatten unwrapped SPINE products into the Triton numeric contract."""

    batch_size = len(raw_counts)
    _require_event_lists(
        data,
        batch_size,
        (
            "ghost_pred",
            "seg_pred",
            "orig_index",
            "points",
            "depositions",
            "sources",
            "reco_particles",
            "reco_interactions",
        ),
    )
    if len(split_to_event_local) != batch_size:
        raise ValueError("split_to_event_local does not match the batch size")

    raw_counts64 = np.asarray(raw_counts, dtype=np.int64)
    event_offsets = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(raw_counts64)]
    )

    ghost_entries: list[np.ndarray] = []
    seg_entries: list[np.ndarray] = []
    orig_entries: list[np.ndarray] = []
    point_entries: list[np.ndarray] = []
    deposition_entries: list[np.ndarray] = []
    source_entries: list[np.ndarray] = []

    for event_id in range(batch_size):
        reorder = np.asarray(split_to_event_local[event_id], dtype=np.int64)
        if reorder.shape != (int(raw_counts64[event_id]),):
            raise ValueError(f"Invalid internal reorder map for event {event_id}")
        if not np.array_equal(np.sort(reorder), np.arange(len(reorder))):
            raise ValueError(f"Internal reorder map for event {event_id} is not a permutation")

        ghost_split = _flat_array(data["ghost_pred"][event_id], np.int64)
        if len(ghost_split) != len(reorder):
            raise ValueError(f"ghost_pred length mismatch for event {event_id}")
        ghost_raw = np.empty_like(ghost_split)
        ghost_raw[reorder] = ghost_split
        ghost_entries.append(ghost_raw)

        segmentation = _flat_array(data["seg_pred"][event_id], np.int64)
        orig_split = _flat_array(data["orig_index"][event_id], np.int64)
        points = _matrix(data["points"][event_id], 3, np.float32, "points")
        depositions = _flat_array(data["depositions"][event_id], np.float32)
        sources = _matrix(data["sources"][event_id], 2, np.int32, "sources")
        deghosted_count = len(segmentation)
        if not all(
            len(value) == deghosted_count
            for value in (orig_split, points, depositions, sources)
        ):
            raise ValueError(f"Deghosted product length mismatch for event {event_id}")
        if np.any(orig_split < 0) or np.any(orig_split >= len(reorder)):
            raise ValueError(f"orig_index is out of range for event {event_id}")

        orig_local = reorder[orig_split]
        seg_entries.append(segmentation)
        orig_entries.append(orig_local + event_offsets[event_id])
        point_entries.append(points)
        deposition_entries.append(depositions)
        source_entries.append(sources)

    segmentation_counts = np.asarray(
        [len(value) for value in seg_entries], dtype=np.int32
    )

    particle_rows: list[list[float]] = []
    particle_voxel_rows: list[tuple[int, int, int]] = []
    particle_voxel_counts: list[int] = []
    particle_counts: list[int] = []

    for event_id, particles in enumerate(data["reco_particles"]):
        particle_counts.append(len(particles))
        reorder = np.asarray(split_to_event_local[event_id], dtype=np.int64)
        for particle in particles:
            particle_row = len(particle_rows)
            particle_rows.append(_particle_row(event_id, particle))

            orig_split = _flat_array(
                getattr(particle, "orig_index", np.empty(0)), np.int64
            )
            if np.any(orig_split < 0) or np.any(orig_split >= len(reorder)):
                raise ValueError(
                    f"Particle {getattr(particle, 'id', -1)} has invalid orig_index"
                )
            orig_local = reorder[orig_split]
            particle_voxel_counts.append(len(orig_local))
            for local_index in orig_local:
                particle_voxel_rows.append(
                    (
                        particle_row,
                        int(local_index),
                        int(event_offsets[event_id] + local_index),
                    )
                )

    interaction_rows: list[list[float]] = []
    interaction_counts: list[int] = []
    interaction_flash_rows: list[list[float]] = []
    interaction_flash_counts: list[int] = []
    if flash_match_ran:
        _require_event_lists(data, batch_size, ("flashes",))
        flashes_by_event = data["flashes"]
    else:
        flashes_by_event = [[] for _ in range(batch_size)]

    for event_id, interactions in enumerate(data["reco_interactions"]):
        interaction_counts.append(len(interactions))
        flash_lookup = {
            (
                int(getattr(flash, "id", -1)),
                int(getattr(flash, "volume_id", -1)),
            ): flash
            for flash in flashes_by_event[event_id]
        }
        for interaction in interactions:
            interaction_row = len(interaction_rows)
            interaction_rows.append(_interaction_row(event_id, interaction))

            ids = _flat_array(getattr(interaction, "flash_ids", []), np.int64)
            volumes = _flat_array(
                getattr(interaction, "flash_volume_ids", []), np.int64
            )
            times = _flat_array(getattr(interaction, "flash_times", []), np.float32)
            scores = _flat_array(getattr(interaction, "flash_scores", []), np.float32)
            num_matches = len(ids)
            if not all(len(value) == num_matches for value in (volumes, times, scores)):
                raise ValueError("Interaction flash-match arrays have different lengths")
            interaction_flash_counts.append(num_matches)
            for index in range(num_matches):
                flash_key = (int(ids[index]), int(volumes[index]))
                if flash_key not in flash_lookup:
                    raise ValueError(
                        "Interaction references a missing flash "
                        f"(event={event_id}, id={flash_key[0]}, "
                        f"volume={flash_key[1]})"
                    )
                total_pe = float(flash_lookup[flash_key].total_pe)
                interaction_flash_rows.append(
                    [
                        float(interaction_row),
                        float(ids[index]),
                        float(volumes[index]),
                        float(times[index]),
                        float(scores[index]),
                        total_pe,
                    ]
                )

    return {
        "EVENT_ID": np.ascontiguousarray(run_info, dtype=np.int64),
        "GHOST_PRED": _concatenate(ghost_entries, np.int64),
        "GHOST_COUNTS": np.ascontiguousarray(raw_counts, dtype=np.int32),
        "SEGMENTATION": _concatenate(seg_entries, np.int64),
        "SEGMENTATION_COUNTS": segmentation_counts,
        "ORIG_INDEX": _concatenate(orig_entries, np.int64),
        "POINTS": _vstack(point_entries, 3, np.float32),
        "DEPOSITIONS": _concatenate(deposition_entries, np.float32),
        "DEGHOSTED_SOURCES": _vstack(source_entries, 2, np.int32),
        "PARTICLES": _rows(particle_rows, len(PARTICLE_COLUMNS)),
        "PARTICLE_COUNTS": np.asarray(particle_counts, dtype=np.int32),
        "PARTICLE_VOXELS": np.asarray(
            particle_voxel_rows, dtype=np.int64
        ).reshape(-1, 3),
        "PARTICLE_VOXEL_COUNTS": np.asarray(
            particle_voxel_counts, dtype=np.int32
        ),
        "INTERACTIONS": _rows(interaction_rows, len(INTERACTION_COLUMNS)),
        "INTERACTION_COUNTS": np.asarray(interaction_counts, dtype=np.int32),
        "INTERACTION_FLASHES": _rows(
            interaction_flash_rows, len(INTERACTION_FLASH_COLUMNS)
        ),
        "INTERACTION_FLASH_COUNTS": np.asarray(
            interaction_flash_counts, dtype=np.int32
        ),
        "FLASH_MATCH_RAN": np.full(
            batch_size, int(flash_match_ran), dtype=np.int32
        ),
    }


def _particle_row(event_id: int, particle: Any) -> list[float]:
    start = _vector_attr(particle, "start_point", 3)
    end = _vector_attr(particle, "end_point", 3)
    start_dir = _vector_attr(particle, "start_dir", 3)
    end_dir = _vector_attr(particle, "end_dir", 3)
    pid_scores = _vector_attr(particle, "pid_scores", 6)
    primary_scores = _vector_attr(particle, "primary_scores", 2)
    row = [
        event_id,
        _scalar_attr(particle, "id", -1),
        _scalar_attr(particle, "interaction_id", -1),
        _scalar_attr(particle, "shape", -1),
        _scalar_attr(particle, "pid", -1),
        _scalar_attr(particle, "is_primary", False),
        _scalar_attr(particle, "is_valid", True),
        _scalar_attr(particle, "size", 0),
        _scalar_attr(particle, "calo_ke"),
        _scalar_attr(particle, "csda_ke"),
        _scalar_attr(particle, "mcs_ke"),
        _scalar_attr(particle, "ke"),
        _scalar_attr(particle, "length"),
        *start,
        *end,
        *start_dir,
        *end_dir,
        _scalar_attr(particle, "start_dedx"),
        _scalar_attr(particle, "end_dedx"),
        _scalar_attr(particle, "vertex_distance"),
        _scalar_attr(particle, "start_straightness"),
        _scalar_attr(particle, "directional_spread"),
        _scalar_attr(particle, "axial_spread"),
        _scalar_attr(particle, "is_contained", False),
        _scalar_attr(particle, "is_time_contained", False),
        *pid_scores,
        *primary_scores,
    ]
    if len(row) != len(PARTICLE_COLUMNS):
        raise RuntimeError("Internal PARTICLES schema mismatch")
    return [float(value) for value in row]


def _interaction_row(event_id: int, interaction: Any) -> list[float]:
    vertex = _vector_attr(interaction, "vertex", 3)
    direction = _vector_attr(interaction, "dir", 3)
    particle_counts = _vector_attr(interaction, "particle_counts", 6, fill=0.0)
    primary_counts = _vector_attr(
        interaction, "primary_particle_counts", 6, fill=0.0
    )
    row = [
        event_id,
        _scalar_attr(interaction, "id", -1),
        _scalar_attr(interaction, "size", 0),
        _scalar_attr(interaction, "num_particles", 0),
        _scalar_attr(interaction, "num_primary_particles", 0),
        _scalar_attr(interaction, "is_fiducial", False),
        _scalar_attr(interaction, "is_flash_matched", False),
        _scalar_attr(interaction, "flash_total_pe"),
        _scalar_attr(interaction, "flash_hypo_pe"),
        *vertex,
        *direction,
        _scalar_attr(interaction, "is_contained", False),
        _scalar_attr(interaction, "is_time_contained", False),
        *particle_counts,
        *primary_counts,
    ]
    if len(row) != len(INTERACTION_COLUMNS):
        raise RuntimeError("Internal INTERACTIONS schema mismatch")
    return [float(value) for value in row]


def _require_dtype(array: np.ndarray, dtype: Any, name: str) -> None:
    if array.dtype != np.dtype(dtype):
        raise ValueError(f"{name} must use {np.dtype(dtype).name}")


def _require_event_lists(
    data: Mapping[str, Any], batch_size: int, keys: Sequence[str]
) -> None:
    for key in keys:
        if key not in data:
            raise KeyError(f"SPINE output is missing required product: {key}")
        if not isinstance(data[key], (list, tuple)) or len(data[key]) != batch_size:
            raise ValueError(f"SPINE product {key} is not a {batch_size}-event list")


def _flat_array(value: Any, dtype: Any) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1)


def _matrix(value: Any, width: int, dtype: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape [N, {width}]")
    return np.ascontiguousarray(array)


def _concatenate(values: Sequence[np.ndarray], dtype: Any) -> np.ndarray:
    if not values:
        return np.empty(0, dtype=dtype)
    return np.ascontiguousarray(np.concatenate(values), dtype=dtype)


def _vstack(values: Sequence[np.ndarray], width: int, dtype: Any) -> np.ndarray:
    if not values:
        return np.empty((0, width), dtype=dtype)
    return np.ascontiguousarray(np.vstack(values), dtype=dtype).reshape(-1, width)


def _rows(values: Sequence[Sequence[float]], width: int) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1, width)


def _scalar_attr(obj: Any, name: str, default: Any = np.nan) -> float:
    try:
        value = getattr(obj, name)
    except (AttributeError, KeyError, ValueError, RuntimeError):
        value = default
    array = np.asarray(value)
    if array.size != 1:
        return float(default)
    return float(array.reshape(-1)[0])


def _vector_attr(
    obj: Any, name: str, width: int, fill: float = np.nan
) -> np.ndarray:
    try:
        value = np.asarray(getattr(obj, name), dtype=np.float32).reshape(-1)
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
        value = np.empty(0, dtype=np.float32)
    result = np.full(width, fill, dtype=np.float32)
    result[: min(width, len(value))] = value[:width]
    return result
