# syntax=docker/dockerfile:1.7

ARG TRITON_IMAGE=nvcr.io/nvidia/tritonserver@sha256:e335164412828c8958797deb1caf516ba0ce4a53b64afd7ee0160cc42f5defd2
ARG SPINE_IMAGE=ghcr.io/deeplearnphysics/spine@sha256:8b4399a1dbd4ab00086f9ec902f5f1c911f5c1fc83d17ca885e2c229ebc90ac4

FROM ${TRITON_IMAGE} AS triton

FROM triton AS triton-native-runtime

RUN set -eux; \
    mkdir -p /runtime-libs; \
    for library in \
        /usr/lib/x86_64-linux-gnu/libre2.so.9 \
        /usr/lib/x86_64-linux-gnu/libdcgm.so.2; do \
        cp -L "${library}" /runtime-libs/; \
    done

# The authoritative Python/CUDA/SPINE runtime
FROM ${SPINE_IMAGE} AS runtime

USER root

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends libb64-0d; \
    rm -rf /var/lib/apt/lists/*

# Copy Triton and its non-base native dependencies without touching SPINE's
# Python installation. Keep these libraries from the same pinned Triton image
# so their ABI versions match the server binary exactly.
COPY --from=triton /opt/tritonserver/bin/tritonserver /opt/tritonserver/bin/tritonserver
COPY --from=triton /opt/tritonserver/lib/ /opt/tritonserver/lib/
COPY --from=triton /opt/tritonserver/backends/python/ /opt/tritonserver/backends/python/
COPY --from=triton-native-runtime /runtime-libs/ /usr/local/lib/

RUN ldconfig

ENV PATH="/opt/tritonserver/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/tritonserver/lib:/opt/tritonserver/backends/python:${LD_LIBRARY_PATH}" \
    PYTHONNOUSERSITE=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    NUMBA_CACHE_DIR=/tmp/spine-numba-cache \
    SPINE_CACHE_DIR=/model-cache/weights

RUN mkdir -p /model-cache/weights

# Fail the build if Triton is missing a native dependency or uses the wrong Python.
RUN set -eux; \
    ldd /opt/tritonserver/bin/tritonserver | tee /tmp/tritonserver.ldd; \
    ldd /opt/tritonserver/backends/python/triton_python_backend_stub \
        | tee /tmp/python-stub.ldd; \
    if grep -q "not found" /tmp/tritonserver.ldd /tmp/python-stub.ldd; then \
        exit 1; \
    fi; \
    grep -Eq "libpython3\\.10" /tmp/python-stub.ldd


RUN python3 -m pip check

# Exercise every compatibility boundary that previously failed.
RUN python3 - <<'PY'
import importlib.metadata as metadata
import os
import sys

import numpy as np
import torch

from spine.math.graph import connected_components
from spine.math.neighbors import RadiusNeighborsClassifier
from spine.utils import jit as spine_jit
from spine.utils.gnn.network import inter_cluster_distance
from spine.utils.ppn import ParticlePointPredictor

assert sys.version_info[:2] == (3, 10)
assert os.environ.get("NUMBA_DISABLE_JIT") in (None, "", "0")

numpy_distributions = [
    d for d in metadata.distributions()
    if (d.metadata.get("Name") or "").lower() == "numpy"
]
assert len(numpy_distributions) == 1, numpy_distributions

probe = torch.zeros(1, dtype=torch.float32).numpy()
assert np is spine_jit.np
assert np.ndarray is spine_jit.np.ndarray
assert isinstance(probe, spine_jit.np.ndarray)

edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
labels = connected_components(edges, 4)
assert labels.tolist() == [0, 0, 0, 1]

classifier = RadiusNeighborsClassifier(radius=1.9)
neighbor_labels, orphan_index = classifier.fit_predict(
    np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
    np.array([3, 3], dtype=np.int64),
    np.array([[0.5, 0, 0]], dtype=np.float32),
)
assert neighbor_labels.tolist() == [3]
assert len(orphan_index) == 0

voxels = np.array(
    [[0, 0, 0], [1, 0, 0], [5, 0, 0], [6, 0, 0]],
    dtype=np.float32,
)
clusters = [
    np.array([0, 1], dtype=np.int64),
    np.array([2, 3], dtype=np.int64),
]
distances, closest = inter_cluster_distance(
    voxels,
    clusters,
    counts=np.array([2], dtype=np.int64),
    return_index=True,
)
assert distances.shape == (2, 2)
assert closest.shape == (2, 2)

predictor = ParticlePointPredictor(use_numpy=True)
empty_points = predictor.get_end_points_numpy(
    torch.empty((0, 3), dtype=torch.float32),
    [],
    np.empty(0, dtype=np.int64),
    torch.empty((0, 1), dtype=torch.float32),
)
assert isinstance(empty_points, torch.Tensor)
assert tuple(empty_points.shape) == (0, 6)

print("Clean SPINE/Triton compatibility tests: OK")
PY

RUN rm -rf /tmp/spine-numba-cache

WORKDIR /workspace

EXPOSE 8000 8001 8002

ENTRYPOINT ["/opt/tritonserver/bin/tritonserver"]
