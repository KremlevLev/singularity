from __future__ import annotations

from collections.abc import Sequence

import jax
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def build_mesh(
    mesh_shape: Sequence[int] | None = None,
    mesh_axes: Sequence[str] | None = None,
    devices: Sequence[jax.Device] | None = None,
) -> Mesh:
    devices_array = np.asarray(devices or jax.devices()).reshape(mesh_shape or (-1,))
    return Mesh(devices_array, tuple(mesh_axes or (f"axis_{idx}" for idx in range(devices_array.ndim))))


def make_named_sharding(mesh: Mesh, spec: PartitionSpec) -> NamedSharding:
    return NamedSharding(mesh, spec)


def default_partition_spec(ndim: int) -> PartitionSpec:
    return PartitionSpec(*(f"axis_{idx}" for idx in range(ndim)))
