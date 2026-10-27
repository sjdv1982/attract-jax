"""Small synthetic scoring data; usable without JAX or compiled kernels."""

from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def scoring_inputs():
    # Nonconstant energy and stored gradient channels, charged atoms, and
    # several atoms per pose exercise both interpolation and scatter sums.
    rng = np.random.default_rng(10)
    dim = (5, 5, 5)
    values = dict(
        origin=np.zeros(3), gridspacing=1.0, gridextension=2,
        dim=np.array(dim), dim2=np.array(dim), plateaudis=2.0,
        alphabet_atomtypes=np.array([1, 2], dtype=np.int32),
        inner_potential_grid=rng.normal(size=(2, *dim, 4)).astype(np.float32),
        outer_potential_grid=rng.normal(size=(2, *dim, 4)).astype(np.float32),
        inner_elec_grid=rng.normal(size=(*dim, 4)).astype(np.float32),
        outer_elec_grid=rng.normal(size=(*dim, 4)).astype(np.float32),
        neighbour_grid=np.broadcast_to(np.array([0, 1, 65535], np.uint16), (*dim, 3)).copy(),
        nr_neighbours=np.full(np.prod(dim), 2, dtype=np.int16),
        max_nr_neighbours=3, neighbourdis=3., natoms=2,
        alphabet=np.array([True, True] + [False] * 97),
    )
    grid = namedtuple("TestGrid", values)(**values)
    return dict(
        receptor_ensemble=np.array([[[1., 1., 1.], [3., 3., 3.]],
                                    [[1.2, 1., 1.], [3., 2.8, 3.]]]),
        receptor_atomtypes=np.array([1, 2]),
        receptor_charges=np.array([0.2, -0.1]),
        ligand_ensemble=np.array([[[1.5, 1.6, 1.7], [2.5, 2.4, 2.3]],
                                  [[1.6, 1.5, 1.8], [2.4, 2.5, 2.2]]]),
        ligand_atomtypes=np.array([1, 2]),
        ligand_charges=np.array([-0.2, 0.1]),
        lig_pivot=np.array([0.1, 0.2, 0.3]),
        grid_object=grid,
        attract_par_npz=str(ROOT / "attract-original/attract-par.npz"),
        energy_batch=2,
    )


