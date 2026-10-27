"""Energy-only scoring must not build or evaluate derivative machinery."""

from collections import namedtuple
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("jax")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "util"))
sys.path.insert(0, str(ROOT / "nb_kernel"))

from jax_scorer import JaxScoreOracle
from score_pairs import score_pairs_jax
from forcefields import nonbon8


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
        max_nr_neighbours=3,
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


@pytest.mark.parametrize("rotation", ["euler", "rotvec"])
def test_oracle_energy_only_skips_autodiff(scoring_inputs, rotation):
    dofs = np.array([[0.1, 0.2, 0.3, 0., 0., 0.],
                     [0.2, -0.1, 0.1, 0.1, -0.1, 0.1]])
    ens = np.array([1, 2])
    full = JaxScoreOracle(**scoring_inputs, dof_type=rotation)
    expected, grads = full.score_batch(ens, dofs)
    assert grads.shape == (2, 6)
    assert np.isfinite(grads).all()
    with patch("jax.grad", side_effect=AssertionError("unexpected grad")), \
         patch("jax.value_and_grad", side_effect=AssertionError("unexpected value_and_grad")), \
         patch("jax.custom_jvp", side_effect=AssertionError("unexpected custom_jvp")):
        energy = JaxScoreOracle(**scoring_inputs, dof_type=rotation, energy_only=True)
        actual, grads = energy.score_batch(ens, dofs)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)
        assert grads.shape == (0, 6)
        # Also exercise the conformer-specific and potential-only entry points.
        result, _ = energy.score_batch(ens, dofs, conformers=np.array([0, 1]))
        assert np.isfinite(result).all()
        assert np.isfinite(energy.score_potential_energy_batch(ens, dofs)).all()


@pytest.mark.parametrize("mode", ["clamp", "correction"])
def test_pairs_energy_only_skips_autodiff(mode):
    FF = namedtuple("FF", "rc ac ivor emin rmin2")
    ff = FF(*(np.array([[x]]) for x in (4., 2., 1., -0.1, 2.)))
    args = dict(
        query_coords=np.array([[1., 0., 0.], [3., 0., 0.]]),
        rec_coords=np.array([[0., 0., 0.]]),
        rec_atomtypes_idx=np.array([0]), rec_charges_scaled=np.array([0.2]),
        query_atomtype_idx=0, query_charge=-0.1, ff_params=ff, ff_module=nonbon8,
        nr_neigh=np.array([1, 1]), nb_start=np.array([0, 1]),
        nb_concat=np.array([0, 0]), plateaudis_sq=4., plateau_mode=mode,
    )
    expected, grads = score_pairs_jax(**args, return_gradients=True)
    assert np.isfinite(grads).all()
    with patch("jax.value_and_grad", side_effect=AssertionError("unexpected autodiff")):
        actual, grads = score_pairs_jax(**args)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert grads is None
