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
