"""A pair outside the plateau still needs its interpolated baseline corrected.

Run from the Alaric repository root:
  conda run --live-stream -n alaric python -m pytest -q attract-jax/tests/test_w8_plateau_shell.py

One receptor, one ligand, one analytic nonbon8 pair. The center lies outside
the plateau, but its interpolation cell has corners on both sides. There are
no distant pairs whose approximation could obscure this regression.
"""
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "util"))


@pytest.fixture(params=[5.0, 10.0], ids=["plateau-5", "plateau-10"])
def plateau_shell_pair(request, scoring_inputs, tmp_path):
    plateau = request.param
    spacing = 0.9
    coordinate = np.array([plateau + 0.05, 0.2, 0.2])
    dim = (16, 16, 16)
    # Scaling preserves order-one energies at each plateau, making the shell
    # error visible above float32 storage noise for both default distances.
    rc, ac, ivor = 4.0 * plateau**8, 2.0 * plateau**6, 1.0
    parameter_path = tmp_path / "one-pair.npz"
    np.savez(parameter_path, rc=np.array([[rc]]), ac=np.array([[ac]]),
             ivor=np.array([[ivor]]))

    def pair_energy(dsq):
        # ivor=1 removes the short-range emin shift in both LJ branches.
        return (rc / dsq - ac) / dsq**3

    rec_charge = 1.0
    lig_charge = -0.75
    electro_scale = 332.053986 / 15.0

    def electro_per_charge(dsq):
        return np.maximum(1.0 / dsq - 1.0 / 2500.0, 0.0) * rec_charge * electro_scale

    indices = np.indices(dim).transpose(1, 2, 3, 0)
    inner_dsq = np.sum((indices * spacing)**2, axis=-1)
    outer_dsq = np.sum(((indices * 2 - 2) * spacing)**2, axis=-1)
    inner = np.zeros((1, *dim, 4), dtype=np.float32)
    outer = np.zeros_like(inner)
    inner[0, ..., 0] = pair_energy(np.maximum(inner_dsq, plateau**2))
    outer[0, ..., 0] = pair_energy(np.maximum(outer_dsq, plateau**2))
    inner_elec = np.zeros((*dim, 4), dtype=np.float32)
    outer_elec = np.zeros_like(inner_elec)
    inner_elec[..., 0] = electro_per_charge(np.maximum(inner_dsq, plateau**2))
    outer_elec[..., 0] = electro_per_charge(np.maximum(outer_dsq, plateau**2))
    grid = scoring_inputs["grid_object"]._replace(
        origin=np.zeros(3), gridspacing=spacing, gridextension=2,
        dim=np.array(dim), dim2=np.array(dim), plateaudis=plateau,
        alphabet_atomtypes=np.array([1], dtype=np.int32),
        alphabet=np.array([True] + [False] * 98),
        inner_potential_grid=inner, outer_potential_grid=outer,
        inner_elec_grid=inner_elec,
        outer_elec_grid=outer_elec,
        neighbour_grid=np.zeros((*dim, 1), dtype=np.uint16),
        nr_neighbours=np.ones(np.prod(dim), dtype=np.int16),
        max_nr_neighbours=1, neighbourdis=plateau + 3.0, natoms=1,
    )
    lower = np.floor(coordinate / spacing).astype(int)
    corners = np.array(list(product([0, 1], repeat=3))) + lower
    corner_dsq = np.sum((corners * spacing)**2, axis=1)
    assert np.dot(coordinate, coordinate) > plateau**2
    assert corner_dsq.min() < plateau**2 < corner_dsq.max()
    fractional = coordinate / spacing - lower
    weights = np.prod(np.where(corners - lower, fractional, 1 - fractional), axis=1)
    baseline = np.dot(weights, pair_energy(np.maximum(corner_dsq, plateau**2)))
    baseline += lig_charge * np.dot(
        weights, electro_per_charge(np.maximum(corner_dsq, plateau**2))
    )
    exact_dsq = np.dot(coordinate, coordinate)
    expected = pair_energy(exact_dsq) + lig_charge * electro_per_charge(exact_dsq)
    assert abs(baseline - expected) > 1e-4, "fixture must expose the shell error"
    return dict(
        receptor_ensemble=np.zeros((1, 1, 3)), receptor_atomtypes=np.array([1]),
        receptor_charges=np.array([rec_charge]), ligand_ensemble=coordinate.reshape(1, 1, 3),
        ligand_atomtypes=np.array([1]), ligand_charges=np.array([lig_charge]),
        lig_pivot=np.zeros(3), grid_object=grid, attract_par_npz=str(parameter_path),
    ), expected, baseline


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_outside_plateau_straddling_cell_matches_exact_pair(plateau_shell_pair, backend):
    if backend == "jax":
        pytest.importorskip("jax")
    from jax_scorer import JaxScoreOracle

    inputs, expected, baseline = plateau_shell_pair
    oracle = JaxScoreOracle(**inputs, backend=backend, energy_only=True,
                            energy_batch=1, score_batch_size=1, dof_type="rotvec")
    dofs = np.zeros((1, 6))
    # Verify this fixture actually presents the intended interpolated plateau
    # field before checking the independent scalar exact-pair result.
    potential = oracle.score_potential_energy_batch([1], dofs)
    np.testing.assert_allclose(potential, [baseline], rtol=0, atol=1e-6)
    energy, _ = oracle.score_batch([1], dofs)
    np.testing.assert_allclose(energy, [expected], rtol=0, atol=1e-6)
