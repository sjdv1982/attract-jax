"""NumPy scoring parity, boundary behavior and execution without JAX."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "util"))

from grid_io import write_grid_npz, read_grid_npz
from jax_scorer import JaxScoreOracle
from minfor import dofs_to_mats_np, rotvec_dofs_to_mats_np, parse_args
from numpy_grid_score import _direct_energies


@pytest.mark.parametrize("rotation", ["euler", "rotvec"])
@pytest.mark.parametrize("pool", [False, True])
def test_numpy_matches_jax(scoring_inputs, rotation, pool):
    pytest.importorskip("jax")
    # Includes inner-only, outer-only, out-of-both-grids, and tiny rotations.
    dofs = np.array([
        [.1, .2, .3, 0., 0., 0.], [.2, -.1, .1, .1, -.1, .1],
        [0., 0., 0., -2., -2., -2.], [0., 0., 0., 20., 20., 20.],
        [1e-12, -1e-12, 1e-12, .12, -.13, .14],
    ])
    ens = np.array([1, 2, 1, 2, 1])
    conformers = np.array([0, 1, 1, 0, 0])
    kwargs = dict(**scoring_inputs, dof_type=rotation, pool_conformers=pool, energy_only=True)
    jax_oracle = JaxScoreOracle(**kwargs)
    numpy_oracle = JaxScoreOracle(**kwargs, backend="numpy")
    # Check both default-conformer and explicit mixed-conformer calls.
    for conf in (None, conformers):
        expected, _ = jax_oracle.score_batch(ens, dofs, conformers=conf)
        actual, gradients = numpy_oracle.score_batch(ens, dofs, conformers=conf)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-3)
        assert gradients.shape == (0, 6)
        expected_pot = jax_oracle.score_potential_energy_batch(ens, dofs, conformers=conf)
        actual_pot = numpy_oracle.score_potential_energy_batch(ens, dofs, conformers=conf)
        np.testing.assert_allclose(actual_pot, expected_pot, rtol=0, atol=1e-3)
    e, g = numpy_oracle.score_single(1, dofs[0])
    assert np.isfinite(e) and g is None
    empty, gradients = numpy_oracle.score_batch([], np.empty((0, 6)))
    assert empty.shape == (0,) and gradients.shape == (0, 6)


@pytest.mark.parametrize("zero_neighbours", [False, True])
def test_zero_neighbours_and_uncharged_ligand(scoring_inputs, zero_neighbours):
    pytest.importorskip("jax")
    grid = scoring_inputs["grid_object"]
    scoring_inputs["ligand_charges"] = np.zeros(2)
    if zero_neighbours:
        grid = grid._replace(
            neighbour_grid=np.empty((*grid.dim, 0), dtype=np.uint16),
            nr_neighbours=np.zeros(np.prod(grid.dim), dtype=np.int16),
            max_nr_neighbours=0,
        )
        scoring_inputs["grid_object"] = grid
    dofs = np.zeros((1, 6))
    opts = dict(**scoring_inputs, energy_only=True)
    expected, _ = JaxScoreOracle(**opts).score_batch([1], dofs)
    actual, _ = JaxScoreOracle(**opts, backend="numpy").score_batch([1], dofs)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-3)


def test_grid_npz_roundtrip(scoring_inputs, tmp_path):
    grid = scoring_inputs["grid_object"]
    path = tmp_path / "grid.npz"
    write_grid_npz(grid, str(path))
    restored = read_grid_npz(str(path))
    for field in grid._fields:
        np.testing.assert_array_equal(getattr(restored, field), getattr(grid, field))


@pytest.mark.parametrize("grid_free", [False, True])
def test_numpy_cli_never_imports_jax(scoring_inputs, tmp_path, grid_free):
    # Fail on any attempted JAX import, even in an environment with JAX installed.
    (tmp_path / "sitecustomize.py").write_text('''
import sys
class NoJax:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("jax", "jaxlib", "grid_generator"):
            raise AssertionError("NumPy mode attempted to import " + fullname)
sys.meta_path.insert(0, NoJax())
''')
    grid_path = tmp_path / "grid.npz"
    if not grid_free:
        write_grid_npz(scoring_inputs["grid_object"], str(grid_path))
    dofs = np.array([[.1, .2, .3, 0., 0., 0.], [0., 0., 0., -2., -2., -2.]])
    ens, conformers = np.array([1, 2]), np.array([0, 1])
    files = dict(
        dofs=dofs, ens=ens, conformers=conformers,
        rec=scoring_inputs["receptor_ensemble"], rt=scoring_inputs["receptor_atomtypes"],
        rq=scoring_inputs["receptor_charges"], lig=scoring_inputs["ligand_ensemble"],
        lt=scoring_inputs["ligand_atomtypes"], lq=scoring_inputs["ligand_charges"],
    )
    for name, array in files.items():
        np.save(tmp_path / f"{name}.npy", array)
    output = tmp_path / "energies.npy"
    cmd = [
        sys.executable, str(ROOT / "util/minfor.py"), "--oracle", "numpy",
        "--score", "--energy-only",
        *(["--no-grid"] if grid_free else ["--grid", str(grid_path)]),
        "--attract-par-npz", scoring_inputs["attract_par_npz"],
        "--input-npy", str(tmp_path / "dofs.npy"), "--input-format", "euler",
        "--input-ens", str(tmp_path / "ens.npy"),
        "--input-conformers", str(tmp_path / "conformers.npy"),
        "--receptor-coordinates", str(tmp_path / "rec.npy"),
        "--receptor-atomtypes", str(tmp_path / "rt.npy"),
        "--receptor-charges", str(tmp_path / "rq.npy"),
        "--ligand-ensemble", str(tmp_path / "lig.npy"),
        "--ligand-atomtypes", str(tmp_path / "lt.npy"),
        "--ligand-charges", str(tmp_path / "lq.npy"),
        "--output-npy", str(output),
    ]
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    # CLI derives the pivot from the ligand when none is supplied.
    kwargs = dict(scoring_inputs, energy_only=True, backend="numpy", grid_free=grid_free)
    if grid_free:
        kwargs.pop("grid_object")
    kwargs["lig_pivot"] = scoring_inputs["ligand_ensemble"][0].mean(axis=0)
    expected, _ = JaxScoreOracle(**kwargs).score_batch(ens, dofs, conformers=conformers)
    np.testing.assert_array_equal(np.load(output), expected.astype(np.float32))
    # The same process also exercises streamed output and pooled conformers.
    result = subprocess.run(cmd + ["--score-batch-size", "1", "--pool-conformers"],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    np.testing.assert_array_equal(np.load(output), expected.astype(np.float32))


def test_numpy_requires_energy_only(scoring_inputs):
    with pytest.raises(ValueError, match="energy_only=True"):
        JaxScoreOracle(**scoring_inputs, backend="numpy")


def _scalar_direct_reference(inputs, ens, dofs, conformers, rotation):
    """Independent scalar all-pairs reference using full parameter tables."""
    par = np.load(inputs["attract_par_npz"])
    transform = rotvec_dofs_to_mats_np if rotation == "rotvec" else dofs_to_mats_np
    mats = transform(dofs, inputs["lig_pivot"])
    energies = []
    for pose, (ens_id, conf) in enumerate(zip(ens, conformers)):
        lig = inputs["ligand_ensemble"][conf]
        lig = np.column_stack((lig, np.ones(len(lig)))) @ mats[pose]
        total = 0.
        for l, lt in enumerate(inputs["ligand_atomtypes"] - 1):
            for r, rt in enumerate(inputs["receptor_atomtypes"] - 1):
                dsq = max(0.001, float(sum(
                    (lig[l, :3] - inputs["receptor_ensemble"][ens_id - 1, r]) ** 2
                )))
                rc, ac, ivor = (float(par[key][rt, lt]) for key in ("rc", "ac", "ivor"))
                emin = -27. * ac ** 4 / (256. * rc ** 3)
                vlj = (rc / dsq - ac) / dsq ** 3
                total += vlj + (ivor - 1.) * emin if dsq < 4. * rc / (3. * ac) else ivor * vlj
                charge = (inputs["ligand_charges"][l] * inputs["receptor_charges"][r]
                          * 332.053986 / 15.)
                if abs(charge) > 1e-3:
                    total += charge * max(1. / dsq - 1. / 2500., 0.)
        energies.append(total)
    return np.array(energies)


@pytest.mark.parametrize("rotation", ["euler", "rotvec"])
@pytest.mark.parametrize("pool", [False, True])
def test_grid_free_matches_scalar_reference(scoring_inputs, rotation, pool):
    inputs = dict(scoring_inputs)
    inputs.pop("grid_object")
    dofs = np.array([[.1, .2, .3, 0., 0., 0.], [.2, -.1, .1, .1, -.1, .1],
                     [0., 0., 0., 55., 0., 0.]])
    ens, conformers = np.array([1, 2, 1]), np.array([0, 1, 0])
    oracle = JaxScoreOracle(**inputs, energy_only=True, backend="numpy", grid_free=True,
                            pool_conformers=pool, dof_type=rotation)
    assert oracle._grid_j is None
    for conf in (None, conformers):
        expected = _scalar_direct_reference(
            inputs, ens, dofs, np.zeros(3, dtype=int) if conf is None else conf, rotation,
        )
        actual, gradients = oracle.score_batch(ens, dofs, conformers=conf)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-8)
        assert gradients.shape == (0, 6)
    with pytest.raises(ValueError, match="grid-free"):
        oracle.score_potential_energy_batch(ens, dofs)
    empty, _ = oracle.score_batch([], np.empty((0, 6)))
    assert empty.shape == (0,)


def test_direct_chunking_and_charge_cutoff(scoring_inputs):
    inputs = dict(scoring_inputs)
    inputs.pop("grid_object")
    oracle = JaxScoreOracle(**inputs, energy_only=True, backend="numpy", grid_free=True)
    # Charge product below threshold must behave exactly like an uncharged pair.
    charge = np.array([1e-8, 0.])
    args = (inputs["ligand_ensemble"], inputs["receptor_ensemble"][0],
            oracle._rec_atomtypes_ff_j, charge, oracle._lig_atomtypes_ff_j,
            oracle._lig_charge_scaled_j, oracle._ff)
    chunked = _direct_energies(*args, ligand_chunk_size=1, receptor_chunk_size=1)
    whole = _direct_energies(*args)
    np.testing.assert_allclose(chunked, whole, rtol=1e-12)
    uncharged = _direct_energies(args[0], args[1], args[2], np.zeros(2), *args[4:])
    np.testing.assert_array_equal(whole, uncharged)


def test_grid_free_rejects_grid(scoring_inputs):
    with pytest.raises(ValueError, match="cannot be combined"):
        JaxScoreOracle(**scoring_inputs, energy_only=True, backend="numpy", grid_free=True)


def test_grid_free_masks_dummy_atoms(scoring_inputs):
    inputs = dict(scoring_inputs)
    inputs.pop("grid_object")
    opts = dict(energy_only=True, backend="numpy", grid_free=True)
    expected, _ = JaxScoreOracle(**inputs, **opts).score_batch([1], np.zeros((1, 6)))
    for prefix in ("receptor", "ligand"):
        coords = inputs[f"{prefix}_ensemble"]
        inputs[f"{prefix}_ensemble"] = np.concatenate((coords, coords[:, :1]), axis=1)
        inputs[f"{prefix}_atomtypes"] = np.append(inputs[f"{prefix}_atomtypes"], 99)
        inputs[f"{prefix}_charges"] = np.append(inputs[f"{prefix}_charges"], 100.)
    actual, _ = JaxScoreOracle(**inputs, **opts).score_batch([1], np.zeros((1, 6)))
    np.testing.assert_array_equal(actual, expected)


def test_direct_overlap_uses_legacy_distance_floor(scoring_inputs):
    inputs = dict(scoring_inputs)
    inputs.pop("grid_object")
    inputs["ligand_ensemble"][0, 0] = inputs["receptor_ensemble"][0, 0]
    dofs = np.zeros((1, 6))
    oracle = JaxScoreOracle(**inputs, energy_only=True, backend="numpy", grid_free=True)
    actual, _ = oracle.score_batch([1], dofs)
    expected = _scalar_direct_reference(inputs, [1], dofs, [0], "euler")
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


@pytest.mark.parametrize("extra_args", [["--grid", "unused.grid"], ["--generate-grid"],
                                       ["--oracle", "jax"]])
def test_no_grid_cli_rejects_incompatible_options(monkeypatch, extra_args):
    monkeypatch.setattr(sys, "argv", [
        "minfor.py", "--oracle", "numpy", "--score", "--energy-only", "--no-grid",
        "--identity", "--attract-par-npz", "unused.npz", *extra_args,
    ])
    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 2
