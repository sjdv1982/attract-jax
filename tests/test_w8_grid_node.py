"""D16: an identity pose on a 10–12 grid node retains its grid energy.

The historical integer-node interpolation drops a 0.23 contribution. A
constant field makes the expected value independent of a second scorer.
"""
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'util'))


@pytest.mark.parametrize('backend',['numpy','jax'])
@pytest.mark.parametrize('coordinate',[[1.,1.,1.],[1.,1.25,1.5],[1.25,1.5,1.75]])
def test_identity_pose_retains_023_grid_energy_at_nodes(scoring_inputs,backend,coordinate):
    if backend == 'jax':
        pytest.importorskip('jax')
    from jax_scorer import JaxScoreOracle
    grid = scoring_inputs['grid_object']
    pot = np.zeros_like(grid.inner_potential_grid)
    pot[...,0] = .23
    grid = grid._replace(inner_potential_grid=pot,outer_potential_grid=pot.copy(),
        inner_elec_grid=np.zeros_like(grid.inner_elec_grid),outer_elec_grid=np.zeros_like(grid.outer_elec_grid),
        plateaudis=10.,neighbourdis=12.,
        neighbour_grid=np.empty((*grid.dim,0),dtype=np.uint16),
        nr_neighbours=np.zeros(np.prod(grid.dim),dtype=np.int16),max_nr_neighbours=0)
    opts = dict(scoring_inputs)
    opts.update(grid_object=grid,ligand_ensemble=np.array([[coordinate,coordinate]]),
        ligand_charges=np.zeros(2),lig_pivot=np.zeros(3),energy_only=True,backend=backend)
    oracle = JaxScoreOracle(**opts)
    energy,_ = oracle.score_batch([1],np.zeros((1,6)))
    np.testing.assert_allclose(energy,[.46],rtol=0,atol=1e-6)
