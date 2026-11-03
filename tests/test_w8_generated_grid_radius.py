"""The W8 correction-neighbour radius is applied by generate_grid itself."""
from collections import namedtuple
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "util"))


def test_generated_neighbour_grid_covers_expanded_plateau_shell(monkeypatch):
    import grid_generator

    # Keep this focused on the generated correction list: potential values do
    # not affect which receptor indices the runtime correction can visit.
    monkeypatch.setattr(
        grid_generator,
        "_compute_potential_grid",
        lambda *, shape, **kwargs: np.zeros((*shape, 4), dtype=np.float32),
    )
    params = namedtuple("FFParams", ("rc", "ac", "ivor", "emin", "rmin2"))(
        rc=np.ones((1, 1)),
        ac=np.ones((1, 1)),
        ivor=np.ones((1, 1)),
        emin=np.ones((1, 1)),
        rmin2=np.ones((1, 1)),
    )
    plateau = 5.0
    spacing = 0.9
    requested_radius = plateau + 2.0
    receptor = np.array([0.0, 0.0, 0.0])
    grid = grid_generator.generate_grid(
        rec_coords=np.array([receptor]),
        rec_atomtypes=np.array([1]),
        rec_charges_raw=np.zeros(1),
        ff_params=params,
        forcefield="nonbon8",
        ffelec=1.0,
        gridspacing=spacing,
        gridextension=2,
        plateaudis=plateau,
        neighbourdis=requested_radius,
        lig_atomtypes=np.array([1]),
        return_gradients=False,
    )

    # The scorer selects this neighbor cell using floor(voxel + .5). Its lower
    # interpolation corner is inside the plateau while the ligand center is
    # outside it, so this pair's trilinear plateau-shell baseline needs exact
    # correction. The receptor is outside p+2 from the selected list center.
    query = np.array([3.9, 3.9, 3.0])
    index = (10, 10, 9)
    center = grid.origin + (np.asarray(index) + 0.5) * spacing
    lower_corner = grid.origin + np.floor((query - grid.origin) / spacing) * spacing
    assert np.linalg.norm(query - receptor) > plateau
    assert np.linalg.norm(lower_corner - receptor) < plateau
    assert np.linalg.norm(receptor - center) > requested_radius
    count = int(grid.nr_neighbours[index])
    assert 0 in grid.neighbour_grid[index][:count]
    assert grid.neighbourdis == requested_radius
