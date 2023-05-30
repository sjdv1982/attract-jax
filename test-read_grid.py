import struct, sys
import numpy as np
from collections import namedtuple

gridfile = sys.argv[1]
with open(gridfile, "rb") as f:
    grid_data = f.read()


def read_grid(grid_data):
    pos = 0

    def _r(token, size):
        nonlocal pos
        result = struct.unpack_from(token, grid_data, pos)
        pos += size
        return result

    def _rr(token):
        size = struct.calcsize(token)

        def func():
            result = _r(token, size)
            return result[0]

        return func

    read_bool = _rr("?")
    read_short = _rr("h")
    read_int = _rr("i")
    read_float = _rr("f")
    read_double = _rr("d")
    read_long = _rr("l")

    is_torquegrid = read_bool()
    assert not is_torquegrid, is_torquegrid
    arch = read_short()
    assert arch == 64, arch
    d = {}
    d["gridspacing"] = read_double()
    d["gridextension"] = read_int()
    d["plateaudis"] = read_double()
    d["neighbourdis"] = read_double()
    alphabet = np.array(_r("?" * 99, 99), bool)
    d["alphabet"] = alphabet
    nr_potentials = d["alphabet"].sum()  # ignore electrostatic potential at 99
    x, y, z = read_float(), read_float(), read_float()
    d["origin"] = x, y, z
    dx, dy, dz = read_int(), read_int(), read_int()
    d["dim"] = dx, dy, dz
    dx2, dy2, dz2 = read_int(), read_int(), read_int()
    d["dim2"] = dx2, dy2, dz2
    d["natoms"] = read_int()
    x, y, z = read_double(), read_double(), read_double()
    # d["pivot"] = x, y, z #ignored

    nr_energrads = read_int()
    shm_energrads = read_int()    
    if nr_energrads:
        assert shm_energrads == -1, "Can't read grid from shared memory"
        energrads = np.frombuffer(grid_data, offset=pos, count=nr_energrads * 4, dtype=np.float32)
        energrads = energrads.reshape(nr_energrads, 4)
        energies = np.ascontiguousarray(energrads[:, 0])
        pos += energrads.nbytes
    
    nr_neighbours = read_int()
    shm_neighbours = read_int()
    assert shm_neighbours == -1, "Can't read grid from shared memory"
    nb_dtype = np.dtype([("type",np.uint8),("index", np.uint32)], align=True)

    neighbours = np.frombuffer(grid_data, offset=pos, count=nr_neighbours, dtype=nb_dtype)
    d["neighbours"] = np.ascontiguousarray(neighbours["index"])    
    pos += neighbours.nbytes

    innergridsize = read_long()
    assert innergridsize == dx * dy * dz, (innergridsize, dx * dy * dz)
    innergrid_dtype = np.dtype([("potential", np.uint32, 100), ("neighbourlist", np.int32), ("nr_neighbours", np.int16)], align=True)
    innergrid = np.frombuffer(grid_data, offset=pos, count=innergridsize, dtype=innergrid_dtype)
    pos += innergrid.nbytes
    innergrid = innergrid.reshape((dx, dy, dz))
    if nr_energrads:
        inner_potential_grid = np.zeros((nr_potentials, dx, dy, dz), energies.dtype)
        pot_ind = innergrid["potential"]
        pot_pos = 0
        for n in range(99):
            curr_pot_ind = pot_ind[:, :, :, n]
            if not alphabet[n]:
                assert curr_pot_ind.max() == 0, (n, curr_pot_ind.min(), curr_pot_ind.max())
                continue
            assert curr_pot_ind.min() >= 1 and curr_pot_ind.max() <= len(energies), (n, curr_pot_ind.min(), curr_pot_ind.max(), len(energies))
            inner_potential_grid[pot_pos] = energies[curr_pot_ind-1]
            pot_pos += 1
        del pot_ind
        d["inner_potential_grid"] = inner_potential_grid
    neighbour_grid = np.zeros((dx, dy, dz, 2), np.int32)
    neighbour_grid[:, :, :, 0] = innergrid["nr_neighbours"]
    neighbour_grid[:, :, :, 1] = innergrid["neighbourlist"]
    d["neighbour_grid"] = neighbour_grid

    biggridsize = read_long()
    if nr_energrads:
        assert biggridsize == dx2 * dy2 * dz2, (biggridsize, dx2 * dy2 * dz2, (dx2, dy2, dz2))
        biggrid = np.frombuffer(grid_data, offset=pos, count=biggridsize*100, dtype=np.uint32)
        pos += biggrid.nbytes

        outer_potential_grid = np.zeros((nr_potentials, dx2, dy2, dz2), energies.dtype)
        pot_ind = biggrid.reshape(dx2, dy2, dz2, 100)
        pot_pos = 0
        for n in range(99):
            curr_pot_ind = pot_ind[:, :, :, n]
            if not alphabet[n]:
                assert curr_pot_ind.max() == 0, (n, curr_pot_ind.min(), curr_pot_ind.max())
                continue
            assert curr_pot_ind.max() <= len(energies), (curr_pot_ind.max(), len(energies))
            mask = (curr_pot_ind>0)
            outer_potential_grid[pot_pos][mask] = energies[curr_pot_ind[mask]-1]
            pot_pos += 1
        del pot_ind
        del energies
        d["outer_potential_grid"] = outer_potential_grid
    else:
        assert biggridsize == 0, biggridsize

    grid_class = namedtuple("Grid", d.keys())
    grid = grid_class(*d.values())
    return grid


grid = read_grid(grid_data)
neighbour_grid = grid.neighbour_grid