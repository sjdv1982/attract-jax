import struct
import numpy as np
from collections import namedtuple


# from: https://stackoverflow.com/questions/47125697/concatenate-range-arrays-given-start-stop-numbers-in-a-vectorized-way-numpy
def _create_ranges(starts, ends):
    l = ends - starts
    clens = l.cumsum()
    ids = np.ones(clens[-1],dtype=int)
    ids[0] = starts[0]
    ids[clens[:-1]] = starts[1:] - ends[:-1]+1
    out = ids.cumsum()
    return out

def read_grid(grid_data, read_gradients=False):
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
    d["origin"] = np.array((x, y, z))
    dx, dy, dz = read_int(), read_int(), read_int()
    d["dim"] = np.array((dx, dy, dz))
    dx2, dy2, dz2 = read_int(), read_int(), read_int()
    d["dim2"] = np.array((dx2, dy2, dz2))
    d["natoms"] = read_int()
    x, y, z = read_double(), read_double(), read_double()
    # d["pivot"] = x, y, z #ignored

    nr_energrads = read_int()
    shm_energrads = read_int()    
    if nr_energrads:
        assert shm_energrads == -1, "Can't read grid from shared memory"
        energrads = np.frombuffer(grid_data, offset=pos, count=nr_energrads * 4, dtype=np.float32)
        energrads = energrads.reshape(nr_energrads, 4)
        if not read_gradients:
            energies = np.ascontiguousarray(energrads[:, 0])
        pos += energrads.nbytes
    
    nr_neighbours = read_int()
    shm_neighbours = read_int()
    assert shm_neighbours == -1, "Can't read grid from shared memory"
    nb_dtype = np.dtype([("type",np.uint8),("index", np.uint32)], align=True)

    neighbours = np.frombuffer(grid_data, offset=pos, count=nr_neighbours, dtype=nb_dtype)
    pos += neighbours.nbytes
    neighbours = np.ascontiguousarray(neighbours["index"])    

    innergridsize = read_long()
    assert innergridsize == dx * dy * dz, (innergridsize, dx * dy * dz)
    innergrid_dtype = np.dtype([("potential", np.uint32, 100), ("neighbourlist", np.int32), ("nr_neighbours", np.int16)], align=True)
    innergrid = np.frombuffer(grid_data, offset=pos, count=innergridsize, dtype=innergrid_dtype)
    assert neighbours.max() < 2**16, "Too many atoms"

    pos += innergrid.nbytes
    innergrid = innergrid.reshape((dz, dy, dx))
    innergrid = innergrid.swapaxes(0, 2)
    innergrid = np.ascontiguousarray(innergrid)
    if nr_energrads:
        if read_gradients:
            inner_potential_grid = np.zeros((nr_potentials, dx, dy, dz, 4), energrads.dtype)
        else:
            inner_potential_grid = np.zeros((nr_potentials, dx, dy, dz), energrads.dtype)
        pot_ind = innergrid["potential"]
        pot_pos = 0
        for n in range(99):
            curr_pot_ind = pot_ind[:, :, :, n]
            if not alphabet[n]:
                assert curr_pot_ind.max() == 0, (n, curr_pot_ind.min(), curr_pot_ind.max())
                continue
            assert curr_pot_ind.min() >= 1 and curr_pot_ind.max() <= len(energrads), (n, curr_pot_ind.min(), curr_pot_ind.max(), len(energrads))
            if read_gradients:
                inner_potential_grid[pot_pos] = energrads[curr_pot_ind-1]
            else:
                inner_potential_grid[pot_pos] = energies[curr_pot_ind-1]
            pot_pos += 1
        del pot_ind
        d["inner_potential_grid"] = inner_potential_grid
        
    nr_neighbours = np.ascontiguousarray(innergrid["nr_neighbours"])
    d["nr_neighbours"] = nr_neighbours
    neighbourlist = innergrid["neighbourlist"]
    max_nr_neighbours = nr_neighbours.max() 
    d["max_nr_neighbours"] = max_nr_neighbours
    neighbour_grid = np.full((dx, dy, dz, max_nr_neighbours), 2**16 - 1, np.uint16)
    for n in range(max_nr_neighbours):
        mask = (nr_neighbours > n)
        nb_ind = neighbourlist[mask]
        nb = neighbours[nb_ind-1+n]
        neighbour_grid[mask, n] = nb
    d["neighbour_grid"] = neighbour_grid

    biggridsize = read_long()
    if nr_energrads:
        assert biggridsize == dx2 * dy2 * dz2, (biggridsize, dx2 * dy2 * dz2, (dx2, dy2, dz2))
        biggrid = np.frombuffer(grid_data, offset=pos, count=biggridsize*100, dtype=np.uint32)
        pos += biggrid.nbytes

        if read_gradients:
            outer_potential_grid = np.zeros((nr_potentials, dx2, dy2, dz2, 4), energrads.dtype)
        else:
            outer_potential_grid = np.zeros((nr_potentials, dx2, dy2, dz2), energrads.dtype)
        pot_ind = biggrid.reshape(dz2, dy2, dx2, 100)
        pot_ind = pot_ind.swapaxes(0, 2)
        pot_pos = 0
        for n in range(99):
            curr_pot_ind = pot_ind[:, :, :, n]
            if not alphabet[n]:
                assert curr_pot_ind.max() == 0, (n, curr_pot_ind.min(), curr_pot_ind.max())
                continue
            assert curr_pot_ind.max() <= len(energrads), (curr_pot_ind.max(), len(energrads))
            mask = (curr_pot_ind>0)
            if read_gradients:
                outer_potential_grid[pot_pos][mask] = energrads[curr_pot_ind[mask]-1]
            else:
                outer_potential_grid[pot_pos][mask] = energies[curr_pot_ind[mask]-1]
            pot_pos += 1
        del pot_ind
        del energrads
        if not read_gradients:
            del energies
        d["outer_potential_grid"] = outer_potential_grid
    else:
        assert biggridsize == 0, biggridsize

    grid_class = namedtuple("Grid", tuple(d.keys()) + ("neighbour_grid_ravel",))
    grid = grid_class(*d.values(), neighbour_grid_ravel=None)
    return grid
