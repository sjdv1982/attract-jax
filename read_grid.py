import struct
import numpy as np
from collections import namedtuple

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
    d["neighbours"] = np.ascontiguousarray(neighbours["index"])    
    pos += neighbours.nbytes

    innergridsize = read_long()
    assert innergridsize == dx * dy * dz, (innergridsize, dx * dy * dz)
    innergrid_dtype = np.dtype([("potential", np.uint32, 100), ("neighbourlist", np.int32), ("nr_neighbours", np.int16)], align=True)
    innergrid = np.frombuffer(grid_data, offset=pos, count=innergridsize, dtype=innergrid_dtype)
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
    neighbour_grid = np.zeros((dx, dy, dz, 2), np.int32)
    neighbour_grid[:, :, :, 0] = innergrid["nr_neighbours"]
    neighbour_grid[:, :, :, 1] = innergrid["neighbourlist"]
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

    grid_class = namedtuple("Grid", d.keys())
    grid = grid_class(*d.values())
    return grid

def pad_grid(grid, padding):
    # pads neighbourlist grid, padding a list of size N with padding[N-1]
    # if padding is an integer, it is repeated.
    # if padding is a list with too few elements, the last element is repeated
    neighbour_grid = np.array(grid.neighbour_grid)
    g = neighbour_grid.reshape(-1,2)
    max_contacts = g[:, 0].max()
    try:
        len(padding)
    except Exception:    
        padding = np.repeat(int(padding), max_contacts)
    else:
        padding2 = np.empty(max_contacts,np.uint16)
        lp = min(len(padding2), len(padding))
        padding2[:lp] = padding[:lp]
        padding2[len(padding):max_contacts] = padding[-1]
        padding = padding2
        
    lengths = g[:, 0]
    pos_old = g[:, 1] - 1

    padded_lengths = lengths + padding[lengths-1]
    mask_length0 = (lengths==0)
    padded_lengths[mask_length0] = 0
    pos = np.empty_like(lengths)
    pos[0] = 0
    pos[1:] = np.cumsum(padded_lengths[:-1])

    # from: https://stackoverflow.com/questions/47125697/concatenate-range-arrays-given-start-stop-numbers-in-a-vectorized-way-numpy
    def create_ranges(starts, ends):
        l = ends - starts
        clens = l.cumsum()
        ids = np.ones(clens[-1],dtype=int)
        ids[0] = starts[0]
        ids[clens[:-1]] = starts[1:] - ends[:-1]+1
        out = ids.cumsum()
        return out
    
    mask_length = ~mask_length0
    pos_old_nonzero = pos_old[mask_length]
    pos_nonzero = pos[mask_length]
    lengths_nonzero = lengths[mask_length]
    indices_old = create_ranges(pos_old_nonzero, pos_old_nonzero + lengths_nonzero)
    indices_new = create_ranges(pos_nonzero, pos_nonzero + lengths_nonzero)
    nb2 = np.zeros(padded_lengths.sum(), dtype=np.uint32)
    nan = np.iinfo(np.uint32).max
    nb2[:] = nan
    nb2[indices_new] = grid.neighbours[indices_old]
    
    g[:, 1] = pos + 1
    g[:, 1][mask_length0] = 0
    
    try:
        import jax.numpy as jnp
        if isinstance(grid.neighbours, jnp.ndarray):
            nb2 = jnp.array(nb2)
        if isinstance(grid.neighbour_grid, jnp.ndarray):
            neighbour_grid = jnp.array(neighbour_grid)
    except ImportError:
        pass
    grid = grid._replace(neighbours=nb2)
    grid = grid._replace(neighbour_grid=neighbour_grid)
    return grid

