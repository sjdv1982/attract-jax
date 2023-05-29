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
    d["alphabet"] = np.array(_r("?" * 99, 99), bool)
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
        d["energies"] = np.ascontiguousarray(energrads[:, 0])
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
    innergrid_dtype = np.dtype([("Potential", np.int32, 100), ("neighbourlist", np.int32), ("nr_neighbours", np.int16)], align=True)
    innergrid = np.frombuffer(grid_data, offset=pos, count=innergridsize, dtype=innergrid_dtype)
    innergrid = innergrid.reshape((dx, dy, dz))
    neighbour_grid = np.zeros((dx, dy, dz, 2), np.int32)
    neighbour_grid[:, :, :, 0] = innergrid["nr_neighbours"]
    neighbour_grid[:, :, :, 1] = innergrid["neighbourlist"]
    d["neighbour_grid"] = neighbour_grid

    grid_class = namedtuple("Grid", d.keys())
    grid = grid_class(*d.values())
    return grid


grid = read_grid(grid_data)
#energies = grid.energies
neighbour_grid = grid.neighbour_grid