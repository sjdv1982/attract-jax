"""ATTRACT grid I/O shared by JAX and NumPy scorers."""

import struct
from collections import namedtuple

import numpy as np


def read_grid_with_electro(grid_data: bytes):
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
    if is_torquegrid:
        raise ValueError("Torque grids are not supported by this utility")
    arch = read_short()
    if arch != 64:
        raise ValueError(f"Unexpected grid architecture: {arch}")

    d = {}
    d["gridspacing"] = read_double()
    d["gridextension"] = read_int()
    d["plateaudis"] = read_double()
    d["neighbourdis"] = read_double()
    alphabet = np.array(_r("?" * 99, 99), bool)
    d["alphabet"] = alphabet
    alphabet_atomtypes = (np.where(alphabet)[0] + 1).astype(np.int32)
    nr_vdw_channels = len(alphabet_atomtypes)

    d["origin"] = np.array((read_float(), read_float(), read_float()), dtype=np.float64)
    dx, dy, dz = read_int(), read_int(), read_int()
    d["dim"] = np.array((dx, dy, dz), dtype=np.int32)
    dx2, dy2, dz2 = read_int(), read_int(), read_int()
    d["dim2"] = np.array((dx2, dy2, dz2), dtype=np.int32)
    d["natoms"] = read_int()
    _pivot = np.array((read_double(), read_double(), read_double()), dtype=np.float64)

    nr_energrads = read_int()
    shm_energrads = read_int()
    if nr_energrads == 0:
        raise ValueError("Grid has no potentials/gradients")
    if shm_energrads != -1:
        raise ValueError("Can't read grid potentials from shared memory")
    energrads = np.frombuffer(
        grid_data, offset=pos, count=nr_energrads * 4, dtype=np.float32
    )
    energrads = energrads.reshape(nr_energrads, 4)
    pos += energrads.nbytes

    nr_neighbours = read_int()
    shm_neighbours = read_int()
    if shm_neighbours != -1:
        raise ValueError("Can't read neighbour grid from shared memory")
    nb_dtype = np.dtype([("type", np.uint8), ("index", np.uint32)], align=True)
    neighbours = np.frombuffer(
        grid_data, offset=pos, count=nr_neighbours, dtype=nb_dtype
    )
    pos += neighbours.nbytes
    neighbour_index = np.ascontiguousarray(neighbours["index"])

    innergridsize = read_long()
    if innergridsize != dx * dy * dz:
        raise ValueError(f"Inner grid size mismatch: {innergridsize} vs {dx*dy*dz}")

    innergrid_dtype = np.dtype(
        [
            ("potential", np.uint32, 100),
            ("neighbourlist", np.int32),
            ("nr_neighbours", np.int16),
        ],
        align=True,
    )
    innergrid = np.frombuffer(
        grid_data, offset=pos, count=innergridsize, dtype=innergrid_dtype
    )
    pos += innergrid.nbytes
    innergrid = innergrid.reshape((dz, dy, dx)).swapaxes(0, 2)
    innergrid = np.ascontiguousarray(innergrid)

    inner_potential_grid = np.zeros((nr_vdw_channels, dx, dy, dz, 4), dtype=np.float32)
    inner_elec_grid = np.zeros((dx, dy, dz, 4), dtype=np.float32)
    pot_ind = innergrid["potential"]
    pot_pos = 0
    for n in range(99):
        curr = pot_ind[:, :, :, n]
        if not alphabet[n]:
            if curr.max() != 0:
                raise ValueError(
                    f"Unexpected potentials for alphabet-disabled atomtype {n+1}"
                )
            continue
        if curr.min() < 1 or curr.max() > len(energrads):
            raise ValueError(f"Inner potential index out of range for atomtype {n+1}")
        inner_potential_grid[pot_pos] = energrads[curr - 1]
        pot_pos += 1
    elec_ind = pot_ind[:, :, :, 99]
    mask = elec_ind > 0
    if elec_ind.max() > len(energrads):
        raise ValueError("Inner electrostatic potential index out of range")
    inner_elec_grid[mask] = energrads[elec_ind[mask] - 1]
    del pot_ind

    nr_neighbours_inner = np.ascontiguousarray(innergrid["nr_neighbours"])
    d["nr_neighbours"] = nr_neighbours_inner
    neighbourlist = innergrid["neighbourlist"]
    max_nr_neighbours = int(nr_neighbours_inner.max())
    d["max_nr_neighbours"] = max_nr_neighbours
    neighbour_grid = np.full((dx, dy, dz, max_nr_neighbours), 2**16 - 1, np.uint16)
    for n in range(max_nr_neighbours):
        maskn = nr_neighbours_inner > n
        nb_ind = neighbourlist[maskn]
        nb = nb_ind - 1 + n
        neighbour_grid[maskn, n] = neighbour_index[nb]
    d["neighbour_grid"] = neighbour_grid

    biggridsize = read_long()
    if biggridsize != dx2 * dy2 * dz2:
        raise ValueError(f"Outer grid size mismatch: {biggridsize} vs {dx2*dy2*dz2}")
    biggrid = np.frombuffer(
        grid_data, offset=pos, count=biggridsize * 100, dtype=np.uint32
    )
    pos += biggrid.nbytes
    pot_ind = biggrid.reshape((dz2, dy2, dx2, 100)).swapaxes(0, 2)

    outer_potential_grid = np.zeros(
        (nr_vdw_channels, dx2, dy2, dz2, 4), dtype=np.float32
    )
    outer_elec_grid = np.zeros((dx2, dy2, dz2, 4), dtype=np.float32)
    pot_pos = 0
    for n in range(99):
        curr = pot_ind[:, :, :, n]
        if not alphabet[n]:
            if curr.max() != 0:
                raise ValueError(f"Unexpected outer potentials for atomtype {n+1}")
            continue
        if curr.max() > len(energrads):
            raise ValueError(f"Outer potential index out of range for atomtype {n+1}")
        maskn = curr > 0
        outer_potential_grid[pot_pos][maskn] = energrads[curr[maskn] - 1]
        pot_pos += 1
    elec_ind = pot_ind[:, :, :, 99]
    if elec_ind.max() > len(energrads):
        raise ValueError("Outer electrostatic potential index out of range")
    maske = elec_ind > 0
    outer_elec_grid[maske] = energrads[elec_ind[maske] - 1]

    # Legacy .grid stores force convention: channels 1:4 = -(dE/d_i).
    # Negate to gradient convention: channels 1:4 = +(dE/d_i).
    inner_potential_grid[:, :, :, :, 1:4] *= -1
    outer_potential_grid[:, :, :, :, 1:4] *= -1
    inner_elec_grid[:, :, :, 1:4] *= -1
    outer_elec_grid[:, :, :, 1:4] *= -1

    d["inner_potential_grid"] = inner_potential_grid
    d["outer_potential_grid"] = outer_potential_grid
    d["inner_elec_grid"] = inner_elec_grid
    d["outer_elec_grid"] = outer_elec_grid
    d["alphabet_atomtypes"] = alphabet_atomtypes

    grid_class = namedtuple("Grid", tuple(d.keys()) + ("neighbour_grid_ravel",))
    return grid_class(*d.values(), neighbour_grid_ravel=None)


# ---------------------------------------------------------------------------
# NPZ grid serialisation
# ---------------------------------------------------------------------------

_GRID_SCALAR_FIELDS = (
    "gridspacing",
    "gridextension",
    "plateaudis",
    "neighbourdis",
    "natoms",
    "max_nr_neighbours",
)
_GRID_ARRAY_FIELDS = (
    "alphabet",
    "origin",
    "dim",
    "dim2",
    "nr_neighbours",
    "neighbour_grid",
    "inner_potential_grid",
    "outer_potential_grid",
    "inner_elec_grid",
    "outer_elec_grid",
    "alphabet_atomtypes",
)


def write_grid_npz(grid, path: str) -> None:
    """Serialise a Grid namedtuple (as returned by read_grid_with_electro or
    grid_generator.generate_grid) to a compressed .npz file.

    All array fields are stored verbatim.  Scalar fields are wrapped in
    0-d numpy arrays so np.savez_compressed handles them uniformly.
    ``neighbour_grid_ravel`` is recomputed on load and is NOT stored.
    """
    arrays = {}
    for name in _GRID_SCALAR_FIELDS:
        arrays[name] = np.array(getattr(grid, name))
    for name in _GRID_ARRAY_FIELDS:
        arrays[name] = np.asarray(getattr(grid, name))
    np.savez_compressed(path, **arrays)


def read_grid_npz(path: str):
    """Load a Grid namedtuple previously written by write_grid_npz.

    Returns the same namedtuple layout as read_grid_with_electro, with
    ``neighbour_grid_ravel`` set to None (it will be built lazily by
    JaxScoreOracle if needed).
    """
    data = np.load(path)
    d = {}
    for name in _GRID_SCALAR_FIELDS:
        val = data[name]
        # Restore Python scalars for the fields that were ints/floats
        if name in ("gridextension", "natoms", "max_nr_neighbours"):
            d[name] = int(val)
        else:
            d[name] = float(val)
    for name in _GRID_ARRAY_FIELDS:
        d[name] = data[name]
    grid_class = namedtuple("Grid", tuple(d.keys()) + ("neighbour_grid_ravel",))
    return grid_class(*d.values(), neighbour_grid_ravel=None)


