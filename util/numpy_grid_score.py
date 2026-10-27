"""NumPy forward scoring with grids or direct all-pairs evaluation.

The kernel interface matches reproduce_grid_score.build_kernel so input loading,
atom-type mapping, ensembles, conformers and CLI output can share the same oracle.
No JAX or compiled nonbonded kernel is imported or required.
"""

import numpy as np


def _dofs_to_mats(dofs, pivot, rotation):
    a, b, c = dofs[:, :3].T
    if rotation == "euler":
        cs, cp, ss, sp = np.cos(b), np.cos(a), np.sin(b), np.sin(a)
        cr, sr = np.cos(c), np.sin(c)
        rot = np.array([
            [cr * cs * cp + sr * sp, sr * cs * cp - cr * sp, ss * cp],
            [cr * cs * sp - sr * cp, sr * cs * sp + cr * cp, ss * sp],
            [-cr * ss, -sr * ss, cs],
        ])
    elif rotation == "rotvec":
        theta = np.sqrt(a * a + b * b + c * c)
        safe_t = np.where(theta > 1e-10, theta, 1.)
        k0, k1, k2 = a / safe_t, b / safe_t, c / safe_t
        s = np.where(theta > 1e-10, np.sin(theta), theta)
        co = np.where(theta > 1e-10, np.cos(theta), 1. - .5 * theta * theta)
        omc = 1. - co
        rot = np.array([
            [co + omc * k0 * k0, omc * k0 * k1 - s * k2, omc * k0 * k2 + s * k1],
            [omc * k1 * k0 + s * k2, co + omc * k1 * k1, omc * k1 * k2 - s * k0],
            [omc * k2 * k0 - s * k1, omc * k2 * k1 + s * k0, co + omc * k2 * k2],
        ])
    else:
        raise ValueError(f"Unsupported rotation={rotation!r}")
    rot_row = np.swapaxes(np.moveaxis(rot, -1, 0), 1, 2)
    mats = np.zeros((len(dofs), 4, 4), dtype=np.float64)
    mats[:, :3, :3] = rot_row
    mats[:, 3, :3] = dofs[:, 3:6] + pivot - np.einsum("j,bji->bi", pivot, rot_row)
    mats[:, 3, 3] = 1.
    return mats


def _transform(mats, coords):
    homogeneous = np.concatenate((coords, np.ones((*coords.shape[:-1], 1))), axis=-1)
    signature = "bjk,bkl->bjl" if coords.ndim == 3 else "jk,bkl->bjl"
    return np.einsum(signature, homogeneous, mats)[..., :3]


def _interpolate(channels, voxel, grid):
    lo = np.floor(voxel).astype(np.intp)
    hi = np.ceil(voxel).astype(np.intp)
    w0, w1 = hi - voxel, voxel - lo
    # Inactive inner/outer grid lookups can be outside their grids. Clip them
    # before indexing; the caller masks their energies after interpolation.
    upper = np.array(grid.shape[1:4]) - 1
    lo, hi = np.clip(lo, 0, upper), np.clip(hi, 0, upper)
    result = np.zeros(voxel.shape[:2], dtype=np.float64)
    for wx, x in ((w0[..., 0], lo[..., 0]), (w1[..., 0], hi[..., 0])):
        for wy, y in ((w0[..., 1], lo[..., 1]), (w1[..., 1], hi[..., 1])):
            for wz, z in ((w0[..., 2], lo[..., 2]), (w1[..., 2], hi[..., 2])):
                result += wx * wy * wz * grid[channels[None, :], x, y, z, 0]
    return result


def _inside(voxel, dim):
    return ((voxel >= 0) & (voxel < np.asarray(dim) - 1)).all(axis=-1)


def _lj(dsq, ff, at1, at2):
    rr2 = 1. / dsq
    vlj = (ff.rc[at1, at2] * rr2 - ff.ac[at1, at2]) * (rr2 * rr2 * rr2)
    ivor = ff.ivor[at1, at2]
    return np.where(dsq < ff.rmin2[at1, at2],
                    vlj + (ivor - 1.) * ff.emin[at1, at2], ivor * vlj)


def _direct_energies(coords, rec, rec_types, rec_q, lig_types, lig_q, ff,
                     ligand_chunk_size=128, receptor_chunk_size=512):
    """Full all-pairs LJ plus rdie electrostatics, with bounded pair buffers.

    LJ is evaluated at every distance. Electrostatics uses the existing shifted
    50 A cutoff and scaled-charge threshold. No plateau or neighbour list is used.
    """
    energies = np.zeros(len(coords), dtype=np.float64)
    for pose, lig in enumerate(coords):
        for start_lig in range(0, len(lig), ligand_chunk_size):
            stop_lig = start_lig + ligand_chunk_size
            at2 = lig_types[start_lig:stop_lig, None]
            for start_rec in range(0, len(rec), receptor_chunk_size):
                stop_rec = start_rec + receptor_chunk_size
                delta = lig[start_lig:stop_lig, None, :] - rec[None, start_rec:stop_rec, :]
                # Match the legacy explicit scorer at very short distances.
                dsq = np.maximum((delta * delta).sum(axis=-1), 0.001)
                at1 = rec_types[None, start_rec:stop_rec]
                e_lj = _lj(dsq, ff, at1, at2)
                charge = lig_q[start_lig:stop_lig, None] * rec_q[None, start_rec:stop_rec]
                charge = np.where(np.abs(charge) > 1e-3, charge, 0.)
                e_el = charge * np.maximum(1. / dsq - 1. / (50. * 50.), 0.)
                energies[pose] += (e_lj + e_el).sum()
    return energies


def _build_direct_kernel(rotation):
    """Use the shared oracle interface while requiring no grid data."""
    def evaluate(dofs, rec, rec_types, rec_q, lig, lig_types, lig_q, ff, pivot,
                 conformers=None):
        mats = _dofs_to_mats(dofs, pivot, rotation)
        coords = _transform(mats, lig if conformers is None else lig[conformers])
        energies = _direct_energies(coords, rec, rec_types, rec_q, lig_types, lig_q, ff)
        return energies.sum(), energies

    def main(dofs, rec, rec_types, rec_q, lig, lig_types, channels, lig_qraw,
             lig_q, ff, grid, nb_chunk_thresholds, grid_dim, lig_pivot):
        return evaluate(dofs, rec, rec_types, rec_q, lig, lig_types, lig_q, ff, lig_pivot)

    def pooled(dofs, rec, rec_types, rec_q, lig, conformers, lig_types, channels,
               lig_qraw, lig_q, ff, grid, nb_chunk_thresholds, grid_dim, lig_pivot):
        return evaluate(dofs, rec, rec_types, rec_q, lig, lig_types, lig_q, ff,
                        lig_pivot, conformers)

    main.pooled = pooled
    main.ad = main.pot_ad = main.pot_pooled = None
    return main


def build_kernel(
    grid, ff, lig_atomtypes_ff, lig_vdw_channel_idx, lig_charge_raw,
    lig_charge_scaled, cdie, padded_nb_size=0, max_nb_cap=0,
    use_precomputed_grid_gradients=True, rotation="euler", energy_only=False,
):
    """Build a forward-only NumPy kernel with the JAX scoring call signature."""
    if not energy_only:
        raise ValueError("NumPy kernels only support energy_only=True")
    if grid is None:
        if cdie:
            raise ValueError("Grid-free NumPy scoring currently supports rdie only")
        return _build_direct_kernel(rotation)
    pdsq = float(grid.plateaudis) ** 2
    charged = np.flatnonzero(np.abs(lig_charge_raw) > 1e-3)
    k = max(1, int(grid.max_nr_neighbours))
    if max_nb_cap > 0:
        k = min(k, max_nb_cap)
    rt, lt = np.indices(ff.rc.shape)
    plateau_lj = _lj(pdsq, ff, rt, lt)

    def elec_per_charge(dsq):
        if cdie:
            return np.maximum(1. / np.sqrt(dsq) - 1. / 50., 0.)
        return np.maximum(1. / dsq - 1. / (50. * 50.), 0.)

    plateau_el = elec_per_charge(pdsq)

    def potential(coords, channels, charges, grid0):
        inner = (coords - grid0.origin) / grid0.gridspacing
        outer = (inner + grid0.gridextension) / 2.
        in_inner, in_outer = _inside(inner, grid0.dim), _inside(outer, grid0.dim2)

        def sample(ch, vi, vo, mi, mo):
            ei = _interpolate(ch, vi, grid0.inner_potential_grid_all)
            eo = _interpolate(ch, vo, grid0.outer_potential_grid_all)
            return np.where(mi, ei, np.where(mo, eo, 0.))

        energies = sample(channels, inner, outer, in_inner, in_outer)
        if charged.size:
            ch = np.full(len(charged), grid0.elec_channel_index, dtype=np.intp)
            energies[:, charged] += charges[charged] * sample(
                ch, inner[:, charged], outer[:, charged],
                in_inner[:, charged], in_outer[:, charged],
            )
        return energies.sum(axis=1)

    def neighbours(coords, rec_coords, rec_types, rec_charges, lig_types, lig_charges, ff0, grid0):
        energies = np.zeros(len(coords), dtype=np.float64)
        vox = (coords - grid0.origin) / grid0.gridspacing
        pose, atom = np.nonzero(_inside(vox, grid0.dim))
        pos = np.floor(vox[pose, atom] + .5).astype(np.intp)
        ind = np.ravel_multi_index(pos.T, tuple(grid0.dim), mode="clip")
        active = np.asarray(grid0.nr_neighbours).reshape(-1)[ind] > 0
        pose, atom, ind = pose[active], atom[active], ind[active]
        if not len(pose):
            return energies
        rec_atoms = grid0.neighbour_grid_ravel[ind, :k]
        safe = np.where(rec_atoms < len(rec_coords), rec_atoms, 0)
        delta = coords[pose, atom, None, :] - rec_coords[safe]
        dsq = (delta * delta).sum(axis=-1)
        valid = (rec_atoms < 65535) & (dsq < pdsq)
        dsq = np.where(valid, dsq, 1.)
        at1, at2 = rec_types[safe], lig_types[atom, None]
        e_lj = _lj(dsq, ff0, at1, at2) - plateau_lj[at1, at2]
        charge = rec_charges[safe] * lig_charges[atom, None]
        charge = np.where(np.abs(charge) > 1e-3, charge, 0.)
        e_el = charge * (elec_per_charge(dsq) - plateau_el)
        atom_energy = np.where(valid, e_lj + e_el, 0.).sum(axis=1)
        np.add.at(energies, pose, atom_energy)
        return energies

    def evaluate(dofs, rec, rec_types, rec_q, lig, lig_types, channels, lig_qraw,
                 lig_q, ff0, grid0, pivot, conformers=None):
        mats = _dofs_to_mats(dofs, pivot, rotation)
        coords = _transform(mats, lig if conformers is None else lig[conformers])
        energies = potential(coords, channels, lig_qraw, grid0)
        energies += neighbours(coords, rec, rec_types, rec_q, lig_types, lig_q, ff0, grid0)
        return energies.sum(), energies

    def main(dofs, rec, rec_types, rec_q, lig, lig_types, channels, lig_qraw,
             lig_q, ff0, grid0, nb_chunk_thresholds, grid_dim, lig_pivot):
        return evaluate(dofs, rec, rec_types, rec_q, lig, lig_types, channels,
                        lig_qraw, lig_q, ff0, grid0, lig_pivot)

    def pooled(dofs, rec, rec_types, rec_q, lig, conformers, lig_types, channels,
               lig_qraw, lig_q, ff0, grid0, nb_chunk_thresholds, grid_dim, lig_pivot):
        return evaluate(dofs, rec, rec_types, rec_q, lig, lig_types, channels,
                        lig_qraw, lig_q, ff0, grid0, lig_pivot, conformers)

    def pot_ad(dofs, rec, rec_types, rec_q, lig, lig_types, channels, lig_qraw,
               lig_q, ff0, grid0, lig_pivot):
        coords = _transform(_dofs_to_mats(dofs, lig_pivot, rotation), lig)
        energies = potential(coords, channels, lig_qraw, grid0)
        return energies.sum(), energies

    def pot_pooled(dofs, lig, conformers, channels, lig_qraw, grid0, lig_pivot):
        coords = _transform(_dofs_to_mats(dofs, lig_pivot, rotation), lig[conformers])
        energies = potential(coords, channels, lig_qraw, grid0)
        return energies.sum(), energies

    main.ad = None
    main.pooled, main.pot_ad, main.pot_pooled = pooled, pot_ad, pot_pooled
    return main
