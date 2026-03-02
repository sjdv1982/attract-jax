"""
params.py — Force field parameter loading for nonbon8.

Follows the Section 5.3 parameter module contract:
  load_params(npz_path) -> FFParams

The FFParams namedtuple exposes all per-type-pair arrays needed by lj_energy.
Electrostatics uses per-atom charges (not per-pair), so they are not included
here; they are loaded separately from the PDB file by the framework.

The .npz file (attract-par.npz) is produced by
attract-original/convert-attract-par.py from the legacy attract.par file.
"""

from collections import namedtuple

import numpy as np

FFParams = namedtuple("FFParams", ("rc", "ac", "ivor", "emin", "rmin2"))


def load_params(npz_path):
    """Load nonbon8 force field parameters from an NPZ file.

    Parameters
    ----------
    npz_path : str or Path
        Path to an attract-par.npz file produced by convert-attract-par.py.

    Returns
    -------
    FFParams
        Namedtuple with arrays of shape (nrec_types, nlig_types):
        - rc     : repulsive coefficient
        - ac     : attractive coefficient
        - ivor   : 1.0 for attractive pairs, 0.0 for repulsive
        - emin   : energy at the LJ minimum  = -27 * ac^4 / (256 * rc^3)
        - rmin2  : squared distance at min   = 4 * rc / (3 * ac)

    Notes
    -----
    emin and rmin2 are derived quantities computed here to match exactly the
    formulas used in jax_scorer.py and minfor_nb.py.  This ensures numerical
    consistency across all force field evaluation paths.
    """
    par = np.load(npz_path)
    rc = par["rc"]
    ac = par["ac"]
    ivor = par["ivor"]
    emin = -27.0 * ac**4 / (256.0 * rc**3)
    rmin2 = 4.0 * rc / (3.0 * ac)
    return FFParams(
        rc=rc.astype(np.float64, copy=False),
        ac=ac.astype(np.float64, copy=False),
        ivor=ivor.astype(np.float64, copy=False),
        emin=emin.astype(np.float64, copy=False),
        rmin2=rmin2.astype(np.float64, copy=False),
    )
