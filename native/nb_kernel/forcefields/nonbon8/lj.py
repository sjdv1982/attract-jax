"""
lj.py — Energy-only 8/6 Lennard-Jones for nonbon8.

JAX-traceable Python reference implementation that matches the C kernel
in forcefields/nonbon8/lj.h.

Physics (identical to lj.h):
    rr23 = rr2^3
    rep  = rc * rr2       (nonbon8: repulsive power = 8, rrd = rr2)
    vlj  = (rep - ac) * rr23
    E    = vlj + (ivor - 1) * emin   if dsq < rmin2
    E    = ivor * vlj                 otherwise
"""

import jax.numpy as jnp


def lj_energy(rc, ac, emin, rmin2, ivor, dsq, rr2):
    """Energy-only 8/6 LJ for a single atom pair (nonbon8, rdie).

    Parameters
    ----------
    rc     : repulsive coefficient  (array element, float)
    ac     : attractive coefficient (array element, float)
    emin   : energy at the LJ minimum (float)
    rmin2  : squared distance at the LJ minimum (float)
    ivor   : 1 for attractive pairs, 0 for repulsive (float-valued)
    dsq    : squared interatomic distance (float)
    rr2    : 1/dsq, precomputed (float)

    Returns
    -------
    energy : float — pairwise LJ energy contribution (kcal/mol)

    Notes
    -----
    Use jnp.where instead of Python if/else so this function is JAX-traceable
    and differentiable.  The C port (lj.h) uses a plain if branch, which is
    equivalent when tracing is not needed (C compiles to the same machine code
    as lj.h with -O3).
    """
    rr23 = rr2 * rr2 * rr2
    rep = rc * rr2
    vlj = (rep - ac) * rr23
    return jnp.where(
        dsq < rmin2,
        vlj + (ivor - 1.0) * emin,
        ivor * vlj,
    )
