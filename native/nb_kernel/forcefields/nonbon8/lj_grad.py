"""
lj_grad.py — Energy + gradient 8/6 Lennard-Jones for nonbon8.

JAX-traceable Python reference implementation that matches the C kernel
in forcefields/nonbon8/lj_grad.h.

Physics (identical to lj_grad.h):
    rr23 = rr2^3
    rep  = rc * rr2
    vlj  = (rep - ac) * rr23
    fb_inner = 6*vlj + 2*rep*rr23    (= -dE/d(dsq) * 2*dsq for the inner branch)
    fb   = fb_inner             if dsq < rmin2
    fb   = ivor * fb_inner      otherwise
    gx   = fb * dx              (dx = delta_x / dsq)
"""

import jax.numpy as jnp


def lj_grad(rc, ac, emin, rmin2, ivor, dsq, rr2, dx, dy, dz):
    """Energy and gradient for a single nonbon8 LJ pair.

    Parameters
    ----------
    rc, ac, emin, rmin2, ivor : same as lj_energy (see lj.py)
    dsq   : squared interatomic distance (Å²)
    rr2   : 1/dsq, precomputed (Å⁻²)
    dx, dy, dz : (lig_xyz - rec_xyz) / dsq — pre-scaled displacement components

    Returns
    -------
    energy : float — pairwise LJ energy (kcal/mol)
    gx, gy, gz : float — force components −dE/d(lig_xyz)
        (force direction, i.e. negative gradient; callers negate to get
        the gradient +dE/dx for storage in grid channels 1–3)

    Notes
    -----
    Uses jnp.where for JAX-traceability.  The C port (lj_grad.h) uses plain
    if/else branches; both produce identical floating-point results.
    """
    rr23 = rr2 * rr2 * rr2
    rep = rc * rr2
    vlj = (rep - ac) * rr23

    energy = jnp.where(
        dsq < rmin2,
        vlj + (ivor - 1.0) * emin,
        ivor * vlj,
    )

    fb_inner = 6.0 * vlj + 2.0 * rep * rr23
    fb = jnp.where(dsq < rmin2, fb_inner, ivor * fb_inner)

    return energy, fb * dx, fb * dy, fb * dz
