"""
elec_grad.py — Energy + gradient rdie electrostatics for nonbon8.

JAX-traceable Python reference implementation that matches the C kernel
in forcefields/nonbon8/elec_grad.h.

Physics (rdie model, identical to elec_grad.h):
    dd = max(0, rr2 - (1/50)²)
    E  = charge * dd
    gx = 2 * charge * rr2 * dx   (when dd > 0, else 0)
"""

import jax.numpy as jnp

_RDIE_THRESH = (1.0 / 50.0) ** 2  # 1/r_cut² = 4×10⁻⁴ Å⁻²


def elec_grad(charge, rr2, dx, dy, dz):
    """Energy and gradient for a single nonbon8 electrostatic pair.

    Parameters
    ----------
    charge : float — product of prescaled atomic charges (q_i * q_j * felec²)
    rr2    : float — 1/dsq, precomputed
    dx, dy, dz : (lig_xyz - rec_xyz) / dsq — pre-scaled displacement components

    Returns
    -------
    energy : float
    gx, gy, gz : float — force components −dE/d(lig_xyz)
        (force direction, i.e. negative gradient; callers negate to get
        the gradient +dE/dx for storage in grid channels 1–3)

    Notes
    -----
    Uses jnp.where for JAX-traceability.  The C port (elec_grad.h) uses a plain
    if branch; both produce identical floating-point results.
    """
    dd = rr2 - _RDIE_THRESH
    active = dd > 0.0
    dd_pos = jnp.where(active, dd, 0.0)
    energy = charge * dd_pos
    g = jnp.where(active, 2.0 * charge * rr2, 0.0)
    return energy, g * dx, g * dy, g * dz
