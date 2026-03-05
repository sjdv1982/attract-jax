/**
 * lj_grad.h — Energy + gradient 8/6 Lennard-Jones evaluation for nonbon8.
 *
 * Force field: ATTRACT nonbon8, rdie (distance-dependent dielectric).
 *
 * Function signature (fixed contract; do not change):
 *
 *   void lj_grad(rc, ac, emin, rmin2, ivor, dsq, rr2, dx, dy, dz,
 *                energy, gx, gy, gz)
 *
 * Parameters
 * ----------
 * rc, ac, emin, rmin2, ivor : same as lj_energy (see lj.h)
 * dsq   : squared interatomic distance (Å^2)
 * rr2   : 1/dsq (Å^{-2})
 * dx    : (atom_lig_x - atom_rec_x) / dsq — scaled x-displacement
 * dy    : (atom_lig_y - atom_rec_y) / dsq — scaled y-displacement
 * dz    : (atom_lig_z - atom_rec_z) / dsq — scaled z-displacement
 *
 * Outputs
 * -------
 * energy : pairwise LJ energy (kcal/mol)
 * gx, gy, gz : force components −dE/d(lig_xyz)  (force direction = negative
 *              gradient).  The pose loop negates these before writing to
 *              out_grad, yielding the gradient convention +dE/dx stored in
 *              grid channels 1–3.
 *
 * Gradient derivation:
 *   E_lj = (rc * rr2 - ac) * rr2^3
 *   fb   = 6 * vlj + 2 * rc * rr2 * rr2^3   (= -dE/d(dsq) * 2 * dsq)
 *   g_x  = fb * dx  where dx = delta_x / dsq
 */
#ifndef NONBON8_LJ_GRAD_H
#define NONBON8_LJ_GRAD_H

namespace nonbon8
{

    static inline void lj_grad(
        double rc, double ac, double emin, double rmin2,
        int ivor, double dsq, double rr2,
        double dx, double dy, double dz,
        double &energy, double &gx, double &gy, double &gz)
    {
        const double shapedelta = 2.0; // repulsive exponent - attractive exponent: 8 - 6 = 2
        const double rr23 = rr2 * rr2 * rr2;
        const double rep = rc * rr2;
        const double vlj = (rep - ac) * rr23;

        if (dsq < rmin2)
        {
            energy = vlj + (static_cast<double>(ivor) - 1.0) * emin;
            const double fb = 6.0 * vlj + shapedelta * (rep * rr23);
            gx = fb * dx;
            gy = fb * dy;
            gz = fb * dz;
        }
        else
        {
            const double iv = static_cast<double>(ivor);
            energy = iv * vlj;
            const double fb = 6.0 * vlj + shapedelta * (rep * rr23);
            gx = iv * fb * dx;
            gy = iv * fb * dy;
            gz = iv * fb * dz;
        }
    }

} // namespace nonbon8

#endif // NONBON8_LJ_GRAD_H
