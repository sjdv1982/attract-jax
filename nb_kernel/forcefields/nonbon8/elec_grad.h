/**
 * elec_grad.h — Energy + gradient electrostatics for nonbon8 (rdie).
 *
 * Companion to elec.h; adds gradient computation for the gradient kernel path.
 *
 * Function signature (fixed contract; do not change):
 *
 *   void elec_grad(charge, rr2, dx, dy, dz, energy, gx, gy, gz)
 *
 * Parameters
 * ----------
 * charge : q_i * q_j, prescaled by felec (see elec.h)
 * rr2    : 1/dsq (Å^{-2})
 * dx     : (atom_lig_x - atom_rec_x) / dsq — scaled x-displacement
 * dy     : (atom_lig_y - atom_rec_y) / dsq — scaled y-displacement
 * dz     : (atom_lig_z - atom_rec_z) / dsq — scaled z-displacement
 *
 * Outputs
 * -------
 * energy : pairwise electrostatic energy contribution (kcal/mol)
 * gx, gy, gz : force components −dE/d(lig_xyz)  (force direction = negative
 *              gradient).  The pose loop negates these before writing to
 *              out_grad, yielding the gradient convention +dE/dx.
 *
 * Gradient derivation:
 *   E = charge * (rr2 - 1/2500)          [when rr2 > 1/2500, else 0]
 *   dE/d(lig_x) = charge * d(rr2)/d(lig_x)
 *               = charge * (-2 * lig_x - rec_x) / dsq^2
 *   Using dx = (lig_x - rec_x)/dsq: dE/d(lig_x) = 2 * charge * rr2 * dx
 */
#ifndef NONBON8_ELEC_GRAD_H
#define NONBON8_ELEC_GRAD_H

namespace nonbon8
{

    static inline void elec_grad(
        double charge, double rr2,
        double dx, double dy, double dz,
        double &energy, double &gx, double &gy, double &gz)
    {
        constexpr double rdie_thresh = (1.0 / 50.0) * (1.0 / 50.0);
        double dd = rr2 - rdie_thresh;
        if (dd < 0.0)
            dd = 0.0;

        energy = charge * dd;

        if (dd <= 0.0)
        {
            gx = 0.0;
            gy = 0.0;
            gz = 0.0;
            return;
        }

        const double et2 = charge * rr2;
        gx = 2.0 * et2 * dx;
        gy = 2.0 * et2 * dy;
        gz = 2.0 * et2 * dz;
    }

} // namespace nonbon8

#endif // NONBON8_ELEC_GRAD_H
