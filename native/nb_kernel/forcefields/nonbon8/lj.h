/**
 * lj.h — Energy-only 8/6 Lennard-Jones evaluation for nonbon8.
 *
 * Force field: ATTRACT nonbon8, rdie (distance-dependent dielectric).
 * Repulsive exponent: 8.  Attractive exponent: 6.
 *
 * Function signature (fixed contract; do not change):
 *
 *   double lj_energy(rc, ac, emin, rmin2, ivor, dsq, rr2)
 *
 * Parameters
 * ----------
 * rc    : repulsive coefficient  (units: Å^8 * kcal/mol^(1/2) / atom-pair)
 * ac    : attractive coefficient (units: Å^6 * kcal/mol^(1/2) / atom-pair)
 * emin  : energy at the LJ minimum for this type pair (kcal/mol)
 * rmin2 : squared distance at the LJ minimum (Å^2)
 * ivor  : 0 for purely repulsive pair, 1 for attractive pair
 * dsq   : squared interatomic distance (Å^2)
 * rr2   : 1/dsq (Å^{-2}), precomputed by the pose loop
 *
 * Returns: pairwise LJ energy contribution (kcal/mol)
 *
 * This file is the energy-only half of the nonbon8 LJ interaction.
 * The gradient version is in lj_grad.h (required for gradient-capable kernels).
 *
 * Physics note:
 *   E_lj = (rc/r^8 - ac/r^6)   for r^2 >= rmin2
 *   E_lj = (rc/r^8 - ac/r^6) + (ivor - 1) * emin   for r^2 < rmin2
 * The second branch handles the plateau/minimum region.
 */
#ifndef NONBON8_LJ_H
#define NONBON8_LJ_H

namespace nonbon8
{

    static inline double lj_energy(
        double rc, double ac, double emin, double rmin2,
        int ivor, double dsq, double rr2)
    {
        const double rr23 = rr2 * rr2 * rr2;
        // nonbon8: rrd = rr2 (repulsive power = 8, so r^{-8} = rr2 * rr2^3)
        const double rep = rc * rr2;
        const double vlj = (rep - ac) * rr23;

        if (dsq < rmin2)
        {
            return vlj + (static_cast<double>(ivor) - 1.0) * emin;
        }
        return static_cast<double>(ivor) * vlj;
    }

} // namespace nonbon8

#endif // NONBON8_LJ_H
