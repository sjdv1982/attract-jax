/**
 * elec.h — Energy-only electrostatics for nonbon8 (rdie: distance-dependent dielectric).
 *
 * Dielectric model: ε(r) ∝ r  (rdie), hardcoded for nonbon8.
 * The effective potential is E_elec = q_ij * max(0, rr2 - (1/50)^2)
 * where q_ij = q_i * q_j (product of prescaled charges).
 *
 * Cutoff: atoms further than 50 Å contribute zero electrostatic energy.
 * In practice the neighbor grid cutoff (plateaudissq) is applied first,
 * and the hard 50 Å cap handles the residual range inside the plateau distance.
 *
 * Function signature (fixed contract; do not change):
 *
 *   double elec_energy(charge, rr2)
 *
 * Parameters
 * ----------
 * charge : q_i * q_j, prescaled by felec = sqrt(332.054 / epsilon) (kcal/mol)^{1/2}
 * rr2    : 1/dsq (Å^{-2}), precomputed by the pose loop
 *
 * Returns: pairwise electrostatic energy contribution (kcal/mol)
 */
#ifndef NONBON8_ELEC_H
#define NONBON8_ELEC_H

namespace nonbon8
{

    static inline double elec_energy(double charge, double rr2)
    {
        // rdie: E = charge * (rr2 - 1/50^2), floored at 0.
        // Equivalent to: charge * max(0, 1/r^2 - 1/r_cut^2)
        constexpr double rdie_thresh = (1.0 / 50.0) * (1.0 / 50.0); // 1/r_cut^2
        double dd = rr2 - rdie_thresh;
        if (dd < 0.0)
            dd = 0.0;
        return charge * dd;
    }

} // namespace nonbon8

#endif // NONBON8_ELEC_H
