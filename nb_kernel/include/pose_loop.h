/**
 * pose_loop.h — Template pose loop for the nonbonded C kernel.
 *
 * The single function template:
 *
 *   template <typename RotPolicy, typename FFPolicy, bool ComputeGrad,
 *             PlateauMode Mode = PlateauMode::Correction>
 *   inline void run_pose_loop_fused(...)
 *
 * is parameterized by four compile-time axes:
 *
 *   RotPolicy   — rotation DOF → matrix mapping (see euler_rot.h)
 *   FFPolicy    — nonbonded force field physics  (see forcefields/<ff>/)
 *   ComputeGrad — whether to compute gradients (true) or energy only (false)
 *   Mode        — plateau handling (Correction vs Clamp)
 *
 * When ComputeGrad=false the compiler eliminates:
 *   - pm2 computation and storage (dR/dq derivatives)
 *   - torque[3][3] matrix and its accumulation
 *   - all fhx/fhy/fhz force accumulation
 *   - torque contraction loop
 *   - gradient writes to out_grad
 *
 * RotPolicy must provide:
 *   static inline void rot_only (const double *dofs, double R[9])
 *   static inline void rot_torque(const double *dofs, double R[9], double pm2[3][3][3])
 *
 * FFPolicy must provide:
 *   static inline double lj_energy  (rc, ac, emin, rmin2, ivor, dsq, rr2)
 *   static inline void   lj_grad    (rc, ac, emin, rmin2, ivor, dsq, rr2, dx, dy, dz,
 *                                     &e, &gx, &gy, &gz)
 *   static inline double elec_energy(charge, rr2)
 *   static inline void   elec_grad  (charge, rr2, dx, dy, dz, &e, &gx, &gy, &gz)
 *
 * lj_grad / elec_grad are only instantiated when ComputeGrad=true.
 * A force field that only provides lj.h / elec.h (no *_grad.h files) is still
 * valid for energy-only kernels.
 *
 * This file is never modified when adding a new force field or rotation scheme.
 */
#ifndef POSE_LOOP_H
#define POSE_LOOP_H

#include "nb_kernel.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <omp.h>

namespace
{

    inline void apply_translation_force_reduction(double &gx, double &gy, double &gz)
    {
        const double flim = 1.0e18;
        for (int i = 0; i < 3; ++i)
        {
            const double fbetr = gx * gx + gy * gy + gz * gz;
            if (fbetr > flim)
            {
                gx *= 0.01;
                gy *= 0.01;
                gz *= 0.01;
            }
        }
    }

} // anonymous namespace

enum class PlateauMode
{
    Correction, // E(d) - E(plateau), current NB correction behavior
    Clamp       // E(max(d, plateau)), grid precomputation behavior
};

template <typename RotPolicy, typename FFPolicy, bool ComputeGrad,
          PlateauMode Mode = PlateauMode::Correction>
inline void run_pose_loop_fused(
    const NbFusedStepData *step,
    const NbFusedGridData *grid,
    const NbGlobalData *global,
    double *out_energy,
    double *out_grad)
{
    const double plateaudissq = global->plateaudissq;
    const double plateaudissqinv = 1.0 / plateaudissq;
    const int32_t nlig_types = global->nlig_types;
    const int32_t dim0 = grid->dim[0];
    const int32_t dim1 = grid->dim[1];
    const int32_t dim2 = grid->dim[2];
    const int64_t yz = static_cast<int64_t>(dim1) * static_cast<int64_t>(dim2);
    const double inv_spacing = 1.0 / grid->spacing;
    const double ox = grid->origin[0];
    const double oy = grid->origin[1];
    const double oz = grid->origin[2];
    const double pivot_x = step->lig_pivot[0];
    const double pivot_y = step->lig_pivot[1];
    const double pivot_z = step->lig_pivot[2];

#pragma omp parallel for schedule(guided, 16)
    for (int32_t p = 0; p < step->nposes; ++p)
    {
        const double *pose_dofs = step->dofs + static_cast<int64_t>(p) * 6;
        const int16_t ens = step->ens[p];
        if (ens < 0 || ens >= global->nens)
        {
            continue;
        }
        const int64_t rec_base = static_cast<int64_t>(ens) * global->nrec;

        // --- Rotation ---
        double R[9];
        double pm2[3][3][3]; // only written when ComputeGrad=true; eliminated otherwise
        if constexpr (ComputeGrad)
        {
            RotPolicy::rot_torque(pose_dofs, R, pm2);
        }
        else
        {
            RotPolicy::rot_only(pose_dofs, R);
        }

        const double tx = pose_dofs[3];
        const double ty = pose_dofs[4];
        const double tz = pose_dofs[5];

        // --- Per-pose accumulators ---
        double e_acc = 0.0;
        // Gradient accumulators: only live when ComputeGrad=true.
        double g0 = 0.0, g1 = 0.0, g2 = 0.0;
        double g3 = 0.0, g4 = 0.0, g5 = 0.0;
        double torque[3][3];
        if constexpr (ComputeGrad)
        {
            std::memset(torque, 0, sizeof(torque));
        }

        // --- Atom loop ---
        for (int32_t a = 0; a < step->natoms; ++a)
        {
            const int16_t lig_type = step->lig_type[a];
            if (lig_type < 0 || lig_type >= nlig_types)
            {
                continue;
            }

            const int64_t a3 = static_cast<int64_t>(a) * 3;
            const double *lig = step->lig_coords + a3;

            // Vector from pivot to ligand atom (pre-rotation)
            const double vx = lig[0] - pivot_x;
            const double vy = lig[1] - pivot_y;
            const double vz = lig[2] - pivot_z;

            // Apply rotation + translation → world-frame ligand atom position
            const double hx = vx * R[0] + vy * R[3] + vz * R[6] + tx + pivot_x;
            const double hy = vx * R[1] + vy * R[4] + vz * R[7] + ty + pivot_y;
            const double hz = vx * R[2] + vy * R[5] + vz * R[8] + tz + pivot_z;

            // Grid voxel lookup
            const int32_t ix = static_cast<int32_t>(std::floor((hx - ox) * inv_spacing + 0.5));
            const int32_t iy = static_cast<int32_t>(std::floor((hy - oy) * inv_spacing + 0.5));
            const int32_t iz = static_cast<int32_t>(std::floor((hz - oz) * inv_spacing + 0.5));
            if (ix < 0 || ix >= dim0 || iy < 0 || iy >= dim1 || iz < 0 || iz >= dim2)
            {
                continue;
            }

            const int64_t flat = static_cast<int64_t>(ix) * yz +
                                 static_cast<int64_t>(iy) * dim2 +
                                 static_cast<int64_t>(iz);
            const int32_t k = grid->nr_neigh[flat];
            if (k <= 0)
            {
                continue;
            }
            const int64_t n = grid->nb_start[flat];
            if (n < 0 || n >= global->nb_concat_len ||
                n + static_cast<int64_t>(k) > global->nb_concat_len)
            {
                continue;
            }

            const double lig_charge = step->lig_charge[a];
            double fhx = 0.0, fhy = 0.0, fhz = 0.0; // eliminated when !ComputeGrad
            const int32_t *nb_ptr = global->nb_concat + n;

            // --- Neighbor loop ---
            for (int32_t j = 0; j < k; ++j)
            {
                const int64_t nb_i64 = n + static_cast<int64_t>(j);
                if (nb_i64 < 0 || nb_i64 >= global->nb_concat_len)
                {
                    continue;
                }

                const int32_t rec_idx = nb_ptr[j];
                if (rec_idx < 0 || rec_idx >= global->nrec)
                {
                    continue;
                }

                const int64_t rec3 = (rec_base + rec_idx) * 3;
                const double *rxyz = global->rec_coord + rec3;

                const double dx0 = hx - rxyz[0];
                const double dy0 = hy - rxyz[1];
                const double dz0 = hz - rxyz[2];
                const double dsq = dx0 * dx0 + dy0 * dy0 + dz0 * dz0;
                const bool within_plateau = dsq <= plateaudissq;
                if constexpr (Mode == PlateauMode::Correction)
                {
                    if (!within_plateau)
                    {
                        continue;
                    }
                }

                const double rr2 = 1.0 / dsq;

                const int16_t rec_type = global->rec_type[rec_idx];
                if (rec_type < 0 || rec_type >= global->nrec_types)
                {
                    continue;
                }
                const int64_t tindex = static_cast<int64_t>(rec_type) * nlig_types +
                                       static_cast<int64_t>(lig_type);

                double e_pair;
                double gx_pair = 0.0, gy_pair = 0.0, gz_pair = 0.0; // eliminated when !ComputeGrad

                if constexpr (ComputeGrad)
                {
                    // Scaled displacement: (atom_i - atom_j) / dsq
                    const double sdx = dx0 * rr2;
                    const double sdy = dy0 * rr2;
                    const double sdz = dz0 * rr2;

                    if constexpr (Mode == PlateauMode::Correction)
                    {
                        // Plateau displacement: scale toward plateau surface.
                        const double ratio = std::sqrt(dsq * plateaudissqinv);
                        const double pdx = sdx * ratio;
                        const double pdy = sdy * ratio;
                        const double pdz = sdz * ratio;

                        // LJ energy + gradient, with plateau correction
                        double e0, gx0, gy0, gz0;
                        FFPolicy::lj_grad(
                            global->rc[tindex], global->ac[tindex],
                            global->emin[tindex], global->rmin2[tindex],
                            static_cast<int>(global->ivor[tindex]),
                            dsq, rr2, sdx, sdy, sdz,
                            e0, gx0, gy0, gz0);

                        double ep, gxp, gyp, gzp;
                        FFPolicy::lj_grad(
                            global->rc[tindex], global->ac[tindex],
                            global->emin[tindex], global->rmin2[tindex],
                            static_cast<int>(global->ivor[tindex]),
                            plateaudissq, plateaudissqinv, pdx, pdy, pdz,
                            ep, gxp, gyp, gzp);

                        e_pair = e0 - ep;
                        gx_pair = gx0 - gxp;
                        gy_pair = gy0 - gyp;
                        gz_pair = gz0 - gzp;

                        // Electrostatics (gradient path)
                        const double charge = lig_charge * global->rec_charge[rec_base + rec_idx];
                        if (std::fabs(charge) > 1.0e-3)
                        {
                            double ee0, egx0, egy0, egz0;
                            FFPolicy::elec_grad(charge, rr2, sdx, sdy, sdz,
                                                ee0, egx0, egy0, egz0);

                            double eep, egxp, egyp, egzp;
                            FFPolicy::elec_grad(charge, plateaudissqinv, pdx, pdy, pdz,
                                                eep, egxp, egyp, egzp);

                            e_pair += (ee0 - eep);
                            gx_pair += (egx0 - egxp);
                            gy_pair += (egy0 - egyp);
                            gz_pair += (egz0 - egzp);
                        }
                    }
                    else
                    {
                        // Clamp mode: evaluate at actual distance unless inside plateau,
                        // in which case use the projected plateau distance.
                        const bool use_plateau = within_plateau;
                        const double ratio = use_plateau ? std::sqrt(dsq * plateaudissqinv) : 1.0;
                        const double eval_dsq = use_plateau ? plateaudissq : dsq;
                        const double eval_rr2 = use_plateau ? plateaudissqinv : rr2;
                        const double eval_dx = sdx * ratio;
                        const double eval_dy = sdy * ratio;
                        const double eval_dz = sdz * ratio;

                        FFPolicy::lj_grad(
                            global->rc[tindex], global->ac[tindex],
                            global->emin[tindex], global->rmin2[tindex],
                            static_cast<int>(global->ivor[tindex]),
                            eval_dsq, eval_rr2, eval_dx, eval_dy, eval_dz,
                            e_pair, gx_pair, gy_pair, gz_pair);

                        const double charge = lig_charge * global->rec_charge[rec_base + rec_idx];
                        if (std::fabs(charge) > 1.0e-3)
                        {
                            double ee, egx, egy, egz;
                            FFPolicy::elec_grad(charge, eval_rr2, eval_dx, eval_dy, eval_dz,
                                                ee, egx, egy, egz);
                            e_pair += ee;
                            gx_pair += egx;
                            gy_pair += egy;
                            gz_pair += egz;
                        }
                    }
                }
                else
                {
                    if constexpr (Mode == PlateauMode::Correction)
                    {
                        // Energy-only correction path.
                        const double e0 = FFPolicy::lj_energy(
                            global->rc[tindex], global->ac[tindex],
                            global->emin[tindex], global->rmin2[tindex],
                            static_cast<int>(global->ivor[tindex]),
                            dsq, rr2);

                        const double ep = FFPolicy::lj_energy(
                            global->rc[tindex], global->ac[tindex],
                            global->emin[tindex], global->rmin2[tindex],
                            static_cast<int>(global->ivor[tindex]),
                            plateaudissq, plateaudissqinv);

                        e_pair = e0 - ep;

                        // Electrostatics (energy-only path)
                        const double charge = lig_charge * global->rec_charge[rec_base + rec_idx];
                        if (std::fabs(charge) > 1.0e-3)
                        {
                            const double ee0 = FFPolicy::elec_energy(charge, rr2);
                            const double eep = FFPolicy::elec_energy(charge, plateaudissqinv);
                            e_pair += (ee0 - eep);
                        }
                    }
                    else
                    {
                        const double eval_dsq = within_plateau ? plateaudissq : dsq;
                        const double eval_rr2 = within_plateau ? plateaudissqinv : rr2;
                        e_pair = FFPolicy::lj_energy(
                            global->rc[tindex], global->ac[tindex],
                            global->emin[tindex], global->rmin2[tindex],
                            static_cast<int>(global->ivor[tindex]),
                            eval_dsq, eval_rr2);

                        const double charge = lig_charge * global->rec_charge[rec_base + rec_idx];
                        if (std::fabs(charge) > 1.0e-3)
                        {
                            e_pair += FFPolicy::elec_energy(charge, eval_rr2);
                        }
                    }
                }

                e_acc += e_pair;

                // Force accumulation: only in gradient path
                if constexpr (ComputeGrad)
                {
                    fhx += gx_pair;
                    fhy += gy_pair;
                    fhz += gz_pair;
                }

            } // neighbor loop

            // Torque accumulation: only in gradient path
            if constexpr (ComputeGrad)
            {
                torque[0][0] += vx * fhx;
                torque[1][0] += vx * fhy;
                torque[2][0] += vx * fhz;
                torque[0][1] += vy * fhx;
                torque[1][1] += vy * fhy;
                torque[2][1] += vy * fhz;
                torque[0][2] += vz * fhx;
                torque[1][2] += vz * fhy;
                torque[2][2] += vz * fhz;
                g3 += fhx;
                g4 += fhy;
                g5 += fhz;
            }

        } // atom loop

        // --- Write outputs ---
        out_energy[p] = e_acc;

        if constexpr (ComputeGrad)
        {
            // Torque → rotational gradient: g_j = sum_{k,l} pm2[k][j][l] * torque[k][l]
            for (int rot_j = 0; rot_j < 3; ++rot_j)
            {
                double gj = 0.0;
                for (int kk = 0; kk < 3; ++kk)
                {
                    for (int l = 0; l < 3; ++l)
                    {
                        gj += pm2[kk][rot_j][l] * torque[kk][l];
                    }
                }
                if (rot_j == 0)
                    g0 = gj;
                if (rot_j == 1)
                    g1 = gj;
                if (rot_j == 2)
                    g2 = gj;
            }

            apply_translation_force_reduction(g3, g4, g5);

            out_grad[6 * p + 0] = -g0;
            out_grad[6 * p + 1] = -g1;
            out_grad[6 * p + 2] = -g2;
            out_grad[6 * p + 3] = -g3;
            out_grad[6 * p + 4] = -g4;
            out_grad[6 * p + 5] = -g5;
        }

    } // pose loop
}

#endif // POSE_LOOP_H
