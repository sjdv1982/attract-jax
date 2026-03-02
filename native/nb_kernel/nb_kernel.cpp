#include "nb_kernel.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <omp.h>

namespace {

inline void apply_translation_force_reduction(double &gx, double &gy, double &gz) {
  const double flim = 1.0e18;
  for (int i = 0; i < 3; ++i) {
    const double fbetr = gx * gx + gy * gy + gz * gz;
    if (fbetr > flim) {
      gx *= 0.01;
      gy *= 0.01;
      gz *= 0.01;
    }
  }
}

inline void nonbon_eval(
    double welwel,
    double rc,
    double ac,
    double emin,
    double rmin2,
    int ivor,
    double dsq,
    double rr2,
    double dx,
    double dy,
    double dz,
    int potshape,
    double &energy,
    double &gx,
    double &gy,
    double &gz) {
  const double alen = welwel * ac;
  const double rlen = welwel * rc;
  const double rr23 = rr2 * rr2 * rr2;

  double rrd = rr2;
  double shapedelta = 2.0;
  if (potshape == 12) {
    rrd = rr23;
    shapedelta = 6.0;
  }

  const double rep = rlen * rrd;
  const double vlj = (rep - alen) * rr23;

  if (dsq < rmin2) {
    energy = vlj + (static_cast<double>(ivor) - 1.0) * emin;
    const double fb = 6.0 * vlj + shapedelta * (rep * rr23);
    gx = fb * dx;
    gy = fb * dy;
    gz = fb * dz;
  } else {
    const double iv = static_cast<double>(ivor);
    energy = iv * vlj;
    const double fb = 6.0 * vlj + shapedelta * (rep * rr23);
    gx = iv * fb * dx;
    gy = iv * fb * dy;
    gz = iv * fb * dz;
  }
}

inline void elec_eval(
    int cdie,
    double charge,
    double rr2,
    double dx,
    double dy,
    double dz,
    double &energy,
    double &gx,
    double &gy,
    double &gz) {
  double dd;
  if (cdie) {
    dd = std::sqrt(rr2) - 1.0 / 50.0;
  } else {
    dd = rr2 - (1.0 / 50.0) * (1.0 / 50.0);
  }
  if (dd < 0.0) {
    dd = 0.0;
  }

  energy = charge * dd;

  if (dd <= 0.0) {
    gx = 0.0;
    gy = 0.0;
    gz = 0.0;
    return;
  }

  if (cdie) {
    const double et2 = charge * std::sqrt(rr2);
    gx = et2 * dx;
    gy = et2 * dy;
    gz = et2 * dz;
  } else {
    const double et2 = charge * rr2;
    gx = 2.0 * et2 * dx;
    gy = 2.0 * et2 * dy;
    gz = 2.0 * et2 * dz;
  }
}

inline void euler_rot_torque(
    double phi,
    double ssi,
    double rot,
    double R[9],
    double pm2[3][3][3]) {
  const double cp = std::cos(phi);
  const double sp = std::sin(phi);
  const double cs = std::cos(ssi);
  const double ss = std::sin(ssi);
  const double cr = std::cos(rot);
  const double sr = std::sin(rot);

  // Row-major 3x3 matrix matching Python path (coord = v @ R + t + pivot).
  R[0] = cr * cs * cp + sr * sp;
  R[1] = cr * cs * sp - sr * cp;
  R[2] = -cr * ss;
  R[3] = sr * cs * cp - cr * sp;
  R[4] = sr * cs * sp + cr * cp;
  R[5] = -sr * ss;
  R[6] = ss * cp;
  R[7] = ss * sp;
  R[8] = cs;

  // pm2 is dR/d(q_j) in ATTRACT euler2torquemat index order: pm2[k][j][l].
  pm2[0][0][0] = -cr * cs * sp + sr * cp;
  pm2[0][0][1] = -sr * cs * sp - cr * cp;
  pm2[0][0][2] = -ss * sp;
  pm2[1][0][0] = cr * cs * cp + sr * sp;
  pm2[1][0][1] = sr * cs * cp - cr * sp;
  pm2[1][0][2] = ss * cp;
  pm2[2][0][0] = 0.0;
  pm2[2][0][1] = 0.0;
  pm2[2][0][2] = 0.0;

  pm2[0][1][0] = -cr * ss * cp;
  pm2[0][1][1] = -sr * ss * cp;
  pm2[0][1][2] = cs * cp;
  pm2[1][1][0] = -cr * ss * sp;
  pm2[1][1][1] = -sr * ss * sp;
  pm2[1][1][2] = cs * sp;
  pm2[2][1][0] = -cr * cs;
  pm2[2][1][1] = -sr * cs;
  pm2[2][1][2] = -ss;

  pm2[0][2][0] = -sr * cs * cp + cr * sp;
  pm2[0][2][1] = cr * cs * cp + sr * sp;
  pm2[0][2][2] = 0.0;
  pm2[1][2][0] = -sr * cs * sp - cr * cp;
  pm2[1][2][1] = cr * cs * sp - sr * cp;
  pm2[1][2][2] = 0.0;
  pm2[2][2][0] = sr * ss;
  pm2[2][2][1] = -cr * ss;
  pm2[2][2][2] = 0.0;
}

inline int validate_inputs_fused(const NbFusedStepData *step, const NbFusedGridData *grid,
                                 const NbGlobalData *global, const NbRunConfig *cfg,
                                 double *out_energy, double *out_grad) {
  if (!step || !grid || !global || !cfg || !out_energy || !out_grad) {
    return 1;
  }
  if (step->nposes < 0 || step->natoms < 0 || global->nrec <= 0 || global->nens <= 0) {
    return 2;
  }
  if (!step->dofs || !step->ens || !step->lig_coords || !step->lig_pivot ||
      !step->lig_type || !step->lig_charge) {
    return 3;
  }
  if (!grid->nr_neigh || !grid->nb_start || grid->spacing <= 0.0 ||
      grid->dim[0] <= 0 || grid->dim[1] <= 0 || grid->dim[2] <= 0) {
    return 4;
  }
  if (!global->nb_concat || !global->rec_coord || !global->rec_type ||
      !global->rec_charge || !global->rc || !global->ac || !global->emin ||
      !global->rmin2 || !global->ivor) {
    return 5;
  }
  if (global->nrec_types <= 0 || global->nlig_types <= 0) {
    return 6;
  }
  if (global->plateaudissq <= 0.0) {
    return 7;
  }
  return 0;
}

inline void run_pose_loop_fused(const NbFusedStepData *step,
                                const NbFusedGridData *grid,
                                const NbGlobalData *global,
                                double *out_energy,
                                double *out_grad) {
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
  for (int32_t p = 0; p < step->nposes; ++p) {
    const double *pose_dofs = step->dofs + static_cast<int64_t>(p) * 6;
    const int16_t ens = step->ens[p];
    if (ens < 0 || ens >= global->nens) {
      continue;
    }
    const int64_t rec_base = static_cast<int64_t>(ens) * global->nrec;

    double R[9];
    double pm2[3][3][3];
    euler_rot_torque(pose_dofs[0], pose_dofs[1], pose_dofs[2], R, pm2);
    const double tx = pose_dofs[3];
    const double ty = pose_dofs[4];
    const double tz = pose_dofs[5];

    double e_acc = 0.0;
    double g0 = 0.0, g1 = 0.0, g2 = 0.0, g3 = 0.0, g4 = 0.0, g5 = 0.0;
    double torque[3][3];
    std::memset(torque, 0, sizeof(torque));

    for (int32_t a = 0; a < step->natoms; ++a) {
      const int16_t lig_type = step->lig_type[a];
      if (lig_type < 0 || lig_type >= nlig_types) {
        continue;
      }

      const int64_t a3 = static_cast<int64_t>(a) * 3;
      const double *lig = step->lig_coords + a3;
      const double lig_charge = step->lig_charge[a];

      const double vx = lig[0] - pivot_x;
      const double vy = lig[1] - pivot_y;
      const double vz = lig[2] - pivot_z;

      const double hx = vx * R[0] + vy * R[3] + vz * R[6] + tx + pivot_x;
      const double hy = vx * R[1] + vy * R[4] + vz * R[7] + ty + pivot_y;
      const double hz = vx * R[2] + vy * R[5] + vz * R[8] + tz + pivot_z;

      const int32_t ix = static_cast<int32_t>(std::floor((hx - ox) * inv_spacing + 0.5));
      const int32_t iy = static_cast<int32_t>(std::floor((hy - oy) * inv_spacing + 0.5));
      const int32_t iz = static_cast<int32_t>(std::floor((hz - oz) * inv_spacing + 0.5));
      if (ix < 0 || ix >= dim0 || iy < 0 || iy >= dim1 || iz < 0 || iz >= dim2) {
        continue;
      }

      const int64_t flat = static_cast<int64_t>(ix) * yz +
                           static_cast<int64_t>(iy) * dim2 +
                           static_cast<int64_t>(iz);
      const int32_t k = grid->nr_neigh[flat];
      if (k <= 0) {
        continue;
      }
      const int64_t n = grid->nb_start[flat];
      if (n < 0 || n >= global->nb_concat_len ||
          n + static_cast<int64_t>(k) > global->nb_concat_len) {
        continue;
      }

      double fhx = 0.0;
      double fhy = 0.0;
      double fhz = 0.0;

      const int32_t *nb_ptr = global->nb_concat + n;
      for (int32_t j = 0; j < k; ++j) {
        const int64_t nb_i64 = n + static_cast<int64_t>(j);
        if (nb_i64 < 0 || nb_i64 >= global->nb_concat_len) {
          continue;
        }

        const int32_t rec_idx = nb_ptr[j];
        if (rec_idx < 0 || rec_idx >= global->nrec) {
          continue;
        }

        const int64_t rec3 = (rec_base + rec_idx) * 3;
        const double *rxyz = global->rec_coord + rec3;

        const double dx0 = hx - rxyz[0];
        const double dy0 = hy - rxyz[1];
        const double dz0 = hz - rxyz[2];
        const double dsq = dx0 * dx0 + dy0 * dy0 + dz0 * dz0;
        if (dsq > plateaudissq) {
          continue;
        }

        const double rr2 = 1.0 / dsq;
        const double sdx = dx0 * rr2;
        const double sdy = dy0 * rr2;
        const double sdz = dz0 * rr2;

        const int16_t rec_type = global->rec_type[rec_idx];
        if (rec_type < 0 || rec_type >= global->nrec_types) {
          continue;
        }
        const int64_t tindex = static_cast<int64_t>(rec_type) * nlig_types +
                               static_cast<int64_t>(lig_type);

        double e0, gx0, gy0, gz0;
        nonbon_eval(1.0, global->rc[tindex], global->ac[tindex], global->emin[tindex],
                    global->rmin2[tindex], static_cast<int>(global->ivor[tindex]),
                    dsq, rr2, sdx, sdy, sdz, global->potshape, e0, gx0, gy0, gz0);

        const double ratio = std::sqrt(dsq * plateaudissqinv);
        const double pdx = sdx * ratio;
        const double pdy = sdy * ratio;
        const double pdz = sdz * ratio;

        double ep, gxp, gyp, gzp;
        nonbon_eval(1.0, global->rc[tindex], global->ac[tindex], global->emin[tindex],
                    global->rmin2[tindex], static_cast<int>(global->ivor[tindex]),
                    plateaudissq, plateaudissqinv, pdx, pdy, pdz, global->potshape,
                    ep, gxp, gyp, gzp);

        double gx = gx0 - gxp;
        double gy = gy0 - gyp;
        double gz = gz0 - gzp;
        double e_pair = e0 - ep;

        const double charge = lig_charge * global->rec_charge[rec_base + rec_idx];
        if (std::fabs(charge) > 1.0e-3) {
          double ee0, egx0, egy0, egz0;
          elec_eval(global->cdie, charge, rr2, sdx, sdy, sdz, ee0, egx0, egy0, egz0);

          double eep, egxp, egyp, egzp;
          elec_eval(global->cdie, charge, plateaudissqinv, pdx, pdy, pdz, eep, egxp,
                    egyp, egzp);

          e_pair += (ee0 - eep);
          gx += (egx0 - egxp);
          gy += (egy0 - egyp);
          gz += (egz0 - egzp);
        }

        e_acc += e_pair;
        fhx += gx;
        fhy += gy;
        fhz += gz;
      }

      // torque[k][l] = coord_l * force_k, accumulated per pose.
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

    for (int j = 0; j < 3; ++j) {
      double gj = 0.0;
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          gj += pm2[k][j][l] * torque[k][l];
        }
      }
      if (j == 0) g0 = gj;
      if (j == 1) g1 = gj;
      if (j == 2) g2 = gj;
    }

    apply_translation_force_reduction(g3, g4, g5);

    out_energy[p] = e_acc;
    out_grad[6 * p + 0] = -g0;
    out_grad[6 * p + 1] = -g1;
    out_grad[6 * p + 2] = -g2;
    out_grad[6 * p + 3] = -g3;
    out_grad[6 * p + 4] = -g4;
    out_grad[6 * p + 5] = -g5;
  }
}

}  // namespace

extern "C" int nb_kernel_run_fused(const NbFusedStepData *step, const NbFusedGridData *grid,
                                    const NbGlobalData *global, const NbRunConfig *cfg,
                                    double *out_energy, double *out_grad) {
  const int rc = validate_inputs_fused(step, grid, global, cfg, out_energy, out_grad);
  if (rc != 0) {
    return rc;
  }

  std::memset(out_energy, 0, static_cast<size_t>(step->nposes) * sizeof(double));
  std::memset(out_grad, 0, static_cast<size_t>(step->nposes) * 6 * sizeof(double));

  const int32_t nthreads = cfg->num_threads > 0 ? cfg->num_threads : 1;
  omp_set_num_threads(nthreads);
  run_pose_loop_fused(step, grid, global, out_energy, out_grad);
  return 0;
}
