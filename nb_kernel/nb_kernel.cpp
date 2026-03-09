/**
 * nb_kernel_nonbon8.cpp — Milestone 2 manually-generated boilerplate.
 *
 * This file is the hand-written equivalent of what codegen_ff.py (Milestone 3)
 * will generate automatically from the force field templates.
 *
 * Structure:
 *   - Includes: policy headers + forcefield headers + pose loop template
 *   - Nonbon8FF: FFPolicy struct delegating to nonbon8::lj_energy/elec_energy etc.
 *   - nb_kernel_euler_grad:   energy+gradient, Euler rotation, nonbon8 FF
 *   - nb_kernel_euler_energy: energy-only,     Euler rotation, nonbon8 FF
 */
#include "nb_kernel.h"

// Rotation policy
#include "include/euler_rot.h"

// Nonbon8 force field physics headers
#include "forcefields/nonbon8/lj.h"
#include "forcefields/nonbon8/lj_grad.h"
#include "forcefields/nonbon8/elec.h"
#include "forcefields/nonbon8/elec_grad.h"

// Pose loop template (must come after all physics headers and rotation policies)
#include "include/pose_loop.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <omp.h>

// ---------------------------------------------------------------------------
// FFPolicy: Nonbon8 (8/6 LJ + rdie electrostatics)
// Delegates to the pure physics functions in forcefields/nonbon8/*.h.
// codegen_ff.py (Milestone 3) will generate this struct automatically.
// ---------------------------------------------------------------------------
struct Nonbon8FF
{
  static inline double lj_energy(
      double rc, double ac, double emin, double rmin2,
      int ivor, double dsq, double rr2)
  {
    return nonbon8::lj_energy(rc, ac, emin, rmin2, ivor, dsq, rr2);
  }

  static inline void lj_grad(
      double rc, double ac, double emin, double rmin2,
      int ivor, double dsq, double rr2,
      double dx, double dy, double dz,
      double &energy, double &gx, double &gy, double &gz)
  {
    nonbon8::lj_grad(rc, ac, emin, rmin2, ivor, dsq, rr2,
                     dx, dy, dz, energy, gx, gy, gz);
  }

  static inline double elec_energy(double charge, double rr2)
  {
    return nonbon8::elec_energy(charge, rr2);
  }

  static inline void elec_grad(
      double charge, double rr2,
      double dx, double dy, double dz,
      double &energy, double &gx, double &gy, double &gz)
  {
    nonbon8::elec_grad(charge, rr2, dx, dy, dz, energy, gx, gy, gz);
  }
};

// ---------------------------------------------------------------------------
// Input validation helpers
// ---------------------------------------------------------------------------
namespace
{

  inline int validate_grad_inputs(
      const NbFusedStepData *step, const NbFusedGridData *grid,
      const NbGlobalData *global, const NbRunConfig *cfg,
      double *out_energy, double *out_grad)
  {
    if (!step || !grid || !global || !cfg || !out_energy || !out_grad)
    {
      return 1;
    }
    if (step->nposes < 0 || step->natoms < 0 || global->nrec <= 0 || global->nens <= 0)
    {
      return 2;
    }
    if (!step->dofs || !step->ens || !step->lig_coords || !step->lig_pivot ||
        !step->lig_type || !step->lig_charge)
    {
      return 3;
    }
    if (!grid->nr_neigh || !grid->nb_start || grid->spacing <= 0.0 ||
        grid->dim[0] <= 0 || grid->dim[1] <= 0 || grid->dim[2] <= 0)
    {
      return 4;
    }
    if (!global->nb_concat || !global->rec_coord || !global->rec_type ||
        !global->rec_charge || !global->rc || !global->ac || !global->emin ||
        !global->rmin2 || !global->ivor)
    {
      return 5;
    }
    if (global->nrec_types <= 0 || global->nlig_types <= 0)
    {
      return 6;
    }
    if (global->plateaudissq <= 0.0)
    {
      return 7;
    }
    return 0;
  }

  // Energy-only validation: out_grad may be NULL.
  inline int validate_energy_inputs(
      const NbFusedStepData *step, const NbFusedGridData *grid,
      const NbGlobalData *global, const NbRunConfig *cfg,
      double *out_energy)
  {
    if (!step || !grid || !global || !cfg || !out_energy)
    {
      return 1;
    }
    if (step->nposes < 0 || step->natoms < 0 || global->nrec <= 0 || global->nens <= 0)
    {
      return 2;
    }
    if (!step->dofs || !step->ens || !step->lig_coords || !step->lig_pivot ||
        !step->lig_type || !step->lig_charge)
    {
      return 3;
    }
    if (!grid->nr_neigh || !grid->nb_start || grid->spacing <= 0.0 ||
        grid->dim[0] <= 0 || grid->dim[1] <= 0 || grid->dim[2] <= 0)
    {
      return 4;
    }
    if (!global->nb_concat || !global->rec_coord || !global->rec_type ||
        !global->rec_charge || !global->rc || !global->ac || !global->emin ||
        !global->rmin2 || !global->ivor)
    {
      return 5;
    }
    if (global->nrec_types <= 0 || global->nlig_types <= 0)
    {
      return 6;
    }
    if (global->plateaudissq <= 0.0)
    {
      return 7;
    }
    return 0;
  }

} // namespace

// ---------------------------------------------------------------------------
// Codegen-equivalent exported symbols — Milestone 2 manual boilerplate.
// Naming convention: nb_kernel_<rot>_<grad|energy>
// ---------------------------------------------------------------------------

/**
 * nb_kernel_euler_grad — energy + gradient, Euler rotation, nonbon8 force field.
 * Primary gradient-capable entry point (codegen canonical name).
 */
extern "C" int nb_kernel_euler_grad(
    const NbFusedStepData *step,
    const NbFusedGridData *grid,
    const NbGlobalData *global,
    const NbRunConfig *cfg,
    double *out_energy,
    double *out_grad)
{
  const int rc = validate_grad_inputs(step, grid, global, cfg, out_energy, out_grad);
  if (rc != 0)
    return rc;

  std::memset(out_energy, 0, static_cast<size_t>(step->nposes) * sizeof(double));
  std::memset(out_grad, 0, static_cast<size_t>(step->nposes) * 6 * sizeof(double));

  const int32_t nthreads = cfg->num_threads > 0 ? cfg->num_threads : 1;
  omp_set_num_threads(nthreads);
  run_pose_loop_fused<EulerRot, Nonbon8FF, true>(
      step, grid, global, out_energy, out_grad);
  return 0;
}

/**
 * nb_kernel_euler_energy — energy only, Euler rotation, nonbon8 force field.
 * out_grad is accepted for ABI uniformity but not written to; may be NULL.
 */
extern "C" int nb_kernel_euler_energy(
    const NbFusedStepData *step,
    const NbFusedGridData *grid,
    const NbGlobalData *global,
    const NbRunConfig *cfg,
    double *out_energy,
    double *out_grad) /* not written; may be NULL */
{
  const int rc = validate_energy_inputs(step, grid, global, cfg, out_energy);
  if (rc != 0)
    return rc;

  std::memset(out_energy, 0, static_cast<size_t>(step->nposes) * sizeof(double));

  const int32_t nthreads = cfg->num_threads > 0 ? cfg->num_threads : 1;
  omp_set_num_threads(nthreads);
  run_pose_loop_fused<EulerRot, Nonbon8FF, false>(
      step, grid, global, out_energy, nullptr);
  (void)out_grad;
  return 0;
}