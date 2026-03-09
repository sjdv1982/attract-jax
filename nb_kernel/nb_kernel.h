#ifndef NB_KERNEL_H
#define NB_KERNEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  typedef struct
  {
    int32_t nposes;
    int32_t natoms;
    const double *dofs;       // nposes*6 (phi,ssi,rot,tx,ty,tz)
    const int16_t *ens;       // nposes (0-based ensemble index)
    const double *lig_coords; // natoms*3
    const double *lig_pivot;  // 3
    const int16_t *lig_type;  // natoms
    const double *lig_charge; // natoms
  } NbFusedStepData;

  typedef struct
  {
    int32_t dim[3];
    double origin[3];
    double spacing;
    const int32_t *nr_neigh; // dim[0]*dim[1]*dim[2]
    const int64_t *nb_start; // dim[0]*dim[1]*dim[2]
  } NbFusedGridData;

  typedef struct
  {
    int32_t nrec;
    int32_t nens;
    int64_t nb_concat_len;

    const int32_t *nb_concat; // nb_concat_len
    const double *rec_coord;  // nens*nrec*3
    const int16_t *rec_type;  // nrec
    const double *rec_charge; // nens*nrec

    int32_t nrec_types;
    int32_t nlig_types;
    const double *rc;    // nrec_types*nlig_types
    const double *ac;    // nrec_types*nlig_types
    const double *emin;  // nrec_types*nlig_types
    const double *rmin2; // nrec_types*nlig_types
    const int8_t *ivor;  // nrec_types*nlig_types

    double plateaudissq;
  } NbGlobalData;

  typedef struct
  {
    int32_t num_threads;
    int32_t kernel_variant; // 0 baseline, 1 fast
  } NbRunConfig;

  // Returns 0 on success, non-zero on invalid input.
  int nb_kernel_euler_grad(
      const NbFusedStepData *step,
      const NbFusedGridData *grid,
      const NbGlobalData *global,
      const NbRunConfig *cfg,
      double *out_energy, // nposes
      double *out_grad    // nposes*6
  );

  // Energy-only variant (out_grad may be NULL).
  int nb_kernel_euler_energy(
      const NbFusedStepData *step,
      const NbFusedGridData *grid,
      const NbGlobalData *global,
      const NbRunConfig *cfg,
      double *out_energy, // nposes
      double *out_grad    // unused (may be NULL)
  );

  // Rotvec rotation: dofs[0..2]=(v0,v1,v2) Rodrigues vector, dofs[3..5]=translation.
  int nb_kernel_rotvec_grad(
      const NbFusedStepData *step,
      const NbFusedGridData *grid,
      const NbGlobalData *global,
      const NbRunConfig *cfg,
      double *out_energy, // nposes
      double *out_grad    // nposes*6
  );

  int nb_kernel_rotvec_energy(
      const NbFusedStepData *step,
      const NbFusedGridData *grid,
      const NbGlobalData *global,
      const NbRunConfig *cfg,
      double *out_energy, // nposes
      double *out_grad    // unused (may be NULL)
  );

#ifdef __cplusplus
}
#endif

#endif
