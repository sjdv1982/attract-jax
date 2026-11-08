// Full nonbon8 energies, without potential grids, neighbour lists, or plateaus.
#include "nb_kernel.h"
#include "include/euler_rot.h"
#include "include/rotvec_rot.h"
#include "forcefields/nonbon8/lj.h"
#include "forcefields/nonbon8/elec.h"
#include <algorithm>
#include <cmath>
#include <omp.h>

template <typename Rotation>
static int score_gridless(const NbFusedStepData *step, const NbGlobalData *global,
                          const NbRunConfig *cfg, const int32_t *conformers,
                          int32_t nconformers, double *energies)
{
    if (!step || !global || !cfg || !energies || !conformers ||
        step->nposes < 0 || step->natoms <= 0 || global->nrec <= 0 ||
        global->nens <= 0 || nconformers <= 0 || !step->dofs || !step->ens ||
        !step->lig_coords || !step->lig_pivot || !step->lig_type ||
        !step->lig_charge || !global->rec_coord || !global->rec_type ||
        !global->rec_charge || !global->rc || !global->ac || !global->emin ||
        !global->rmin2 || !global->ivor)
        return 1;
    for (int32_t p = 0; p < step->nposes; ++p)
        if (step->ens[p] < 0 || step->ens[p] >= global->nens ||
            conformers[p] < 0 || conformers[p] >= nconformers)
            return 2;
    const int threads = cfg->num_threads > 0 ? cfg->num_threads : omp_get_max_threads();
#pragma omp parallel for schedule(guided, 16) num_threads(threads)
    for (int32_t p = 0; p < step->nposes; ++p)
    {
        const double *dof = step->dofs + int64_t(p) * 6;
        const int64_t rec_base = int64_t(step->ens[p]) * global->nrec;
        const double *ligand = step->lig_coords +
            int64_t(conformers[p]) * step->natoms * 3;
        double rot[9];
        Rotation::rot_only(dof, rot);
        double energy = 0.0;
        for (int32_t a = 0; a < step->natoms; ++a)
        {
            const double *lig = ligand + int64_t(a) * 3;
            const double x = lig[0] - step->lig_pivot[0];
            const double y = lig[1] - step->lig_pivot[1];
            const double z = lig[2] - step->lig_pivot[2];
            const double hx = x * rot[0] + y * rot[3] + z * rot[6] + dof[3] + step->lig_pivot[0];
            const double hy = x * rot[1] + y * rot[4] + z * rot[7] + dof[4] + step->lig_pivot[1];
            const double hz = x * rot[2] + y * rot[5] + z * rot[8] + dof[5] + step->lig_pivot[2];
            for (int32_t r = 0; r < global->nrec; ++r)
            {
                const double *rec = global->rec_coord + (rec_base + r) * 3;
                const double dx = hx - rec[0], dy = hy - rec[1], dz = hz - rec[2];
                const double dsq = std::max(dx * dx + dy * dy + dz * dz, 0.001);
                const double rr2 = 1.0 / dsq;
                const int64_t pair = int64_t(global->rec_type[r]) * global->nlig_types + step->lig_type[a];
                energy += nonbon8::lj_energy(global->rc[pair], global->ac[pair],
                    global->emin[pair], global->rmin2[pair], global->ivor[pair], dsq, rr2);
                const double charge = step->lig_charge[a] * global->rec_charge[rec_base + r];
                if (std::fabs(charge) > 1.0e-3)
                    energy += nonbon8::elec_energy(charge, rr2);
            }
        }
        energies[p] = energy;
    }
    return 0;
}

extern "C" int nb_kernel_gridless_rotvec_energy(
    const NbFusedStepData *step, const NbGlobalData *global, const NbRunConfig *cfg,
    const int32_t *conformers, int32_t nconformers, double *energies)
{
    return score_gridless<RotVecRot>(step, global, cfg, conformers, nconformers, energies);
}

extern "C" int nb_kernel_gridless_euler_energy(
    const NbFusedStepData *step, const NbGlobalData *global, const NbRunConfig *cfg,
    const int32_t *conformers, int32_t nconformers, double *energies)
{
    return score_gridless<EulerRot>(step, global, cfg, conformers, nconformers, energies);
}
