/**
 * rotvec_rot.h — Rodrigues rotation-vector rotation policy for the pose loop template.
 *
 * Provides the RotVecRot struct conforming to the RotPolicy concept:
 *
 *   struct RotPolicy {
 *     static inline void rot_only(const double *dofs, double R[9]);
 *     static inline void rot_torque(const double *dofs, double R[9], double pm2[3][3][3]);
 *   };
 *
 * DOF layout: dofs[0..2]=rotvec (v0,v1,v2), dofs[3..5]=translation.
 *
 * The rotvec v encodes a rotation of angle theta=||v|| about axis k=v/theta
 * using the Rodrigues formula (same convention as scipy Rotation.from_rotvec).
 *
 * R[i*3+j] stores M_rv[i][j] = Rodrigues(v)^T[i][j] so that the pose loop
 * computes h_j = sum_i v_i * M_rv[i][j] = (Rodrigues(v) @ x)_j, matching the
 * Euler convention in euler_rot.h.
 *
 * pm2[k][j][l] = dM_rv[k][l] / d(v_j)  for j in {0,1,2}.
 * Torque contraction is identical to EulerRot:
 *   g_j = sum_{k,l} pm2[k][j][l] * torque[k][l]
 *
 * Small-angle branch (||v|| < 1e-8):
 *   rot_only  : M_rv ≈ I − Khat_v  (first-order, Khat_v = cross-product matrix of v)
 *   rot_torque: pm2[r][j][s] ≈ eps_{rsj}  (Levi-Civita symbol)
 */
#ifndef ROTVEC_ROT_H
#define ROTVEC_ROT_H

#include <cmath>

namespace
{
    // Levi-Civita symbol eps_{rsj}: +1 for even, -1 for odd, 0 for repeated indices.
    inline double lc3(int r, int s, int j) noexcept
    {
        if (r == s || s == j || r == j) return 0.0;
        // Positive permutations of (0,1,2): (0,1,2), (1,2,0), (2,0,1)
        if ((r == 0 && s == 1 && j == 2) ||
            (r == 1 && s == 2 && j == 0) ||
            (r == 2 && s == 0 && j == 1)) return 1.0;
        return -1.0;
    }
} // anonymous namespace

struct RotVecRot
{
    /**
     * rot_only — compute M_rv = Rodrigues(v)^T from rotvec DOFs.
     * Called by run_pose_loop_fused when ComputeGrad=false.
     */
    static inline void rot_only(const double *dofs, double R[9])
    {
        const double v0 = dofs[0];
        const double v1 = dofs[1];
        const double v2 = dofs[2];
        const double theta = std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);

        if (theta < 1.0e-8)
        {
            // First-order: M_rv ≈ I − Khat_v
            R[0] = 1.0;   R[1] = v2;    R[2] = -v1;
            R[3] = -v2;   R[4] = 1.0;   R[5] = v0;
            R[6] = v1;    R[7] = -v0;   R[8] = 1.0;
            return;
        }

        const double inv_t = 1.0 / theta;
        const double k0 = v0 * inv_t;
        const double k1 = v1 * inv_t;
        const double k2 = v2 * inv_t;
        const double s = std::sin(theta);
        const double c = std::cos(theta);
        const double omc = 1.0 - c;

        // M_rv = Rodrigues(v)^T = c*I + (1-c)*k⊗k − s*Khat
        // Khat = [[0,-k2,k1],[k2,0,-k0],[-k1,k0,0]]
        // − s*Khat: [0,+s*k2,-s*k1],[-s*k2,0,+s*k0],[+s*k1,-s*k0,0]
        R[0] = c + omc * k0 * k0;
        R[1] = omc * k0 * k1 + s * k2;
        R[2] = omc * k0 * k2 - s * k1;
        R[3] = omc * k1 * k0 - s * k2;
        R[4] = c + omc * k1 * k1;
        R[5] = omc * k1 * k2 + s * k0;
        R[6] = omc * k2 * k0 + s * k1;
        R[7] = omc * k2 * k1 - s * k0;
        R[8] = c + omc * k2 * k2;
    }

    /**
     * rot_torque — compute M_rv AND dM_rv/dv derivatives.
     * Called by run_pose_loop_fused when ComputeGrad=true.
     *
     * pm2[r][j][s] = dM_rv[r][s] / dv_j
     *
     * Derivation: M_rv = Rodrigues(v)^T.
     * With c=cos(t), s=sin(t), t=||v||, k=v/t, omc=1-c:
     *
     *   dM_rv[r][s]/dv_j = −s*k_j*δ_rs
     *                    − k_j*(c − s/t)*Khat[r][s]
     *                    + k_j*(s − 2*omc/t)*k_r*k_s
     *                    + (omc/t)*(δ_{rj}*k_s + δ_{sj}*k_r)
     *                    + (s/t)*ε_{rsj}
     *
     * Small-angle limit (||v|| < 1e-8): pm2[r][j][s] → ε_{rsj}.
     */
    static inline void rot_torque(const double *dofs, double R[9], double pm2[3][3][3])
    {
        const double v0 = dofs[0];
        const double v1 = dofs[1];
        const double v2 = dofs[2];
        const double theta = std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);

        if (theta < 1.0e-8)
        {
            // First-order rotation matrix
            R[0] = 1.0;   R[1] = v2;    R[2] = -v1;
            R[3] = -v2;   R[4] = 1.0;   R[5] = v0;
            R[6] = v1;    R[7] = -v0;   R[8] = 1.0;
            // Zeroth-order pm2: pm2[r][j][s] = ε_{rsj}
            for (int r = 0; r < 3; ++r)
                for (int j = 0; j < 3; ++j)
                    for (int s = 0; s < 3; ++s)
                        pm2[r][j][s] = lc3(r, s, j);
            return;
        }

        const double inv_t = 1.0 / theta;
        const double k0 = v0 * inv_t;
        const double k1 = v1 * inv_t;
        const double k2 = v2 * inv_t;
        const double s = std::sin(theta);
        const double c = std::cos(theta);
        const double omc = 1.0 - c;

        // Rotation matrix (same as rot_only)
        R[0] = c + omc * k0 * k0;
        R[1] = omc * k0 * k1 + s * k2;
        R[2] = omc * k0 * k2 - s * k1;
        R[3] = omc * k1 * k0 - s * k2;
        R[4] = c + omc * k1 * k1;
        R[5] = omc * k1 * k2 + s * k0;
        R[6] = omc * k2 * k0 + s * k1;
        R[7] = omc * k2 * k1 - s * k0;
        R[8] = c + omc * k2 * k2;

        // Precompute scalars used in pm2
        const double s_over_t   = s * inv_t;
        const double omc_over_t = omc * inv_t;
        const double c_m_s_t    = c - s_over_t;          // c − s/t
        const double s_m_2omc_t = s - 2.0 * omc_over_t; // s − 2*(1-c)/t

        // Khat[r][s] = [[0,-k2,k1],[k2,0,-k0],[-k1,k0,0]]
        const double Khat[3][3] = {
            {0.0,  -k2,  k1},
            {k2,   0.0, -k0},
            {-k1,  k0,  0.0}
        };
        const double k[3] = {k0, k1, k2};

        for (int r = 0; r < 3; ++r)
        {
            for (int j = 0; j < 3; ++j)
            {
                for (int ss = 0; ss < 3; ++ss)
                {
                    double val = 0.0;
                    // -s*k_j*δ_rs
                    if (r == ss) val -= s * k[j];
                    // -k_j*(c - s/t)*Khat[r][s]
                    val -= k[j] * c_m_s_t * Khat[r][ss];
                    // +k_j*(s - 2*omc/t)*k_r*k_s
                    val += k[j] * s_m_2omc_t * k[r] * k[ss];
                    // +(omc/t)*(δ_{rj}*k_s + δ_{sj}*k_r)
                    if (r == j) val += omc_over_t * k[ss];
                    if (ss == j) val += omc_over_t * k[r];
                    // +(s/t)*ε_{rsj}
                    val += s_over_t * lc3(r, ss, j);
                    pm2[r][j][ss] = val;
                }
            }
        }
    }
};

#endif // ROTVEC_ROT_H
