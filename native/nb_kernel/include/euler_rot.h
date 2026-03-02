/**
 * euler_rot.h — Euler-angle rotation policy for the pose loop template.
 *
 * Provides the EulerRot struct conforming to the RotPolicy concept:
 *
 *   struct RotPolicy {
 *     static inline void rot_only(const double *dofs, double R[9]);
 *     static inline void rot_torque(const double *dofs, double R[9], double pm2[3][3][3]);
 *   };
 *
 * DOF layout: dofs[0]=phi, dofs[1]=ssi, dofs[2]=rot, dofs[3..5]=translation.
 *
 * Rotation convention: row-major 3×3, matching the Python path
 *   coord_world = (coord_local - pivot) @ R + translation + pivot
 *
 * pm2[k][j][l] = dR[k][l] / d(q_j)  for j ∈ {0,1,2} (Euler angle index).
 * Torque contraction:  g_j = sum_{k,l} pm2[k][j][l] * torque[k][l]
 */
#ifndef EULER_ROT_H
#define EULER_ROT_H

#include <cmath>

struct EulerRot
{
    /**
     * rot_only — compute the 3×3 rotation matrix from Euler DOFs.
     * Called by run_pose_loop_fused when ComputeGrad=false.
     * Skips all pm2 computation.
     */
    static inline void rot_only(const double *dofs, double R[9])
    {
        const double phi = dofs[0];
        const double ssi = dofs[1];
        const double rot = dofs[2];

        const double cp = std::cos(phi);
        const double sp = std::sin(phi);
        const double cs = std::cos(ssi);
        const double ss = std::sin(ssi);
        const double cr = std::cos(rot);
        const double sr = std::sin(rot);

        R[0] = cr * cs * cp + sr * sp;
        R[1] = cr * cs * sp - sr * cp;
        R[2] = -cr * ss;
        R[3] = sr * cs * cp - cr * sp;
        R[4] = sr * cs * sp + cr * cp;
        R[5] = -sr * ss;
        R[6] = ss * cp;
        R[7] = ss * sp;
        R[8] = cs;
    }

    /**
     * rot_torque — compute rotation matrix AND dR/dq derivatives.
     * Called by run_pose_loop_fused when ComputeGrad=true.
     */
    static inline void rot_torque(const double *dofs, double R[9], double pm2[3][3][3])
    {
        const double phi = dofs[0];
        const double ssi = dofs[1];
        const double rot = dofs[2];

        const double cp = std::cos(phi);
        const double sp = std::sin(phi);
        const double cs = std::cos(ssi);
        const double ss = std::sin(ssi);
        const double cr = std::cos(rot);
        const double sr = std::sin(rot);

        R[0] = cr * cs * cp + sr * sp;
        R[1] = cr * cs * sp - sr * cp;
        R[2] = -cr * ss;
        R[3] = sr * cs * cp - cr * sp;
        R[4] = sr * cs * sp + cr * cp;
        R[5] = -sr * ss;
        R[6] = ss * cp;
        R[7] = ss * sp;
        R[8] = cs;

        // pm2[k][j][l] = dR[k][l] / d(q_j)
        // j=0: d/d(phi)
        pm2[0][0][0] = -cr * cs * sp + sr * cp;
        pm2[0][0][1] = -sr * cs * sp - cr * cp;
        pm2[0][0][2] = -ss * sp;
        pm2[1][0][0] = cr * cs * cp + sr * sp;
        pm2[1][0][1] = sr * cs * cp - cr * sp;
        pm2[1][0][2] = ss * cp;
        pm2[2][0][0] = 0.0;
        pm2[2][0][1] = 0.0;
        pm2[2][0][2] = 0.0;

        // j=1: d/d(ssi)
        pm2[0][1][0] = -cr * ss * cp;
        pm2[0][1][1] = -sr * ss * cp;
        pm2[0][1][2] = cs * cp;
        pm2[1][1][0] = -cr * ss * sp;
        pm2[1][1][1] = -sr * ss * sp;
        pm2[1][1][2] = cs * sp;
        pm2[2][1][0] = -cr * cs;
        pm2[2][1][1] = -sr * cs;
        pm2[2][1][2] = -ss;

        // j=2: d/d(rot)
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
};

#endif // EULER_ROT_H
