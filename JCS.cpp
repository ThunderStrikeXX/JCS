#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

// =========== PARAMETERS ===========
const int    N = 100;           // Number of spatial cells [-]
const double L = 1.0;           // Domain length [m]
const double dx = L / N;        // Cell width [m]
const double CFL = 100.0;       // CFL number (implicit → no stability constraint) [-]
const double GAMMA = 1.4;       // Ratio of specific heats [-]
const double GRAVITY_X = 0.0;   // Axial gravity (set -9.81 for vertical) [m/s²]
const double AREA = 0.01;       // Constant cross-sectional area [m²]
const double R_GAS = 361.5;     // Specific gas constant for sodium vapor [J/kg·K]

// Friction model: F_friction = FRICTION_COEFF * rhoA * u * |u|
const double FRICTION_COEFF = 0.02;

// Newton-Raphson settings
const int    MAX_NEWTON_ITERS = 20;     // Max nonlinear iterations per timestep
const double NEWTON_TOL = 1e-6;         // Nonlinear residual convergence tolerance

// SparseLU is reused every REFACTOR_EVERY Newton iters to reduce factorization cost.
// 1 = always refactor (most robust); higher = faster but less accurate Jacobian.
const int REFACTOR_EVERY = 3;

// Write output every N timesteps
const int SAVE_EVERY = 100; 

using Vector3 = Eigen::Vector3d;
using Matrix3 = Eigen::Matrix3d;
using VectorGlobal = Eigen::VectorXd;

// =========== EQUATION OF STATE ===========

// Returns p*A from conserved variables Q = [rhoA, rhouA, rhoEA]
// Derived from: p = (gamma-1) * rho * e,  e = E - 0.5*u²
double get_pA(const Vector3& Q) {
    if (Q(0) < 1e-8) return 0.0;
    return (GAMMA - 1.0) * (Q(2) - 0.5 * Q(1) * Q(1) / Q(0));
}

// Returns speed of sound: c = sqrt(gamma * p / rho)
double get_sound_speed(const Vector3& Q) {
    double pA = get_pA(Q);
    double rhoA = Q(0);
    if (rhoA < 1e-8 || pA < 0.0) return 0.0;
    return std::sqrt(GAMMA * pA / rhoA);
}

// =========== FLUX AND SOURCE ===========

// Euler convective flux: F = [rhouA, (rhou²+p)A, u(rhoE+p)A]
Vector3 computeFlux(const Vector3& Q) {
    double pA = get_pA(Q);
    double u = Q(1) / Q(0);
    return { Q(1), Q(1) * u + pA, u * (Q(2) + pA) };
}

// Source terms: mass transfer (off), friction, gravity, heat transfer (off)
Vector3 computeSource(const Vector3& Q) {
    double rhoA = Q(0);
    double u = Q(1) / Q(0);

    double friction = FRICTION_COEFF * u * std::abs(u) * rhoA;

    Vector3 S;
    S(0) = 0.0;                                     // No mass transfer
    S(1) = -friction * AREA + rhoA * GRAVITY_X;     // Friction + gravity
    S(2) = rhoA * u * GRAVITY_X;                    // Gravity work (no heat transfer)
    return S;
}

// =========== JACOBIANS ===========

// Analytical flux Jacobian dF/dQ (Euler equations)
Matrix3 computeFluxJacobian(const Vector3& Q) {
    double u = Q(1) / Q(0);
    double u2 = u * u;
    double pA = get_pA(Q);
    double H = (Q(2) + pA) / Q(0);   // Total specific enthalpy
    double gm1 = GAMMA - 1.0;

    Matrix3 A;
    A(0, 0) = 0.0;                    A(0, 1) = 1.0;           A(0, 2) = 0.0;
    A(1, 0) = 0.5 * (gm1 - 3.0) * u2;    A(1, 1) = (3.0 - GAMMA) * u; A(1, 2) = gm1;
    A(2, 0) = u * (0.5 * gm1 * u2 - H);    A(2, 1) = H - gm1 * u2;    A(2, 2) = GAMMA * u;
    return A;
}

// Analytical source Jacobian dS/dQ (friction term only)
Matrix3 computeSourceJacobian(const Vector3& Q) {
    double u = Q(1) / Q(0);
    Matrix3 dS = Matrix3::Zero();
    dS(1, 1) = -FRICTION_COEFF * 2.0 * std::abs(u) * AREA;  // d(friction)/d(rhouA)
    return dS;
}

// =========== ADAPTIVE TIMESTEP ===========

// CFL-based dt: dt = CFL * dx / max(|u| + c)
// With implicit time integration, CFL > 1 is stable — use large CFL to accelerate.
double compute_dt(const VectorGlobal& Q) {
    double max_speed = 0.0;
    for (int i = 0; i < N; ++i) {
        Vector3 Qc = Q.segment<3>(3 * i);
        double speed = std::abs(Qc(1) / Qc(0)) + get_sound_speed(Qc);
        max_speed = std::max(max_speed, speed);
    }
    if (max_speed < 1e-12) return 1e-4;
    return CFL * dx / max_speed;
}

// =========== GHOST CELL BOUNDARY CONDITIONS ===========

// Left inlet: u=1 m/s (Dirichlet), T=350 K (Dirichlet), p=Neumann
Vector3 leftGhostCell(double p_in) {
    double u_b = 1.0;
    double T_b = 350.0;
    double rho_b = p_in / (R_GAS * T_b);
    double E_b = p_in / ((GAMMA - 1.0) * rho_b) + 0.5 * u_b * u_b;
    return { rho_b * AREA, rho_b * u_b * AREA, rho_b * E_b * AREA };
}

// Right outlet: p=10000 Pa (Dirichlet), T=300 K (Dirichlet), u=Neumann
Vector3 rightGhostCell(double u_in) {
    double p_b = 10000.0;
    double T_b = 300.0;
    double rho_b = p_b / (R_GAS * T_b);
    double E_b = p_b / ((GAMMA - 1.0) * rho_b) + 0.5 * u_in * u_in;
    return { rho_b * AREA, rho_b * u_in * AREA, rho_b * E_b * AREA };
}

// =========== MAIN ===========

int main() {

    // --- Initial conditions: uniform field at p=10000 Pa, T=300 K, u=1 m/s ---
    VectorGlobal Q_n(3 * N), Q_new(3 * N);
    {
        double p0 = 10000.0, T0 = 300.0, u0 = 1.0;
        double rho0 = p0 / (R_GAS * T0);
        double E0 = p0 / ((GAMMA - 1.0) * rho0) + 0.5 * u0 * u0;
        for (int i = 0; i < N; ++i) {
            Q_n(3 * i + 0) = rho0 * AREA;
            Q_n(3 * i + 1) = rho0 * u0 * AREA;
            Q_n(3 * i + 2) = rho0 * E0 * AREA;
        }
    }

    Q_new = Q_n;

    double t_final = 1.0, time = 0.0;
    int    step = 0;

    std::cout << "FVM Solver | N=" << N << " cells | system=" << 3 * N << " DOFs\n";

    std::ofstream file("history.csv");
    file << "time,x,rho,u,p,T,energy\n";

    // =========== TIME LOOP ===========
    while (time < t_final) {

        double dt = compute_dt(Q_n);
        if (time + dt > t_final) dt = t_final - time;

        Q_new = Q_n;   // Initial guess for Newton: previous timestep solution

        Eigen::SparseLU<Eigen::SparseMatrix<double>> lu_solver;
        bool factorized = false;

        // =========== NEWTON-RAPHSON ===========
        for (int iter = 0; iter < MAX_NEWTON_ITERS; ++iter) {

            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(N * 27);

            VectorGlobal Residual(3 * N);
            Residual.setZero();

            for (int i = 0; i < N; ++i) {

                Vector3 Uc = Q_new.segment<3>(3 * i);
                double u_in = Uc(1) / Uc(0);
                double p_in = get_pA(Uc) / AREA;

                // Neighbor states (internal or ghost)
                Vector3 Ul = (i > 0) ? Q_new.segment<3>(3 * (i - 1)) : leftGhostCell(p_in);
                Vector3 Ur = (i < N - 1) ? Q_new.segment<3>(3 * (i + 1)) : rightGhostCell(u_in);

                // --- JST numerical flux with scalar artificial dissipation ---
                Vector3 Fc = computeFlux(Uc);
                Vector3 Fl = computeFlux(Ul);
                Vector3 Fr = computeFlux(Ur);

                // Dissipation coefficient: eps * max spectral radius at each interface
                double sp_c = std::abs(u_in) + get_sound_speed(Uc);
                double sp_l = std::abs(Ul(1) / Ul(0)) + get_sound_speed(Ul);
                double sp_r = std::abs(Ur(1) / Ur(0)) + get_sound_speed(Ur);
                double eps = 0.5;
                double nu_l = eps * std::max(sp_c, sp_l);
                double nu_r = eps * std::max(sp_c, sp_r);

                Vector3 F_right = 0.5 * (Fc + Fr) - 0.5 * nu_r * (Ur - Uc);  // Right face flux
                Vector3 F_left = 0.5 * (Fl + Fc) - 0.5 * nu_l * (Uc - Ul);  // Left face flux

                // --- Cell residual: (dx/dt)*(Q-Q_n) + (F_R - F_L) - S*dx = 0 ---
                Vector3 R_cell = (Uc - Q_n.segment<3>(3 * i)) * (dx / dt)
                    + (F_right - F_left)
                    - computeSource(Uc) * dx;
                Residual.segment<3>(3 * i) = R_cell;

                // --- Jacobian blocks dR_i/dQ_j ---
                double nu_avg = 0.5 * (nu_l + nu_r);

                // Diagonal: time term + dissipation - source Jacobian
                Matrix3 J_diag = Matrix3::Identity() * (dx / dt)
                    + nu_avg * Matrix3::Identity()
                    - computeSourceJacobian(Uc) * dx;

                // Off-diagonal: flux Jacobian contributions at faces
                Matrix3 J_right = 0.5 * computeFluxJacobian(Ur) - 0.5 * nu_r * Matrix3::Identity();
                Matrix3 J_left = -0.5 * computeFluxJacobian(Ul) - 0.5 * nu_l * Matrix3::Identity();

                // Assemble triplets
                for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) {
                    triplets.push_back({ 3 * i + r, 3 * i + c, J_diag(r,c) });
                    if (i < N - 1) triplets.push_back({ 3 * i + r, 3 * (i + 1) + c, J_right(r,c) });
                    if (i > 0)   triplets.push_back({ 3 * i + r, 3 * (i - 1) + c, J_left(r,c) });
                }
            }

            // Check convergence before solving
            double res_norm = Residual.norm();
            if (res_norm < NEWTON_TOL) break;

            // Refactorize Jacobian every REFACTOR_EVERY iters (reuse otherwise)
            if (!factorized || iter % REFACTOR_EVERY == 0) {
                Eigen::SparseMatrix<double> J(3 * N, 3 * N);
                J.setFromTriplets(triplets.begin(), triplets.end());
                J.makeCompressed();
                lu_solver.compute(J);
                if (lu_solver.info() != Eigen::Success) {
                    std::cerr << "SparseLU failed at t=" << time << " iter=" << iter << "\n";
                    return -1;
                }
                factorized = true;
            }

            Q_new += lu_solver.solve(-Residual);
        }

        // --- Output ---
        if (step % SAVE_EVERY == 0) {
            std::cout << "t=" << time << " dt=" << dt << "\n";
            for (int i = 0; i < N; ++i) {
                Vector3 Q = Q_new.segment<3>(3 * i);
                double rho = Q(0) / AREA;
                double u = (Q(0) > 1e-8) ? Q(1) / Q(0) : 0.0;
                double p = get_pA(Q) / AREA;
                double T = (rho > 1e-8) ? p / (rho * R_GAS) : 0.0;
                double e = (Q(0) > 1e-8) ? Q(2) / Q(0) : 0.0;
                file << time << "," << (i + 0.5) * dx << "," << rho << ","
                    << u << "," << p << "," << T << "," << e << "\n";
            }
            file.flush();
        }

        Q_n = Q_new;
        time += dt;
        step++;
    }

    file.close();
    std::cout << "Done. " << step << " steps. Output: history.csv\n";
    return 0;
}