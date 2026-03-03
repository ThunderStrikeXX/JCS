#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <fstream>

// =========== PARAMETERS
const int N = 100;                      // Number of spatial cells [-]
const double L = 1.0;                   // Length of the domain [m]
const double dx = L / N;                // Axial length of the volumes [m]
const double CFL = 100;                 // CFL Number [-]
const double gamma = 1.4;               // Ratio of specific heats (Ideal Gas) [-]
const double gravity_x = 0.0;           // g_x (Set to -9.81 if vertical) [m/s2]
const double section = 0.01;            // Constant Area A_v [m2]
const double R_vapor = 361.5;           // Sodium vapor constant [J/kgK]

// Newton-Krylov settings
const int max_newton_iters = 20;        // Maximum number of outer iterations [-]
const double newton_tol = 1e-6;         // Maximum non-linear residual tolerated [-]

// Performance settings
const int refactor_every = 3;           // Re-factorize Jacobian every N Newton iters
// (1 = always refactor, higher = faster but less robust)

using Vector3 = Eigen::Vector3d;
using Matrix3 = Eigen::Matrix3d;
using VectorGlobal = Eigen::VectorXd;

// =========== PHYSICS AND STATE HELPERS

double get_pA(const Vector3& Q) {
    if (Q(0) < 1e-8) return 0.0;
    return (gamma - 1.0) * (Q(2) - 0.5 * Q(1) * Q(1) / Q(0));
}

double get_sound_speed(const Vector3& Q) {
    double pA = get_pA(Q);
    double rhoA = Q(0);
    if (rhoA < 1e-8 || pA < 0.0) return 0.0;
    return std::sqrt(gamma * pA / rhoA);
}

Vector3 computeFlux(const Vector3& Q) {
    double pA = get_pA(Q);
    double u = Q(1) / Q(0);
    Vector3 F;
    F(0) = Q(1);
    F(1) = Q(1) * u + pA;
    F(2) = u * (Q(2) + pA);
    return F;
}

Vector3 computeSource(const Vector3& Q) {
    Vector3 S = Vector3::Zero();
    double rhoA = Q(0);
    double u = Q(1) / Q(0);

    double Gamma_int = 0.0;
    double A_prime_int = 0.0;
    double F_v_friction = 0.02 * u * std::abs(u) * rhoA;
    double q_v_int = 0.0;
    double E_v_int = 0.0;

    S(0) = Gamma_int * A_prime_int;
    S(1) = -F_v_friction * section + rhoA * gravity_x;
    S(2) = (rhoA * u * gravity_x) + (q_v_int + Gamma_int * E_v_int) * A_prime_int;
    return S;
}

// =========== JACOBIANS

Matrix3 computeFluxJacobian(const Vector3& Q) {
    Matrix3 A;
    double q1 = Q(0);
    double q2 = Q(1);
    double q3 = Q(2);
    double u = q2 / q1;
    double u2 = u * u;
    double pA = get_pA(Q);
    double H = (q3 + pA) / q1;
    double gm1 = gamma - 1.0;

    A(0, 0) = 0.0;              A(0, 1) = 1.0;              A(0, 2) = 0.0;
    A(1, 0) = 0.5 * (gm1 - 3.0) * u2; A(1, 1) = (3.0 - gamma) * u;   A(1, 2) = gm1;
    A(2, 0) = u * (0.5 * gm1 * u2 - H); A(2, 1) = H - gm1 * u2;      A(2, 2) = gamma * u;
    return A;
}

Matrix3 computeSourceJacobian(const Vector3& Q) {
    // Friction term: dS(1)/dQ(1) = -0.02 * 2*|u| * section (approx)
    Matrix3 dS = Matrix3::Zero();
    double u = Q(1) / Q(0);
    dS(1, 1) = -0.02 * 2.0 * std::abs(u) * section;
    return dS;
}

// =========== CFL-ADAPTIVE TIMESTEP
// FIX 3: dt is recomputed at every timestep based on the CFL condition.
// This avoids both under-resolving fast waves and wasting time with tiny steps.
double compute_dt(const VectorGlobal& Q_n) {
    double max_speed = 0.0;
    for (int i = 0; i < N; ++i) {
        Vector3 Qc = Q_n.segment<3>(3 * i);
        double speed = std::abs(Qc(1) / Qc(0)) + get_sound_speed(Qc);
        max_speed = std::max(max_speed, speed);
    }
    if (max_speed < 1e-12) return 1e-4; // Fallback for zero-velocity init
    return CFL * dx / max_speed;
}

// =========== SOLVER CORE

int main() {

    VectorGlobal Q_n(3 * N);
    VectorGlobal Q_new(3 * N);

    double p_initial = 10000.0;
    double T_initial = 300.0;
    double u_initial = 1.0;
    double rho_initial = p_initial / (R_vapor * T_initial);

    for (int i = 0; i < N; ++i) {
        double E = p_initial / ((gamma - 1.0) * rho_initial) + 0.5 * u_initial * u_initial;
        Q_n(3 * i + 0) = rho_initial * section;
        Q_n(3 * i + 1) = rho_initial * u_initial * section;
        Q_n(3 * i + 2) = rho_initial * E * section;
    }
    Q_new = Q_n;

    double t_final = 1.0;
    double time = 0.0;

    std::cout << "Starting Newton-Krylov FVM Solver..." << std::endl;
    std::cout << "Grid: " << N << " cells. System Size: " << 3 * N << std::endl;

    std::ofstream file("history.csv");
    file << "time,x,rho,u,p,T,energy\n";

    int step_counter = 0;
    const int save_every = 100; // Save output every N timesteps to reduce I/O

    // =========== TIME STEPPING LOOP
    while (time < t_final) {

        // FIX 3: Adaptive dt from CFL condition
        double dt = compute_dt(Q_n);
        if (time + dt > t_final) dt = t_final - time; // Don't overshoot t_final

        // FIX 1 (from earlier): Reset Q_new to Q_n at start of each timestep
        Q_new = Q_n;

        // FIX 2: Declare SparseLU solver ONCE per timestep, outside Newton loop.
        // Reuse factorization every `refactor_every` Newton iterations.
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        bool factorized = false;

        // =========== NEWTON-RAPHSON LOOP
        for (int iter = 0; iter < max_newton_iters; ++iter) {

            std::vector<Eigen::Triplet<double>> tripletList;
            tripletList.reserve(N * 3 * 3 * 3);

            VectorGlobal Residual(3 * N);
            Residual.setZero();

            for (int i = 0; i < N; ++i) {

                Vector3 Uc = Q_new.segment<3>(3 * i);
                double rho_in = Uc(0) / section;
                double u_in = Uc(1) / Uc(0);
                double p_in = get_pA(Uc) / section;

                // Left ghost cell
                Vector3 Ul;
                if (i > 0) {
                    Ul = Q_new.segment<3>(3 * (i - 1));
                }
                else {
                    double u_b = 1.0;
                    double p_b = p_in;
                    double T_b = 350.0;
                    double rho_b = p_b / (R_vapor * T_b);
                    double E_b = p_b / ((gamma - 1.0) * rho_b) + 0.5 * u_b * u_b;
                    Ul(0) = rho_b * section;
                    Ul(1) = rho_b * u_b * section;
                    Ul(2) = rho_b * E_b * section;
                }

                // Right ghost cell
                Vector3 Ur;
                if (i < N - 1) {
                    Ur = Q_new.segment<3>(3 * (i + 1));
                }
                else {
                    double u_b = u_in;
                    double p_b = 10000.0;
                    double T_b = 300.0;
                    double rho_b = p_b / (R_vapor * T_b);
                    double E_b = p_b / ((gamma - 1.0) * rho_b) + 0.5 * u_b * u_b;
                    Ur(0) = rho_b * section;
                    Ur(1) = rho_b * u_b * section;
                    Ur(2) = rho_b * E_b * section;
                }

                // Residual terms
                Vector3 time_term = (Uc - Q_n.segment<3>(3 * i)) * (dx / dt);

                Vector3 Fc = computeFlux(Uc);
                Vector3 Fl = computeFlux(Ul);
                Vector3 Fr = computeFlux(Ur);

                double spectral_c = std::abs(Uc(1) / Uc(0)) + get_sound_speed(Uc);
                double spectral_l = std::abs(Ul(1) / Ul(0)) + get_sound_speed(Ul);
                double spectral_r = std::abs(Ur(1) / Ur(0)) + get_sound_speed(Ur);

                double eps = 0.5;
                double nu_l = eps * std::max(spectral_c, spectral_l);
                double nu_r = eps * std::max(spectral_c, spectral_r);

                Vector3 Flux_Right = 0.5 * (Fc + Fr) - 0.5 * nu_r * (Ur - Uc);
                Vector3 Flux_Left = 0.5 * (Fl + Fc) - 0.5 * nu_l * (Uc - Ul);
                Vector3 flux_diff = Flux_Right - Flux_Left;

                Vector3 Source = computeSource(Uc) * dx;
                Vector3 R_cell = time_term + flux_diff - Source;
                Residual.segment<3>(3 * i) = R_cell;

                // Jacobian blocks
                Matrix3 dSource = computeSourceJacobian(Uc);
                double nu_total = 0.5 * nu_r + 0.5 * nu_l;

                Matrix3 J_diag = Matrix3::Identity() * (dx / dt);
                J_diag += nu_total * Matrix3::Identity();
                J_diag -= dSource * dx;

                Matrix3 A_r = computeFluxJacobian(Ur);
                Matrix3 J_right = 0.5 * A_r - 0.5 * nu_r * Matrix3::Identity();

                Matrix3 A_l = computeFluxJacobian(Ul);
                Matrix3 J_left = -0.5 * A_l - 0.5 * nu_l * Matrix3::Identity();

                for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++)
                    tripletList.push_back({ 3 * i + r, 3 * i + c, J_diag(r,c) });

                if (i < N - 1)
                    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++)
                        tripletList.push_back({ 3 * i + r, 3 * (i + 1) + c, J_right(r,c) });

                if (i > 0)
                    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++)
                        tripletList.push_back({ 3 * i + r, 3 * (i - 1) + c, J_left(r,c) });
            }

            double res_norm = Residual.norm();
            if (res_norm < newton_tol) {
                // std::cout << "  Newton converged at iter " << iter
                //     << " | res=" << res_norm << std::endl;
                break;
            }

            // FIX 2: Refactorize only every `refactor_every` iterations
            // On iter 0 always factorize. Then reuse the factorization
            // for the next (refactor_every-1) iters — valid near convergence
            // where the Jacobian changes slowly.
            if (iter % refactor_every == 0 || !factorized) {
                Eigen::SparseMatrix<double> J_global(3 * N, 3 * N);
                J_global.setFromTriplets(tripletList.begin(), tripletList.end());
                J_global.makeCompressed();
                solver.compute(J_global);

                if (solver.info() != Eigen::Success) {
                    std::cerr << "SparseLU decomposition failed at t=" << time
                        << " iter=" << iter << std::endl;
                    return -1;
                }
                factorized = true;
            }

            VectorGlobal deltaQ = solver.solve(-Residual);

            if (solver.info() != Eigen::Success) {
                std::cerr << "SparseLU solve failed!" << std::endl;
                return -1;
            }

            Q_new += deltaQ;
        }

        // Save output every `save_every` steps to reduce I/O overhead
        if (step_counter % save_every == 0) {
            std::cout << "Time: " << time << " | dt=" << dt << std::endl;
            for (int i = 0; i < N; ++i) {
                Vector3 Q = Q_new.segment<3>(3 * i);
                double rho = Q(0) / section;
                double u = (Q(0) > 1e-8) ? Q(1) / Q(0) : 0.0;
                double p = get_pA(Q) / section;
                double T = (rho > 1e-8) ? p / (rho * R_vapor) : 0.0;
                double energy = (Q(0) > 1e-8) ? Q(2) / Q(0) : 0.0;
                double x = (i + 0.5) * dx;
                file << time << "," << x << "," << rho << "," << u << ","
                    << p << "," << T << "," << energy << "\n";
            }
            file.flush();
        }

        Q_n = Q_new;
        time += dt;
        step_counter++;
    }

    file.close();
    std::cout << "Simulation Complete. Steps: " << step_counter
        << ". Data saved in history.csv" << std::endl;
    return 0;
}