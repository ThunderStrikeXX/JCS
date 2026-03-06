#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <ctime>

// =========== PARAMETERS ===========
const int    N = 100;               // Number of spatial cells [-]
const double L = 1.0;               // Domain length [m]
const double dx = L / N;            // Cell width [m]
const double GAMMA = 1.57;          // Gamma [-]
const double GRAVITY_X = 0.0;       // Axial gravity [m/s²]
const double AREA = 1.0;            // Constant cross-sectional area [m²]
const double R_GAS = 361.5;         // Specific gas constant for sodium vapor [J/kg·K]
const double CONDUCTIVITY = 1.0;    // Thermal conductivity k [W/m·K]
const double VISCOSITY = 1e-5;      // Dynamic viscosity mu [Pa·s]

// Newton-Raphson settings
const int    MAX_NEWTON_ITERS = 1000;
const double NEWTON_TOL = 1e-6;
const int    REFACTOR_EVERY = 1;
const int    SAVE_EVERY = 10;

using Vector3 = Eigen::Vector3d;
using Matrix3 = Eigen::Matrix3d;
using VectorGlobal = Eigen::VectorXd;

// =========== EQUATION OF STATE ===========

double get_pA(const Vector3& Q) {
    if (Q(0) < 1e-8) return 0.0;
    return (GAMMA - 1.0) * (Q(2) - 0.5 * Q(1) * Q(1) / Q(0));
}

double get_sound_speed(const Vector3& Q) {
    double pA = get_pA(Q), rhoA = Q(0);
    if (rhoA < 1e-8 || pA < 0.0) return 0.0;
    return std::sqrt(GAMMA * pA / rhoA);
}

double get_T(const Vector3& Q) {
    if (Q(0) < 1e-8) return 0.0;
    return get_pA(Q) / (Q(0) * R_GAS);
}

// Returns u from conserved variables
inline double get_u(const Vector3& Q) { return Q(1) / Q(0); }

// =========== FLUX AND SOURCE ===========

Vector3 computeFlux(const Vector3& Q) {
    double pA = get_pA(Q), u = get_u(Q);
    return { Q(1), Q(1) * u + pA, u * (Q(2) + pA) };
}

Vector3 computeSource(const Vector3& Q) {
    double rhoA = Q(0), u = get_u(Q);
    return { 0.0, rhoA * GRAVITY_X, rhoA * u * GRAVITY_X };
}

// =========== THERMAL CONDUCTION ===========
//
// Adds  d/dx(k * dT/dx) * A  to energy equation:
//   R_cond = -k*A/dx² * (T_l - 2*T_c + T_r)

Eigen::RowVector3d dTdQ(const Vector3& Q) {
    double rhoA = Q(0), u = get_u(Q);
    double e_int = Q(2) / rhoA - 0.5 * u * u;
    double c = (GAMMA - 1.0) / R_GAS;
    return { -c * (e_int + 0.5 * u * u) / rhoA, -c * u / rhoA, c / rhoA };
}

double conductionResidual(const Vector3& Ql, const Vector3& Qc, const Vector3& Qr) {
    return -CONDUCTIVITY * AREA / (dx * dx) * (get_T(Ql) - 2.0 * get_T(Qc) + get_T(Qr));
}

Eigen::RowVector3d dRcond_dQl(const Vector3& Ql) {
    return -CONDUCTIVITY * AREA / (dx * dx) * dTdQ(Ql);
}
Eigen::RowVector3d dRcond_dQc(const Vector3& Qc) {
    return  CONDUCTIVITY * AREA / (dx * dx) * 2.0 * dTdQ(Qc);
}
Eigen::RowVector3d dRcond_dQr(const Vector3& Qr) {
    return -CONDUCTIVITY * AREA / (dx * dx) * dTdQ(Qr);
}

Eigen::RowVector3d dudQ(const Vector3& Q) {
    double rhoA = Q(0), u = get_u(Q);
    return { -u / rhoA, 1.0 / rhoA, 0.0 };
}

// Momentum viscous residual (scalar)
double viscMomResidual(const Vector3& Ql, const Vector3& Qc, const Vector3& Qr) {
    return -(4.0 / 3.0) * VISCOSITY * AREA / (dx * dx) * (get_u(Ql) - 2.0 * get_u(Qc) + get_u(Qr));
}

// Energy viscous residual (scalar) — viscous work: d/dx(mu*u*du/dx)*A
double viscEnResidual(const Vector3& Ql, const Vector3& Qc, const Vector3& Qr) {
    double uc = get_u(Qc);
    double ul = get_u(Ql);
    double ur = get_u(Qr);

    double u_right = 0.5 * (uc + ur);
    double u_left = 0.5 * (ul + uc);
    return -(4.0 / 3.0) * VISCOSITY * AREA / (dx * dx) * (u_right * (ur - uc) - u_left * (uc - ul));
}

// Jacobian rows for viscous momentum term
Eigen::RowVector3d dRviscMom_dQl(const Vector3& Ql) {
    return -(4.0 / 3.0) * VISCOSITY * AREA / (dx * dx) * dudQ(Ql);
}
Eigen::RowVector3d dRviscMom_dQc(const Vector3& Qc) {
    return  (4.0 / 3.0) * VISCOSITY * AREA / (dx * dx) * 2.0 * dudQ(Qc);
}
Eigen::RowVector3d dRviscMom_dQr(const Vector3& Qr) {
    return -(4.0 / 3.0) * VISCOSITY * AREA / (dx * dx) * dudQ(Qr);
}

// Jacobian rows for viscous energy term (linearized around current u_c)
Eigen::RowVector3d dRviscEn_dQl(const Vector3& Ql, const Vector3& Qc) {
    double uc = get_u(Qc), ul = get_u(Ql);
    double u_left = 0.5 * (ul + uc);

    return -VISCOSITY * AREA / (dx * dx) * u_left * dudQ(Ql);
}
Eigen::RowVector3d dRviscEn_dQc(const Vector3& Ql, const Vector3& Qc, const Vector3& Qr) {
    double uc = get_u(Qc), ul = get_u(Ql), ur = get_u(Qr);
    double u_right = 0.5 * (uc + ur);
    double u_left = 0.5 * (ul + uc);

    double coeff = (ur - uc) * 0.5 - u_right - (uc - ul) * 0.5 - u_left;

    return -VISCOSITY * AREA / (dx * dx) * (-2.0 * uc) * dudQ(Qc);
}
Eigen::RowVector3d dRviscEn_dQr(const Vector3& Qc, const Vector3& Qr) {
    double uc = get_u(Qc), ur = get_u(Qr);
    double u_right = 0.5 * (uc + ur);
    return -VISCOSITY * AREA / (dx * dx) * u_right * dudQ(Qr);
}

// =========== FLUX JACOBIAN ===========

Matrix3 computeFluxJacobian(const Vector3& Q) {
    double u = get_u(Q), u2 = u * u;
    double pA = get_pA(Q), H = (Q(2) + pA) / Q(0), gm1 = GAMMA - 1.0;
    Matrix3 A;
    A(0, 0) = 0.0;                  A(0, 1) = 1.0;            A(0, 2) = 0.0;
    A(1, 0) = 0.5 * (gm1 - 3.0) * u2;    A(1, 1) = (3.0 - GAMMA) * u;  A(1, 2) = gm1;
    A(2, 0) = u * (0.5 * gm1 * u2 - H);    A(2, 1) = H - gm1 * u2;       A(2, 2) = GAMMA * u;
    return A;
}

Matrix3 computeSourceJacobian(const Vector3&) {
    return Matrix3::Zero();   // no friction, gravity is linear → zero Jacobian
}

// =========== BOUNDARY CONDITIONS ===========

// Left: u=0 (wall), T=350 K, p=Neumann
Vector3 leftGhostCell(double p_in) {
    double u_b = 1.0, T_b = 350.0, rho_b = p_in / (R_GAS * T_b);
    double E_b = p_in / ((GAMMA - 1.0) * rho_b) + 0.5 * u_b * u_b;
    return { rho_b * AREA, rho_b * u_b * AREA, rho_b * E_b * AREA };
}

// Right: p=10000 Pa, T=300 K, u=Neumann
Vector3 rightGhostCell(double u_in, double T_in) {
    double p_b = 10000.0, rho_b = p_b / (R_GAS * T_in);
    double E_b = p_b / ((GAMMA - 1.0) * rho_b) + 0.5 * u_in * u_in;
    return { rho_b * AREA, rho_b * u_in * AREA, rho_b * E_b * AREA };
}

// =========== MAIN ===========

int main() {

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

    double t_final = 1.0, time = 0.0, dt = 1e-3;
    int step = 0;

    std::cout << "FVM Solver (Euler + conduction + viscosity) | N=" << N
        << " | DOFs=" << 3 * N << "\n";

    std::ofstream f_rho("rho.txt"), f_u("u.txt"), f_p("p.txt"),
        f_T("T.txt"), f_energy("energy.txt");

    auto wall_start = std::chrono::steady_clock::now();     // Wall time
    std::clock_t cpu_start = std::clock();                  // CPU time

    // =========== TIME LOOP ===========
    while (time < t_final) {

        Q_new = Q_n;

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
                double u_in = get_u(Uc);
                double p_in = get_pA(Uc) / AREA;
                double T_in = get_T(Uc);

                Vector3 Ul = (i > 0) ? Q_new.segment<3>(3 * (i - 1)) : leftGhostCell(p_in);
                Vector3 Ur = (i < N - 1) ? Q_new.segment<3>(3 * (i + 1)) : rightGhostCell(u_in, T_in);

                // --- Convective fluxes (Rusanov) ---
                Vector3 Fc = computeFlux(Uc), Fl = computeFlux(Ul), Fr = computeFlux(Ur);

                double eps = 1e-8;

                double sp_c = std::abs(u_in) + get_sound_speed(Uc);
                double sp_l = std::abs(get_u(Ul)) + get_sound_speed(Ul);
                double sp_r = std::abs(get_u(Ur)) + get_sound_speed(Ur);
                double nu_l = eps * std::max(sp_c, sp_l);
                double nu_r = eps * std::max(sp_c, sp_r);

                Vector3 F_right = 0.5 * (Fc + Fr) - 0.5 * nu_r * (Ur - Uc);
                Vector3 F_left = 0.5 * (Fl + Fc) - 0.5 * nu_l * (Uc - Ul);

                // --- Cell residual ---
                Vector3 R_cell = (Uc - Q_n.segment<3>(3 * i)) * (dx / dt)
                    + (F_right - F_left)
                    - computeSource(Uc) * dx;

                // --- Conduction (energy row) ---
                R_cell(2) += conductionResidual(Ul, Uc, Ur) * dx;

                // --- Viscosity (momentum row + energy row) ---
                R_cell(1) += viscMomResidual(Ul, Uc, Ur) * dx;
                R_cell(2) += viscEnResidual(Ul, Uc, Ur) * dx;

                Residual.segment<3>(3 * i) = R_cell;

                // --- Jacobian blocks ---
                Matrix3 Jd = Matrix3::Identity() * (dx / dt)
                    + 0.5 * (nu_l + nu_r) * Matrix3::Identity()
                    - computeSourceJacobian(Uc) * dx;

                Matrix3 Jr = 0.5 * computeFluxJacobian(Ur) - 0.5 * nu_r * Matrix3::Identity();
                Matrix3 Jl = -0.5 * computeFluxJacobian(Ul) + 0.5 * nu_l * Matrix3::Identity();

                // Conduction → energy row (row 2)
                Jd.row(2) += dx * dRcond_dQc(Uc);
                Jr.row(2) += dx * dRcond_dQr(Ur);
                Jl.row(2) += dx * dRcond_dQl(Ul);

                // Viscosity momentum → row 1
                Jd.row(1) += dx * dRviscMom_dQc(Uc);
                Jr.row(1) += dx * dRviscMom_dQr(Ur);
                Jl.row(1) += dx * dRviscMom_dQl(Ul);

                // Viscosity energy → row 2
                Jd.row(2) += dx * dRviscEn_dQc(Ul, Uc, Ur);
                Jr.row(2) += dx * dRviscEn_dQr(Uc, Ur);
                Jl.row(2) += dx * dRviscEn_dQl(Ul, Uc);

                // --- Assemble ---
                for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) {
                    triplets.push_back({ 3 * i + r, 3 * i + c,     Jd(r,c) });
                    if (i < N - 1) triplets.push_back({ 3 * i + r, 3 * (i + 1) + c, Jr(r,c) });
                    if (i > 0)   triplets.push_back({ 3 * i + r, 3 * (i - 1) + c, Jl(r,c) });
                }
            }

            double res_norm = Residual.norm();
            // std::cout << "  iter=" << iter << " |R|=" << res_norm << "\n";

            if (res_norm < NEWTON_TOL) break;

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
            for (int i = 0; i < N; ++i) {
                Vector3 Q = Q_new.segment<3>(3 * i);
                double rho = Q(0) / AREA;
                double u = (Q(0) > 1e-8) ? get_u(Q) : 0.0;
                double p = get_pA(Q) / AREA;
                double T = (rho > 1e-8) ? p / (rho * R_GAS) : 0.0;
                double e = (Q(0) > 1e-8) ? Q(2) / Q(0) : 0.0;
                f_rho << rho << ", "; f_u << u << ", "; f_p << p << ", ";
                f_T << T << ", ";     f_energy << e << ", ";
            }
            f_rho << "\n"; f_u << "\n"; f_p << "\n"; f_T << "\n"; f_energy << "\n";
            f_rho.flush(); f_u.flush(); f_p.flush(); f_T.flush(); f_energy.flush();
        }

        Q_n = Q_new;
        time += dt;
        step++;
    }

    f_rho.close(); f_u.close(); f_p.close(); f_T.close(); f_energy.close();

    std::clock_t cpu_end = std::clock();
    std::cout << "CPU time: " << (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC << " s\n";

    auto wall_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> wall_elapsed = wall_end - wall_start;
    std::cout << "Wall clock time: " << wall_elapsed.count() << " s\n";

    system("pause");

    return 0;
}