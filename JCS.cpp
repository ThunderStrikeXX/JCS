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
const double GAMMA = 1.57;      // Gamma [-]
const double GRAVITY_X = 0.0;   // Axial gravity [m/s²]
const double AREA = 1.0;       // Constant cross-sectional area [m²]
const double R_GAS = 361.5;     // Specific gas constant for sodium vapor [J/kg·K]
const double CONDUCTIVITY = 0.01; // Thermal conductivity k [W/m·K]

// Friction model
const double FRICTION_COEFF = 0.0;

// Newton-Raphson settings
const int    MAX_NEWTON_ITERS = 5;
const double NEWTON_TOL = 1e-2;
const int    REFACTOR_EVERY = 3;
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
    double pA = get_pA(Q);
    double rhoA = Q(0);
    if (rhoA < 1e-8 || pA < 0.0) return 0.0;
    return std::sqrt(GAMMA * pA / rhoA);
}

double get_T(const Vector3& Q) {
    if (Q(0) < 1e-8) return 0.0;
    double pA = get_pA(Q);
    double rho = Q(0) / AREA;
    return pA / (AREA * rho * R_GAS);
}

// =========== FLUX AND SOURCE ===========

Vector3 computeFlux(const Vector3& Q) {
    double pA = get_pA(Q);
    double u = Q(1) / Q(0);
    return { Q(1), Q(1) * u + pA, u * (Q(2) + pA) };
}

Vector3 computeSource(const Vector3& Q) {
    double rhoA = Q(0);
    double u = Q(1) / Q(0);
    double friction = FRICTION_COEFF * u * std::abs(u) * rhoA;
    Vector3 S;
    S(0) = 0.0;
    S(1) = -friction * AREA + rhoA * GRAVITY_X;
    S(2) = rhoA * u * GRAVITY_X;
    return S;
}

Eigen::RowVector3d dTdQ(const Vector3& Q) {
    double rhoA = Q(0);
    double u = Q(1) / Q(0);
    double e_int = Q(2) / rhoA - 0.5 * u * u;
    double gm1_R = (GAMMA - 1.0) / R_GAS;
    Eigen::RowVector3d dT;
    dT(0) = -gm1_R * (e_int + 0.5 * u * u) / rhoA;
    dT(1) = -gm1_R * u / rhoA;
    dT(2) = gm1_R / rhoA;
    return dT;
}

// Residual contribution (scalar, added to energy row)
double conductionResidual(const Vector3& Ql, const Vector3& Qc, const Vector3& Qr) {
    return -CONDUCTIVITY * AREA / (dx * dx) * (get_T(Ql) - 2.0 * get_T(Qc) + get_T(Qr));
}

// Jacobian rows (energy row only):  dR_cond / d{Ql, Qc, Qr}
Eigen::RowVector3d dRcond_dQl(const Vector3& Ql) {
    return -CONDUCTIVITY * AREA / (dx * dx) * dTdQ(Ql);
}
Eigen::RowVector3d dRcond_dQc(const Vector3& Qc) {
    return  CONDUCTIVITY * AREA / (dx * dx) * 2.0 * dTdQ(Qc);
}
Eigen::RowVector3d dRcond_dQr(const Vector3& Qr) {
    return -CONDUCTIVITY * AREA / (dx * dx) * dTdQ(Qr);
}

// =========== JACOBIANS ===========

Matrix3 computeFluxJacobian(const Vector3& Q) {
    double u = Q(1) / Q(0);
    double u2 = u * u;
    double pA = get_pA(Q);
    double H = (Q(2) + pA) / Q(0);
    double gm1 = GAMMA - 1.0;
    Matrix3 A;
    A(0, 0) = 0.0;                      A(0, 1) = 1.0;             A(0, 2) = 0.0;
    A(1, 0) = 0.5 * (gm1 - 3.0) * u2;        A(1, 1) = (3.0 - GAMMA) * u;  A(1, 2) = gm1;
    A(2, 0) = u * (0.5 * gm1 * u2 - H);      A(2, 1) = H - gm1 * u2;     A(2, 2) = GAMMA * u;
    return A;
}

Matrix3 computeSourceJacobian(const Vector3& Q) {
    double u = Q(1) / Q(0);
    Matrix3 dS = Matrix3::Zero();
    dS(1, 1) = -FRICTION_COEFF * 2.0 * std::abs(u) * AREA;
    return dS;
}

// =========== TIMESTEP ===========

double compute_dt(const VectorGlobal& Q) {
    double max_speed = 0.0;
    for (int i = 0; i < N; ++i) {
        Vector3 Qc = Q.segment<3>(3 * i);
        double speed = std::abs(Qc(1) / Qc(0)) + get_sound_speed(Qc);
        max_speed = std::max(max_speed, speed);
    }
    if (max_speed < 1e-12) return 1e-4;
    return 1e-3;
}

// =========== BOUNDARY CONDITIONS ===========

// Left: u=0 (wall), T=350 K, p=Neumann
Vector3 leftGhostCell(double p_in) {
    double u_b = 0.0, T_b = 350.0;
    double rho_b = p_in / (R_GAS * T_b);
    double E_b = p_in / ((GAMMA - 1.0) * rho_b) + 0.5 * u_b * u_b;
    return { rho_b * AREA, rho_b * u_b * AREA, rho_b * E_b * AREA };
}

// Right: p=10000 Pa, T=300 K, u=Neumann
Vector3 rightGhostCell(double u_in) {
    double p_b = 10000.0, T_b = 300.0;
    double rho_b = p_b / (R_GAS * T_b);
    double E_b = p_b / ((GAMMA - 1.0) * rho_b) + 0.5 * u_in * u_in;
    return { rho_b * AREA, rho_b * u_in * AREA, rho_b * E_b * AREA };
}

// =========== MAIN ===========

int main() {

    VectorGlobal Q_n(3 * N), Q_new(3 * N);
    {
        double p0 = 10000.0, T0 = 300.0, u0 = 0.0;
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
    int step = 0;

    std::cout << "FVM Solver (Euler + thermal conduction) | N=" << N
        << " | DOFs=" << 3 * N << "\n";

    std::ofstream f_rho("rho.txt"), f_u("u.txt"), f_p("p.txt"),
        f_T("T.txt"), f_energy("energy.txt");

    // =========== TIME LOOP ===========
    while (time < t_final) {

        double dt = compute_dt(Q_n);
        if (time + dt > t_final) dt = t_final - time;

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
                double u_in = Uc(1) / Uc(0);
                double p_in = get_pA(Uc) / AREA;

                Vector3 Ul = (i > 0) ? Q_new.segment<3>(3 * (i - 1)) : leftGhostCell(p_in);
                Vector3 Ur = (i < N - 1) ? Q_new.segment<3>(3 * (i + 1)) : rightGhostCell(u_in);

                // --- Convective fluxes (JST) ---
                Vector3 Fc = computeFlux(Uc);
                Vector3 Fl = computeFlux(Ul);
                Vector3 Fr = computeFlux(Ur);

                double sp_c = std::abs(u_in) + get_sound_speed(Uc);
                double sp_l = std::abs(Ul(1) / Ul(0)) + get_sound_speed(Ul);
                double sp_r = std::abs(Ur(1) / Ur(0)) + get_sound_speed(Ur);
                double eps = 0.5;
                double nu_l = eps * std::max(sp_c, sp_l);
                double nu_r = eps * std::max(sp_c, sp_r);

                Vector3 F_right = 0.5 * (Fc + Fr) - 0.5 * nu_r * (Ur - Uc);
                Vector3 F_left = 0.5 * (Fl + Fc) - 0.5 * nu_l * (Uc - Ul);

                // --- Cell residual (convective + source) ---
                Vector3 R_cell = (Uc - Q_n.segment<3>(3 * i)) * (dx / dt)
                    + (F_right - F_left)
                    - computeSource(Uc) * dx;

                // --- Add thermal conduction to energy equation (row 2) ---
                R_cell(2) += conductionResidual(Ul, Uc, Ur) * dx;

                Residual.segment<3>(3 * i) = R_cell;

                // --- Jacobian blocks ---
                Matrix3 J_diag = Matrix3::Identity() * (dx / dt)
                    + (nu_l + nu_r) * Matrix3::Identity()
                    + 0.5 * computeFluxJacobian(Uc)
                    - computeSourceJacobian(Uc) * dx;

                Matrix3 J_right = 0.5 * computeFluxJacobian(Ur) - 0.5 * nu_r * Matrix3::Identity();
                Matrix3 J_left = -0.5 * computeFluxJacobian(Ul) + 0.5 * nu_l * Matrix3::Identity();

                // --- Add conduction Jacobian to energy row (row 2) ---
                J_diag.row(2) += dx * dRcond_dQc(Uc);
                J_right.row(2) += dx * dRcond_dQr(Ur);
                J_left.row(2) += dx * dRcond_dQl(Ul);

                // --- Assemble ---
                for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) {
                    triplets.push_back({ 3 * i + r, 3 * i + c,     J_diag(r,c) });
                    if (i < N - 1) triplets.push_back({ 3 * i + r, 3 * (i + 1) + c, J_right(r,c) });
                    if (i > 0)   triplets.push_back({ 3 * i + r, 3 * (i - 1) + c, J_left(r,c) });
                }
            }

            double res_norm = Residual.norm();
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

            std::cout << "t=" << time << " dt=" << dt << "\n";
            for (int i = 0; i < N; ++i) {
                Vector3 Q = Q_new.segment<3>(3 * i);
                double rho = Q(0) / AREA;
                double u = (Q(0) > 1e-8) ? Q(1) / Q(0) : 0.0;
                double p = get_pA(Q) / AREA;
                double T = (rho > 1e-8) ? p / (rho * R_GAS) : 0.0;
                double e = (Q(0) > 1e-8) ? Q(2) / Q(0) : 0.0;

                f_rho << rho << ", ";
                f_u << u << ", ";
                f_p << p << ", ";
                f_T << T << ", ";
                f_energy << e << ", ";
            }
            f_rho << "\n"; f_u << "\n"; f_p << "\n"; f_T << "\n"; f_energy << "\n";
            f_rho.flush(); f_u.flush(); f_p.flush(); f_T.flush(); f_energy.flush();
        }

        Q_n = Q_new;
        time += dt;
        step++;
    }

    f_rho.close(); f_u.close(); f_p.close(); f_T.close();   f_energy.close();

    return 0;
}
