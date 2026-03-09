#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iomanip> 

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

const int    MAX_NEWTON_ITERS = 1000;
const double NEWTON_TOL = 1e-6;
const int    REFACTOR_EVERY = 1;
const int    SAVE_EVERY = 10;

using Vector3 = Eigen::Vector3d;
using Matrix3 = Eigen::Matrix3d;
using VectorGlobal = Eigen::VectorXd;

// =========== FUNCTIONS ===========
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

inline double get_u(const Vector3& Q) { return Q(1) / Q(0); }

Vector3 computeFlux(const Vector3& Q) {
    double pA = get_pA(Q), u = get_u(Q);
    return { Q(1), Q(1) * u + pA, u * (Q(2) + pA) };
}

Vector3 computeSource(const Vector3& Q) {
    double rhoA = Q(0), u = get_u(Q);
    return { 0.0, rhoA * GRAVITY_X, rhoA * u * GRAVITY_X };
}

Eigen::RowVector3d dTdQ(const Vector3& Q) {

    double c = (GAMMA - 1.0) / R_GAS;
    double rhoA = Q(0);
    double E = Q(2) / Q(0);
    double u = Q(1) / Q(0);
    return { c * (-E + u * u) / rhoA, -c * u / rhoA, c / rhoA};
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

    double rhoA = Q(0);
    double u = get_u(Q);
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

    double coeff = 0.5 * (ur - uc) - u_right
        - 0.5 * (uc - ul) - u_left;

    return -(4.0 / 3.0) * VISCOSITY * AREA / (dx * dx) * coeff * dudQ(Qc);
}
Eigen::RowVector3d dRviscEn_dQr(const Vector3& Qc, const Vector3& Qr) {
    double uc = get_u(Qc), ur = get_u(Qr);
    double u_right = 0.5 * (uc + ur);
    return -VISCOSITY * AREA / (dx * dx) * u_right * dudQ(Qr);
}

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
Vector3 leftFaceState(const Vector3& Uc) {
    double u_b = 1.0;                       // Dirichlet
    double T_b = 350.0;                     // Dirichlet
    double p_b = get_pA(Uc) / AREA;         // Neumann
    double rho_b = p_b / (R_GAS * T_b);     // Derived

    double E_b = p_b / ((GAMMA - 1.0) * rho_b) + 0.5 * u_b * u_b;
    return { rho_b * AREA, rho_b * u_b * AREA, rho_b * E_b * AREA };
}

Vector3 rightFaceState(const Vector3& Uc) {
    double p_b = 10000.0;                   // Dirichlet
    double T_b = get_T(Uc);                 // Neumann
    double u_b = get_u(Uc);                 // Neumann
    double rho_b = p_b / (R_GAS * T_b);     // Derived
    double E_b = p_b / ((GAMMA - 1.0) * rho_b) + 0.5 * u_b * u_b;
    return { rho_b * AREA, rho_b * u_b * AREA, rho_b * E_b * AREA };
}

// =========== MUSCL RECONSTRUCTION ===========
Vector3 minmod_limit(const Vector3& a, const Vector3& b) {
    Vector3 result;
    for (int k = 0; k < 3; ++k) {
        if (a(k) * b(k) <= 0.0)
            result(k) = 0.0;
        else if (std::abs(a(k)) < std::abs(b(k)))
            result(k) = a(k);
        else
            result(k) = b(k);
    }
    return result; // = φ·(Ul - Ul2)
}

Vector3 vanAlbada_limit(const Vector3& a, const Vector3& b) {
    Vector3 result;
    for (int k = 0; k < 3; ++k) {
        if (a(k) * b(k) <= 0.0)
            result(k) = 0.0;
        else
            result(k) = (a(k) * b(k) * (a(k) + b(k))) / (a(k) * a(k) + b(k) * b(k));
    }
    return result;
}

// Ricostruisce stato sinistro e destro alla faccia i+1/2
// Ul2 = U_{i-1}, Ul = U_i, Ur = U_{i+1}, Ur2 = U_{i+2}
void muscl_reconstruct(const Vector3& Ul2, const Vector3& Ul,
    const Vector3& Ur, const Vector3& Ur2,
    Vector3& Ql_face, Vector3& Qr_face) {
    Vector3 dL = minmod_limit(Ul - Ul2, Ur - Ul);
    Vector3 dR = minmod_limit(Ur - Ul, Ur2 - Ur);
    Ql_face = Ul + 0.5 * dL;
    Qr_face = Ur - 0.5 * dR;
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

    double eps = 1;

    // Residual for the upwind fluxes formulation
    auto cell_residual_upwind = [&](const VectorGlobal& Qvec, int cell) -> Vector3 {

        Vector3 Uc = Qvec.segment<3>(3 * cell);
        Vector3 Ul = (cell > 0) ? Qvec.segment<3>(3 * (cell - 1)) : leftFaceState(Uc);
        Vector3 Ur = (cell < N - 1) ? Qvec.segment<3>(3 * (cell + 1)) : rightFaceState(Uc);

        // Upwind discretizazion
        Vector3 F_left = computeFlux(Ul);
        Vector3 F_right = (cell < N - 1) ? computeFlux(Uc) : computeFlux(Ur);

        Vector3 R = (Uc - Q_n.segment<3>(3 * cell)) * (dx / dt)
            + (F_right - F_left)
            - computeSource(Uc) * dx;

        return R;
    };

    // Residual for the linear fluxes formulation
    auto cell_residual_linear = [&](const VectorGlobal& Qvec, int cell) -> Vector3 {

        Vector3 Uc = Qvec.segment<3>(3 * cell);
        Vector3 Ul = (cell > 0) ? Qvec.segment<3>(3 * (cell - 1)) : leftFaceState(Uc);
        Vector3 Ur = (cell < N - 1) ? Qvec.segment<3>(3 * (cell + 1)) : rightFaceState(Uc);

        // Upwind discretizazion
        Vector3 F_left = (cell > 0) ? 0.5 * (computeFlux(Ul) + computeFlux(Uc)) : computeFlux(Ul);
        Vector3 F_right = (cell < N - 1) ? 0.5 * (computeFlux(Uc) + computeFlux(Ur)) : computeFlux(Ur);

        Vector3 R = (Uc - Q_n.segment<3>(3 * cell)) * (dx / dt)
            + (F_right - F_left)
            - computeSource(Uc) * dx;

        return R;
    };

    // Residual for the linear fluxes formulation with Rusanov correction
    auto cell_residual_rusanov = [&](const VectorGlobal& Qvec, int cell) -> Vector3 {
        Vector3 Uc = Qvec.segment<3>(3 * cell);
        Vector3 Ul = (cell > 0) ? Qvec.segment<3>(3 * (cell - 1)) : leftFaceState(Uc);
        Vector3 Ur = (cell < N - 1) ? Qvec.segment<3>(3 * (cell + 1)) : rightFaceState(Uc);

        double sp_c = std::abs(get_u(Uc)) + get_sound_speed(Uc);
        double sp_l = std::abs(get_u(Ul)) + get_sound_speed(Ul);
        double sp_r = std::abs(get_u(Ur)) + get_sound_speed(Ur);
        double nu_l = eps * std::max(sp_c, sp_l);
        double nu_r = eps * std::max(sp_c, sp_r);

        Vector3 F_left = (cell > 0)
            ? 0.5 * (computeFlux(Ul) + computeFlux(Uc)) - 0.5 * nu_l * (Uc - Ul)
            : computeFlux(Ul);
        Vector3 F_right = (cell < N - 1)
            ? 0.5 * (computeFlux(Uc) + computeFlux(Ur)) - 0.5 * nu_r * (Ur - Uc)
            : computeFlux(Ur);

        Vector3 R = (Uc - Q_n.segment<3>(3 * cell)) * (dx / dt)
            + (F_right - F_left)
            - computeSource(Uc) * dx;

        // --- Conduction (energy row) ---
        R(2) += conductionResidual(Ul, Uc, Ur) * dx;

        // --- Viscosity (momentum row + energy row) ---
        R(1) += viscMomResidual(Ul, Uc, Ur) * dx;
        R(2) += viscEnResidual(Ul, Uc, Ur) * dx;

        return R;
        };

    // Jacobian for the upwind fluxes formulation
    auto cell_jacobian_upwind = [&](const VectorGlobal& Qvec, int cell)
        -> std::tuple<Matrix3, Matrix3, Matrix3> {
        Vector3 Uc = Qvec.segment<3>(3 * cell);
        Vector3 Ul = (cell > 0) ? Qvec.segment<3>(3 * (cell - 1)) : leftFaceState(Uc);
        Vector3 Ur = (cell < N - 1) ? Qvec.segment<3>(3 * (cell + 1)) : rightFaceState(Uc);

        Matrix3 Jd, Jl, Jr;
        Jr = Matrix3::Zero();

        if (cell == 0) {
            double u_c = get_u(Uc);
            double T_b = 350.0;
            double u_b = 1.0;
            double E_b = R_GAS * T_b / (GAMMA - 1.0) + 0.5 * u_b * u_b;
            double coeff = (GAMMA - 1.0) / (R_GAS * T_b);
            double u2 = u_c * u_c;

            Matrix3 dFace_dUc;
            dFace_dUc(0, 0) = coeff * 0.5 * u2;       dFace_dUc(0, 1) = coeff * (-u_c);       dFace_dUc(0, 2) = coeff;
            dFace_dUc(1, 0) = coeff * 0.5 * u2 * u_b; dFace_dUc(1, 1) = coeff * (-u_c) * u_b; dFace_dUc(1, 2) = coeff * u_b;
            dFace_dUc(2, 0) = coeff * 0.5 * u2 * E_b; dFace_dUc(2, 1) = coeff * (-u_c) * E_b; dFace_dUc(2, 2) = coeff * E_b;

            Jd = Matrix3::Identity() * (dx / dt)
                + computeFluxJacobian(Uc)
                - computeFluxJacobian(leftFaceState(Uc)) * dFace_dUc
                - computeSourceJacobian(Uc) * dx;
            Jl = Matrix3::Zero();

        }
        else if (cell == N - 1) {
            double u_c = get_u(Uc);
            double T_c = get_T(Uc);
            double e_c = R_GAS * T_c / (GAMMA - 1.0);
            double p_b = 10000.0;
            double alpha = p_b * AREA / (R_GAS * T_c);
            double beta = alpha / Uc(0);
            double delta = (GAMMA - 1.0) / (T_c * R_GAS);
            double k = 0.5 * u_c * u_c - e_c;
            double u2 = u_c * u_c;

            Matrix3 dFace_dUc;
            dFace_dUc(0, 0) = beta * (-delta * k);
            dFace_dUc(0, 1) = beta * (delta * u_c);
            dFace_dUc(0, 2) = beta * (-delta);
            dFace_dUc(1, 0) = beta * (-u_c - delta * u_c * k);
            dFace_dUc(1, 1) = beta * (1.0 + delta * u2);
            dFace_dUc(1, 2) = beta * (-delta * u_c);
            dFace_dUc(2, 0) = beta * (-u2 - delta * k * 0.5 * u2);
            dFace_dUc(2, 1) = beta * (u_c + delta * 0.5 * u2 * u_c);
            dFace_dUc(2, 2) = beta * (-delta * 0.5 * u2);

            Jd = Matrix3::Identity() * (dx / dt)
                + computeFluxJacobian(rightFaceState(Uc)) * dFace_dUc
                - computeSourceJacobian(Uc) * dx;
            Jl = -computeFluxJacobian(Ul);

        }
        else {
            Jd = Matrix3::Identity() * (dx / dt)
                + computeFluxJacobian(Uc)
                - computeSourceJacobian(Uc) * dx;
            Jl = -computeFluxJacobian(Ul);
        }

        return { Jd, Jl, Jr };
        };

    // Jacobian for the linear fluxes formulation
    auto cell_jacobian_linear = [&](const VectorGlobal& Qvec, int cell)
        -> std::tuple<Matrix3, Matrix3, Matrix3> {
        Vector3 Uc = Qvec.segment<3>(3 * cell);
        Vector3 Ul = (cell > 0) ? Qvec.segment<3>(3 * (cell - 1)) : leftFaceState(Uc);
        Vector3 Ur = (cell < N - 1) ? Qvec.segment<3>(3 * (cell + 1)) : rightFaceState(Uc);

        Matrix3 Jd, Jl, Jr;
        Jr = Matrix3::Zero();

        if (cell == 0) {
            double u_c = get_u(Uc);
            double T_b = 350.0;
            double u_b = 1.0;
            double E_b = R_GAS * T_b / (GAMMA - 1.0) + 0.5 * u_b * u_b;
            double coeff = (GAMMA - 1.0) / (R_GAS * T_b);
            double u2 = u_c * u_c;

            Matrix3 dFace_dUc;
            dFace_dUc(0, 0) = coeff * 0.5 * u2;       dFace_dUc(0, 1) = coeff * (-u_c);       dFace_dUc(0, 2) = coeff;
            dFace_dUc(1, 0) = coeff * 0.5 * u2 * u_b; dFace_dUc(1, 1) = coeff * (-u_c) * u_b; dFace_dUc(1, 2) = coeff * u_b;
            dFace_dUc(2, 0) = coeff * 0.5 * u2 * E_b; dFace_dUc(2, 1) = coeff * (-u_c) * E_b; dFace_dUc(2, 2) = coeff * E_b;

            Jd = Matrix3::Identity() * (dx / dt)
                + 0.5 * computeFluxJacobian(Uc)
                - computeFluxJacobian(leftFaceState(Uc)) * dFace_dUc
                - computeSourceJacobian(Uc) * dx;
            Jl = Matrix3::Zero();
            Jr = 0.5 * computeFluxJacobian(Ur);

        }
        else if (cell == N - 1) {
            double u_c = get_u(Uc);
            double T_c = get_T(Uc);
            double e_c = R_GAS * T_c / (GAMMA - 1.0);
            double p_b = 10000.0;
            double alpha = p_b * AREA / (R_GAS * T_c);
            double beta = alpha / Uc(0);
            double delta = (GAMMA - 1.0) / (T_c * R_GAS);
            double k = 0.5 * u_c * u_c - e_c;
            double u2 = u_c * u_c;

            Matrix3 dFace_dUc;
            dFace_dUc(0, 0) = beta * (-delta * k);
            dFace_dUc(0, 1) = beta * (delta * u_c);
            dFace_dUc(0, 2) = beta * (-delta);
            dFace_dUc(1, 0) = beta * (-u_c - delta * u_c * k);
            dFace_dUc(1, 1) = beta * (1.0 + delta * u2);
            dFace_dUc(1, 2) = beta * (-delta * u_c);
            dFace_dUc(2, 0) = beta * (-u2 - delta * k * 0.5 * u2);
            dFace_dUc(2, 1) = beta * (u_c + delta * 0.5 * u2 * u_c);
            dFace_dUc(2, 2) = beta * (-delta * 0.5 * u2);

            Jd = Matrix3::Identity() * (dx / dt)
                - 0.5 * computeFluxJacobian(Uc)
                + computeFluxJacobian(rightFaceState(Uc)) * dFace_dUc
                - computeSourceJacobian(Uc) * dx;
            Jl = -0.5 * computeFluxJacobian(Ul);
            Jr = Matrix3::Zero();

        }
        else {
            Jd = Matrix3::Identity() * (dx / dt)
                - computeSourceJacobian(Uc) * dx;
            Jl = -0.5 * computeFluxJacobian(Ul);
            Jr = 0.5 * computeFluxJacobian(Ur);
        }

        return { Jd, Jl, Jr };
        };

    // Jacobian for the linear fluxes formulation with Rusanov correction
    auto cell_jacobian_rusanov = [&](const VectorGlobal& Qvec, int cell)
        -> std::tuple<Matrix3, Matrix3, Matrix3> {
        Vector3 Uc = Qvec.segment<3>(3 * cell);
        Vector3 Ul = (cell > 0) ? Qvec.segment<3>(3 * (cell - 1)) : leftFaceState(Uc);
        Vector3 Ur = (cell < N - 1) ? Qvec.segment<3>(3 * (cell + 1)) : rightFaceState(Uc);

        double sp_c = std::abs(get_u(Uc)) + get_sound_speed(Uc);
        double sp_l = std::abs(get_u(Ul)) + get_sound_speed(Ul);
        double sp_r = std::abs(get_u(Ur)) + get_sound_speed(Ur);
        double nu_l = eps * std::max(sp_c, sp_l);
        double nu_r = eps * std::max(sp_c, sp_r);

        Matrix3 Jd, Jl, Jr;

        if (cell == 0) {
            double u_c = get_u(Uc);
            double T_b = 350.0;
            double u_b = 1.0;
            double E_b = R_GAS * T_b / (GAMMA - 1.0) + 0.5 * u_b * u_b;
            double coeff = (GAMMA - 1.0) / (R_GAS * T_b);
            double u2 = u_c * u_c;
            Matrix3 dFace_dUc;
            dFace_dUc(0, 0) = coeff * 0.5 * u2;       dFace_dUc(0, 1) = coeff * (-u_c);       dFace_dUc(0, 2) = coeff;
            dFace_dUc(1, 0) = coeff * 0.5 * u2 * u_b; dFace_dUc(1, 1) = coeff * (-u_c) * u_b; dFace_dUc(1, 2) = coeff * u_b;
            dFace_dUc(2, 0) = coeff * 0.5 * u2 * E_b; dFace_dUc(2, 1) = coeff * (-u_c) * E_b; dFace_dUc(2, 2) = coeff * E_b;

            Jd = Matrix3::Identity() * (dx / dt)
                + 0.5 * computeFluxJacobian(Uc) + 0.5 * nu_r * Matrix3::Identity()
                - computeFluxJacobian(leftFaceState(Uc)) * dFace_dUc
                - computeSourceJacobian(Uc) * dx;
            Jl = Matrix3::Zero();
            Jr = 0.5 * computeFluxJacobian(Ur) - 0.5 * nu_r * Matrix3::Identity();

        }
        else if (cell == N - 1) {
            double u_c = get_u(Uc);
            double T_c = get_T(Uc);
            double e_c = R_GAS * T_c / (GAMMA - 1.0);
            double p_b = 10000.0;
            double alpha = p_b * AREA / (R_GAS * T_c);
            double beta = alpha / Uc(0);
            double delta = (GAMMA - 1.0) / (T_c * R_GAS);
            double k = 0.5 * u_c * u_c - e_c;
            double u2 = u_c * u_c;
            Matrix3 dFace_dUc;
            dFace_dUc(0, 0) = beta * (-delta * k);
            dFace_dUc(0, 1) = beta * (delta * u_c);
            dFace_dUc(0, 2) = beta * (-delta);
            dFace_dUc(1, 0) = beta * (-u_c - delta * u_c * k);
            dFace_dUc(1, 1) = beta * (1.0 + delta * u2);
            dFace_dUc(1, 2) = beta * (-delta * u_c);
            dFace_dUc(2, 0) = beta * (-u2 - delta * k * 0.5 * u2);
            dFace_dUc(2, 1) = beta * (u_c + delta * 0.5 * u2 * u_c);
            dFace_dUc(2, 2) = beta * (-delta * 0.5 * u2);

            Jd = Matrix3::Identity() * (dx / dt)
                - 0.5 * computeFluxJacobian(Uc) + 0.5 * nu_l * Matrix3::Identity()
                + computeFluxJacobian(rightFaceState(Uc)) * dFace_dUc
                - computeSourceJacobian(Uc) * dx;
            Jl = -0.5 * computeFluxJacobian(Ul) - 0.5 * nu_l * Matrix3::Identity();
            Jr = Matrix3::Zero();

        }
        else {
            Jd = Matrix3::Identity() * (dx / dt)
                + 0.5 * (nu_r + nu_l) * Matrix3::Identity()
                - computeSourceJacobian(Uc) * dx;
            Jl = -0.5 * computeFluxJacobian(Ul) - 0.5 * nu_l * Matrix3::Identity();
            Jr = 0.5 * computeFluxJacobian(Ur) - 0.5 * nu_r * Matrix3::Identity();
        }

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

        return { Jd, Jl, Jr };
        };

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
                Vector3 Ul = (i > 0) ? Q_new.segment<3>(3 * (i - 1)) : leftFaceState(Uc);
                Vector3 Ur = (i < N - 1) ? Q_new.segment<3>(3 * (i + 1)) : rightFaceState(Uc);

                Residual.segment<3>(3 * i) = cell_residual_rusanov(Q_new, i);

                Matrix3 Jd = Matrix3::Zero();
                Matrix3 Jl = Matrix3::Zero();
                Matrix3 Jr = Matrix3::Zero();

                std::tie(Jd, Jl, Jr) = cell_jacobian_rusanov(Q_new, i);

                // --- Assemble ---
                for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) {
                    triplets.push_back({ 3 * i + r, 3 * i + c,     Jd(r,c) });
                    if (i < N - 1) triplets.push_back({ 3 * i + r, 3 * (i + 1) + c, Jr(r,c) });
                    if (i > 0)   triplets.push_back({ 3 * i + r, 3 * (i - 1) + c, Jl(r,c) });
                }
            }

            double res_norm = Residual.norm();
            std::cout << "  iter=" << iter << " |R|=" << res_norm << "\n";
            if (res_norm < NEWTON_TOL) break;

            if (!factorized || iter % REFACTOR_EVERY == 0) {
                Eigen::SparseMatrix<double> J(3 * N, 3 * N);
                J.setFromTriplets(triplets.begin(), triplets.end());
                J.makeCompressed();
                lu_solver.compute(J);
                if (lu_solver.info() != Eigen::Success) {
                    std::cerr << "SparseLU failed at t=" << time << " iter=" << iter << "\n";
                    system("pause");
                    return -1;
                }
                factorized = true;
            }

            double alpha = 1.0;

            Q_new += alpha * lu_solver.solve(-Residual);
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
        std::cout << time << "\n";
        step++;

    }

    // Dopo convergenza Newton, all'ultimo step
    for (int i = 1; i < N - 1; ++i) {
        Vector3 Q = Q_new.segment<3>(3 * i);
        double u = get_u(Q);
        std::cout << "i=" << i << " u=" << std::setprecision(10) << u << "\n";
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