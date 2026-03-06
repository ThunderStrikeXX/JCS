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
const int    N = 100;
const double L = 1.0;
const double dx = L / N;
const double GAMMA = 1.57;
const double GRAVITY_X = 0.0;
const double AREA = 1.0;
const double R_GAS = 361.5;
const double CONDUCTIVITY = 1.0;
const double VISCOSITY = 1e-5;

// Newton-Raphson settings
const int    MAX_NEWTON_ITERS = 1000;
const double NEWTON_TOL = 1e-6;
const int    REFACTOR_EVERY = 1;
const int    SAVE_EVERY = 10;

// =========== BOUNDARY CONDITIONS (physical) ===========
// Left inlet: Dirichlet forte
const double BC_LEFT_RHO = 10000.0 / (R_GAS * 350.0); // p=10000, T=350
const double BC_LEFT_U = 1.0;
const double BC_LEFT_T = 350.0;

// Right outlet: fixed pressure, zero-gradient rho and u
const double BC_RIGHT_P = 10000.0;

using Vector3 = Eigen::Vector3d;
using Matrix3 = Eigen::Matrix3d;
using VectorGlobal = Eigen::VectorXd;

// =========== PRIMITIVE / CONSERVATIVE CONVERSIONS ===========

// Primitive W = [rho, u, p]  (per-unit-area, i.e. divided by AREA)
// Conservative Q = [rhoA, rhouA, rhoEA]

struct Prim { double rho, u, p; };

Prim Q2W(const Vector3& Q) {
    double rhoA = Q(0), u = Q(1) / Q(0);
    double pA = (GAMMA - 1.0) * (Q(2) - 0.5 * Q(1) * Q(1) / Q(0));
    return { rhoA / AREA, u, pA / AREA };
}

Vector3 W2Q(double rho, double u, double p) {
    double rhoA = rho * AREA;
    double E = p / ((GAMMA - 1.0) * rho) + 0.5 * u * u;
    return { rhoA, rhoA * u, rhoA * E };
}

// dQ/dW  (3x3 matrix, W = [rho,u,p])
Matrix3 dQdW(double rho, double u, double p) {
    double gm1 = GAMMA - 1.0;
    double E = p / (gm1 * rho) + 0.5 * u * u;
    Matrix3 M;
    // dQ0/dW
    M(0, 0) = AREA;       M(0, 1) = 0.0;              M(0, 2) = 0.0;
    // dQ1/dW
    M(1, 0) = AREA * u;   M(1, 1) = AREA * rho;        M(1, 2) = 0.0;
    // dQ2/dW
    // Q2 = rho*A*E, E = p/(gm1*rho) + 0.5*u^2
    // dQ2/drho = A*(E - p/(gm1*rho)) = A*0.5*u^2
    // dQ2/du   = A*rho*u
    // dQ2/dp   = A*rho/(gm1*rho) = A/gm1
    M(2, 0) = AREA * 0.5 * u * u;
    M(2, 1) = AREA * rho * u;
    M(2, 2) = AREA / gm1;
    return M;
}

// dW/dQ  (3x3 matrix)
Matrix3 dWdQ(double rho, double u, double p) {
    // Inverse of dQdW
    double gm1 = GAMMA - 1.0;
    Matrix3 M;
    double invA = 1.0 / AREA;
    // W = [rhoA/A, Q1/Q0, gm1*(Q2 - 0.5*Q1^2/Q0)/A]
    M(0, 0) = invA;        M(0, 1) = 0.0;              M(0, 2) = 0.0;
    M(1, 0) = -u / (rho * AREA); M(1, 1) = 1.0 / (rho * AREA); M(1, 2) = 0.0;
    // p = gm1*(Q2 - 0.5*Q1^2/Q0)/A
    // dp/dQ0 = gm1/A * 0.5*u^2  (from -0.5*Q1^2/Q0 differentiated)
    // dp/dQ1 = gm1/A * (-u)
    // dp/dQ2 = gm1/A
    M(2, 0) = gm1 * 0.5 * u * u * invA;
    M(2, 1) = -gm1 * u * invA;
    M(2, 2) = gm1 * invA;
    return M;
}

// =========== EOS helpers ===========

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

// =========== FLUX AND SOURCE ===========

Vector3 computeFlux(const Vector3& Q) {
    double pA = get_pA(Q), u = get_u(Q);
    return { Q(1), Q(1) * u + pA, u * (Q(2) + pA) };
}

Vector3 computeSource(const Vector3& Q) {
    double rhoA = Q(0), u = get_u(Q);
    return { 0.0, rhoA * GRAVITY_X, rhoA * u * GRAVITY_X };
}

Matrix3 computeFluxJacobian(const Vector3& Q) {
    double u = get_u(Q), u2 = u * u;
    double pA = get_pA(Q), H = (Q(2) + pA) / Q(0), gm1 = GAMMA - 1.0;
    Matrix3 A;
    A(0, 0) = 0.0;                       A(0, 1) = 1.0;             A(0, 2) = 0.0;
    A(1, 0) = 0.5 * (gm1 - 3.0) * u2;         A(1, 1) = (3.0 - GAMMA) * u;   A(1, 2) = gm1;
    A(2, 0) = u * (0.5 * gm1 * u2 - H);         A(2, 1) = H - gm1 * u2;      A(2, 2) = GAMMA * u;
    return A;
}

// =========== VISCOSITY / CONDUCTION helpers (unchanged) ===========

Eigen::RowVector3d dTdQ(const Vector3& Q) {
    double rhoA = Q(0), u = get_u(Q);
    double e_int = Q(2) / rhoA - 0.5 * u * u;
    double c = (GAMMA - 1.0) / R_GAS;
    return { -c * (e_int + 0.5 * u * u) / rhoA, -c * u / rhoA, c / rhoA };
}

Eigen::RowVector3d dudQ(const Vector3& Q) {
    double rhoA = Q(0), u = get_u(Q);
    return { -u / rhoA, 1.0 / rhoA, 0.0 };
}

double conductionResidual(const Vector3& Ql, const Vector3& Qc, const Vector3& Qr) {
    return -CONDUCTIVITY * AREA / (dx * dx) * (get_T(Ql) - 2.0 * get_T(Qc) + get_T(Qr));
}
Eigen::RowVector3d dRcond_dQl(const Vector3& Ql) { return -CONDUCTIVITY * AREA / (dx * dx) * dTdQ(Ql); }
Eigen::RowVector3d dRcond_dQc(const Vector3& Qc) { return  CONDUCTIVITY * AREA / (dx * dx) * 2.0 * dTdQ(Qc); }
Eigen::RowVector3d dRcond_dQr(const Vector3& Qr) { return -CONDUCTIVITY * AREA / (dx * dx) * dTdQ(Qr); }

double viscMomResidual(const Vector3& Ql, const Vector3& Qc, const Vector3& Qr) {
    return -(4.0 / 3.0) * VISCOSITY * AREA / (dx * dx) * (get_u(Ql) - 2.0 * get_u(Qc) + get_u(Qr));
}
double viscEnResidual(const Vector3& Ql, const Vector3& Qc, const Vector3& Qr) {
    double uc = get_u(Qc), ul = get_u(Ql), ur = get_u(Qr);
    double u_right = 0.5 * (uc + ur), u_left = 0.5 * (ul + uc);
    return -(4.0 / 3.0) * VISCOSITY * AREA / (dx * dx) * (u_right * (ur - uc) - u_left * (uc - ul));
}
Eigen::RowVector3d dRviscMom_dQl(const Vector3& Ql) { return -(4.0 / 3.0) * VISCOSITY * AREA / (dx * dx) * dudQ(Ql); }
Eigen::RowVector3d dRviscMom_dQc(const Vector3& Qc) { return  (4.0 / 3.0) * VISCOSITY * AREA / (dx * dx) * 2.0 * dudQ(Qc); }
Eigen::RowVector3d dRviscMom_dQr(const Vector3& Qr) { return -(4.0 / 3.0) * VISCOSITY * AREA / (dx * dx) * dudQ(Qr); }
Eigen::RowVector3d dRviscEn_dQl(const Vector3& Ql, const Vector3& Qc) {
    double ul = get_u(Ql), uc = get_u(Qc);
    return -(4.0 / 3.0) * VISCOSITY * AREA / (dx * dx) * 0.5 * (ul + uc) * dudQ(Ql);
}
Eigen::RowVector3d dRviscEn_dQc(const Vector3& Ql, const Vector3& Qc, const Vector3& Qr) {
    double ul = get_u(Ql), uc = get_u(Qc), ur = get_u(Qr);
    double u_right = 0.5 * (uc + ur), u_left = 0.5 * (ul + uc);
    // d/dQc [ u_right*(ur-uc) - u_left*(uc-ul) ]
    // = 0.5*(ur-uc)*dudQ(Qc) - u_right*dudQ(Qc) - 0.5*(uc-ul)*dudQ(Qc) - u_left*dudQ(Qc)
    double coeff = 0.5 * (ur - uc) - u_right - 0.5 * (uc - ul) - u_left;
    return -(4.0 / 3.0) * VISCOSITY * AREA / (dx * dx) * coeff * dudQ(Qc);
}
Eigen::RowVector3d dRviscEn_dQr(const Vector3& Qc, const Vector3& Qr) {
    double uc = get_u(Qc), ur = get_u(Qr);
    return -(4.0 / 3.0) * VISCOSITY * AREA / (dx * dx) * 0.5 * (uc + ur) * dudQ(Qr);
}

// =========== BOUNDARY CONDITIONS ===========

// Left ghost: Dirichlet forte su (rho, u, T) → Q completamente fissato
Vector3 leftGhostCell() {
    return W2Q(BC_LEFT_RHO, BC_LEFT_U, BC_LEFT_RHO * R_GAS * BC_LEFT_T);
}

// Right ghost: p fissa, rho e u dal bordo interno (zero-gradient)
Vector3 rightGhostCell(const Vector3& Q_last) {
    Prim w = Q2W(Q_last);
    return W2Q(w.rho, w.u, BC_RIGHT_P);
}

// Jacobiano del ghost destro rispetto a Q_last: dQ_ghost / dQ_last
// Q_ghost = W2Q(rho_last, u_last, p_fixed)
// Solo rho e u variano con Q_last, p è fissa
Matrix3 dRightGhost_dQlast(const Vector3& Q_last) {
    Prim w = Q2W(Q_last);
    // dQ_ghost/dQ_last = dQdW(w.rho, w.u, BC_RIGHT_P) * d[rho,u,p_fixed]/dQ_last
    // d[rho,u,p_fixed]/dQ_last = prima due righe di dWdQ + terza riga zero
    Matrix3 dWdQ_last = dWdQ(w.rho, w.u, w.p);
    Matrix3 dQdW_ghost = dQdW(w.rho, w.u, BC_RIGHT_P);
    // La terza riga di dW/dQ riguarda dp_last/dQ_last, ma p_ghost è fissa → azzeriamo
    Matrix3 dW_fixed_dQ = dWdQ_last;
    dW_fixed_dQ.row(2).setZero(); // p_ghost non dipende da Q_last
    return dQdW_ghost * dW_fixed_dQ;
}

// =========== MINMOD LIMITER ===========

inline double minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return (std::abs(a) <= std::abs(b)) ? a : b;
}

// Slope limitata in primitivi per la cella c, data cella sinistra l e destra r
// Restituisce sigma_W = [sigma_rho, sigma_u, sigma_p]  (slope per cella, non /dx)
Eigen::Vector3d slopeW(const Vector3& Ql, const Vector3& Qc, const Vector3& Qr) {
    Prim wl = Q2W(Ql), wc = Q2W(Qc), wr = Q2W(Qr);
    return {
        minmod(wc.rho - wl.rho, wr.rho - wc.rho),
        minmod(wc.u - wl.u,   wr.u - wc.u),
        minmod(wc.p - wl.p,   wr.p - wc.p)
    };
}

// =========== JACOBIANO DELLO SLOPE LIMITATO ===========
// dsigma_W / dQ_k   dove k = -1(left), 0(center), +1(right)
// sigma_j = minmod(wc_j - wl_j, wr_j - wc_j)
//
// La derivata del minmod è:
//   se |a| < |b| e a*b > 0:  dsigma/da = 1,  dsigma/db = 0
//   se |a| > |b| e a*b > 0:  dsigma/da = 0,  dsigma/db = 1
//   se a*b <= 0:              dsigma = 0
//
// Ogni componente j di sigma dipende da wl_j, wc_j, wr_j
// e wk_j = (dWdQ * Qk)(j), quindi dsigma_j/dQk = ds_j/dwk_j * (dWdQ_k)(j,:)

struct SlopeJac {
    // dsigma_W / dQl,  dsigma_W / dQc,  dsigma_W / dQr   (3x3 each)
    Matrix3 dSdQl, dSdQc, dSdQr;
};

SlopeJac slopeJacobian(const Vector3& Ql, const Vector3& Qc, const Vector3& Qr) {
    Prim wl = Q2W(Ql), wc = Q2W(Qc), wr = Q2W(Qr);
    Matrix3 MWl = dWdQ(wl.rho, wl.u, wl.p);
    Matrix3 MWc = dWdQ(wc.rho, wc.u, wc.p);
    Matrix3 MWr = dWdQ(wr.rho, wr.u, wr.p);

    double wl_arr[3] = { wl.rho, wl.u, wl.p };
    double wc_arr[3] = { wc.rho, wc.u, wc.p };
    double wr_arr[3] = { wr.rho, wr.u, wr.p };

    SlopeJac J;
    J.dSdQl.setZero(); J.dSdQc.setZero(); J.dSdQr.setZero();

    for (int j = 0; j < 3; ++j) {
        double a = wc_arr[j] - wl_arr[j];
        double b = wr_arr[j] - wc_arr[j];
        double da_dwl = -1.0, da_dwc = 1.0;
        double db_dwc = -1.0, db_dwr = 1.0;

        double ds_da = 0.0, ds_db = 0.0;
        if (a * b > 0.0) {
            if (std::abs(a) <= std::abs(b)) ds_da = 1.0;
            else                             ds_db = 1.0;
        }

        // dsigma_j / dQl  = ds/da * da/dwl_j * (dWdQ_l)(j,:)
        J.dSdQl.row(j) = ds_da * da_dwl * MWl.row(j);
        // dsigma_j / dQc  = ds/da * da/dwc_j * (dWdQ_c)(j,:) + ds/db * db/dwc_j * (dWdQ_c)(j,:)
        J.dSdQc.row(j) = (ds_da * da_dwc + ds_db * db_dwc) * MWc.row(j);
        // dsigma_j / dQr  = ds/db * db/dwr_j * (dWdQ_r)(j,:)
        J.dSdQr.row(j) = ds_db * db_dwr * MWr.row(j);
    }
    return J;
}

// =========== STATI RICOSTRUITI AL BORDO ===========
// Q_R(i) = stato ricostruito sul lato destro della cella i  → usato nel flusso F_{i+1/2}
// Q_L(i) = stato ricostruito sul lato sinistro della cella i → usato nel flusso F_{i-1/2}

// Q_iR = W2Q( wc + 0.5*sigma )   con sigma = slopeW(Ql,Qc,Qr)
Vector3 reconstructRight(const Vector3& Ql, const Vector3& Qc, const Vector3& Qr) {
    Prim wc = Q2W(Qc);
    Eigen::Vector3d sig = slopeW(Ql, Qc, Qr);
    return W2Q(wc.rho + 0.5 * sig(0), wc.u + 0.5 * sig(1), wc.p + 0.5 * sig(2));
}

// Q_iL = W2Q( wc - 0.5*sigma )
Vector3 reconstructLeft(const Vector3& Ql, const Vector3& Qc, const Vector3& Qr) {
    Prim wc = Q2W(Qc);
    Eigen::Vector3d sig = slopeW(Ql, Qc, Qr);
    return W2Q(wc.rho - 0.5 * sig(0), wc.u - 0.5 * sig(1), wc.p - 0.5 * sig(2));
}

// =========== JACOBIANI DEGLI STATI RICOSTRUITI ===========
// dQ_iR / dQk  e  dQ_iL / dQk   (k = l, c, r)

struct ReconJac {
    Matrix3 dQR_dQl, dQR_dQc, dQR_dQr;
    Matrix3 dQL_dQl, dQL_dQc, dQL_dQr;
};

ReconJac reconstructionJacobian(const Vector3& Ql, const Vector3& Qc, const Vector3& Qr) {
    Prim wc = Q2W(Qc);
    Eigen::Vector3d sig = slopeW(Ql, Qc, Qr);
    Prim wR = { wc.rho + 0.5 * sig(0), wc.u + 0.5 * sig(1), wc.p + 0.5 * sig(2) };
    Prim wL = { wc.rho - 0.5 * sig(0), wc.u - 0.5 * sig(1), wc.p - 0.5 * sig(2) };

    Matrix3 DQdW_R = dQdW(wR.rho, wR.u, wR.p);
    Matrix3 DQdW_L = dQdW(wL.rho, wL.u, wL.p);
    Matrix3 DWdQ_c = dWdQ(wc.rho, wc.u, wc.p);
    SlopeJac SJ = slopeJacobian(Ql, Qc, Qr);

    // Q_iR = Q(wc + 0.5*sigma)
    // dQ_iR/dQk = dQdW_R * (dWdQc * delta_{k,c} + 0.5 * dSigma/dQk)
    // dQ_iL/dQk = dQdW_L * (dWdQc * delta_{k,c} - 0.5 * dSigma/dQk)

    ReconJac RJ;

    // Right reconstructed state
    RJ.dQR_dQl = DQdW_R * (0.5 * SJ.dSdQl);
    RJ.dQR_dQc = DQdW_R * (DWdQ_c + 0.5 * SJ.dSdQc);
    RJ.dQR_dQr = DQdW_R * (0.5 * SJ.dSdQr);

    // Left reconstructed state
    RJ.dQL_dQl = DQdW_L * (-0.5 * SJ.dSdQl);
    RJ.dQL_dQc = DQdW_L * (DWdQ_c - 0.5 * SJ.dSdQc);
    RJ.dQL_dQr = DQdW_L * (-0.5 * SJ.dSdQr);

    return RJ;
}

// =========== RUSANOV FLUX (tra due stati interfaccia) ===========

Vector3 rusanovFlux(const Vector3& QL, const Vector3& QR, double& nu_out) {
    double sp_L = std::abs(get_u(QL)) + get_sound_speed(QL);
    double sp_R = std::abs(get_u(QR)) + get_sound_speed(QR);
    double nu = std::max(sp_L, sp_R);
    nu_out = nu;
    return 0.5 * (computeFlux(QL) + computeFlux(QR)) - 0.5 * nu * (QR - QL);
}

// Jacobiani del flusso di Rusanov rispetto a QL e QR
// dF/dQL = 0.5*A(QL) + 0.5*nu*I  +  0.5*(F(QL)+F(QR) - nu*(QR-QL)) * d(nu)/dQL  [ultimo termine trascurato per ora]
// Per la consistenza del Jacobiano Newton includiamo solo i termini lineari in nu costante
// (nu trattato come costante locale — sufficiente per convergenza Newton)
Matrix3 dRusanov_dQL(const Vector3& QL, const Vector3& QR, double nu) {
    return 0.5 * computeFluxJacobian(QL) + 0.5 * nu * Matrix3::Identity();
}
Matrix3 dRusanov_dQR(const Vector3& QL, const Vector3& QR, double nu) {
    return 0.5 * computeFluxJacobian(QR) - 0.5 * nu * Matrix3::Identity();
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

    std::cout << "FVM MUSCL Solver | N=" << N << " | DOFs=" << 3 * N << "\n";

    std::ofstream f_rho("rho.txt"), f_u("u.txt"), f_p("p.txt"),
        f_T("T.txt"), f_energy("energy.txt");

    auto wall_start = std::chrono::steady_clock::now();
    std::clock_t cpu_start = std::clock();

    // =========== TIME LOOP ===========
    while (time < t_final) {

        Q_new = Q_n;

        Eigen::SparseLU<Eigen::SparseMatrix<double>> lu_solver;
        bool factorized = false;

        // =========== NEWTON-RAPHSON ===========
        for (int iter = 0; iter < MAX_NEWTON_ITERS; ++iter) {

            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(N * 45);   // più triplet per MUSCL (stencil più larga)

            VectorGlobal Residual(3 * N);
            Residual.setZero();

            for (int i = 0; i < N; ++i) {

                // ---- Celle vicine (conservative) ----
                Vector3 Qc = Q_new.segment<3>(3 * i);

                // Cella i-1, i-2, i+1, i+2  (con ghost ai bordi)
                Vector3 Ql = (i > 0) ? Q_new.segment<3>(3 * (i - 1)) : leftGhostCell();
                Vector3 Qr = (i < N - 1) ? Q_new.segment<3>(3 * (i + 1)) : rightGhostCell(Qc);
                Vector3 Qll = (i > 1) ? Q_new.segment<3>(3 * (i - 2)) : leftGhostCell();
                Vector3 Qrr = (i < N - 2) ? Q_new.segment<3>(3 * (i + 2)) : rightGhostCell(Qr);

                // ----------------------------------------------------------------
                // MUSCL reconstruction
                // Flusso destro  F_{i+1/2}:  usa Q_iR (lato destro di i)
                //                             e Q_{i+1,L} (lato sinistro di i+1)
                // Flusso sinistro F_{i-1/2}: usa Q_{i-1,R} (lato destro di i-1)
                //                             e Q_iL (lato sinistro di i)
                // ----------------------------------------------------------------

                // --- Interfaccia destra: tra cella i e i+1 ---
                Vector3 QiR = reconstructRight(Ql, Qc, Qr);   // lato dx di i
                Vector3 Qi1L = reconstructLeft(Qc, Qr, Qrr);  // lato sx di i+1

                double nu_r;
                Vector3 F_right = rusanovFlux(QiR, Qi1L, nu_r);

                // --- Interfaccia sinistra: tra cella i-1 e i ---
                Vector3 Qi1R_ = reconstructRight(Qll, Ql, Qc);   // lato dx di i-1
                Vector3 QiL = reconstructLeft(Ql, Qc, Qr);   // lato sx di i

                double nu_l;
                Vector3 F_left = rusanovFlux(Qi1R_, QiL, nu_l);

                // --- Residuo convettivo ---
                Vector3 R_cell = (Qc - Q_n.segment<3>(3 * i)) * (dx / dt)
                    + (F_right - F_left)
                    - computeSource(Qc) * dx;

                // --- Conduzione ed energia viscosa ---
                R_cell(2) += conductionResidual(Ql, Qc, Qr) * dx;
                R_cell(1) += viscMomResidual(Ql, Qc, Qr) * dx;
                R_cell(2) += viscEnResidual(Ql, Qc, Qr) * dx;

                Residual.segment<3>(3 * i) = R_cell;

                // ================================================================
                // JACOBIANO MUSCL COMPLETO
                // R_i = (dx/dt)*(Qc - Qc_n) + F_right(QiR,Qi1L) - F_left(Qi1R_,QiL) - S*dx
                //
                // dF_right/dQk = dF/dQiR * dQiR/dQk + dF/dQi1L * dQi1L/dQk
                // dF_left/dQk  = dF/dQi1R_ * dQi1R_/dQk + dF/dQiL * dQiL/dQk
                // ================================================================

                ReconJac RJ_i = reconstructionJacobian(Ql, Qc, Qr);  // per QiR, QiL
                ReconJac RJ_i1 = reconstructionJacobian(Qc, Qr, Qrr);  // per Qi1L
                ReconJac RJ_im1 = reconstructionJacobian(Qll, Ql, Qc);  // per Qi1R_

                Matrix3 dFr_dQiR = dRusanov_dQL(QiR, Qi1L, nu_r);
                Matrix3 dFr_dQi1L = dRusanov_dQR(QiR, Qi1L, nu_r);
                Matrix3 dFl_dQi1R = dRusanov_dQL(Qi1R_, QiL, nu_l);
                Matrix3 dFl_dQiL = dRusanov_dQR(Qi1R_, QiL, nu_l);

                // Contributi di F_right al Jacobiano:
                // dF_right/dQll = 0   (QiR non dipende da Qll, Qi1L non dipende da Qll)
                // dF_right/dQl  = dFr/dQiR * dQiR/dQl
                // dF_right/dQc  = dFr/dQiR * dQiR/dQc + dFr/dQi1L * dQi1L/dQc(=Ql di i+1)
                // dF_right/dQr  = dFr/dQiR * dQiR/dQr + dFr/dQi1L * dQi1L/dQr(=Qc di i+1)
                // dF_right/dQrr = dFr/dQi1L * dQi1L/dQrr(=Qr di i+1)
                //
                // Nota: RJ_i:  (Ql,Qc,Qr) → dQiR/d{Ql,Qc,Qr} e dQiL/d{Ql,Qc,Qr}
                //       RJ_i1: (Qc,Qr,Qrr) → dQi1L = dQL/d{Qc,Qr,Qrr} (left recon di i+1)
                //         qui Ql->Qc, Qc->Qr, Qr->Qrr  per la cella i+1

                Matrix3 dFr_dQl = dFr_dQiR * RJ_i.dQR_dQl;
                Matrix3 dFr_dQc = dFr_dQiR * RJ_i.dQR_dQc + dFr_dQi1L * RJ_i1.dQL_dQl;
                Matrix3 dFr_dQr = dFr_dQiR * RJ_i.dQR_dQr + dFr_dQi1L * RJ_i1.dQL_dQc;
                Matrix3 dFr_dQrr = dFr_dQi1L * RJ_i1.dQL_dQr;

                // Contributi di F_left al Jacobiano:
                // dF_left/dQll = dFl/dQi1R_ * dQi1R_/dQll(=Ql di i-1 → Ql di im1 recon)
                //   RJ_im1: (Qll,Ql,Qc) → dQi1R_/d{Qll,Ql,Qc}
                // dF_left/dQl  = dFl/dQi1R_ * dQi1R_/dQl  + dFl/dQiL * dQiL/dQl
                // dF_left/dQc  = dFl/dQi1R_ * dQi1R_/dQc  + dFl/dQiL * dQiL/dQc
                // dF_left/dQr  = dFl/dQiL  * dQiL/dQr
                // dF_left/dQrr = 0

                Matrix3 dFl_dQll = dFl_dQi1R * RJ_im1.dQR_dQl;
                Matrix3 dFl_dQl = dFl_dQi1R * RJ_im1.dQR_dQc + dFl_dQiL * RJ_i.dQL_dQl;
                Matrix3 dFl_dQc = dFl_dQi1R * RJ_im1.dQR_dQr + dFl_dQiL * RJ_i.dQL_dQc;
                Matrix3 dFl_dQr = dFl_dQiL * RJ_i.dQL_dQr;

                // Blocchi Jacobiani di R_i
                // J_{i,i-2}: -dFl/dQll   (solo se i>=2)
                // J_{i,i-1}: dFr/dQl - dFl/dQl  + viscosità/cond su Ql
                // J_{i,i}:   (dx/dt)*I + dFr/dQc - dFl/dQc - dS/dQc*dx + visc/cond su Qc
                // J_{i,i+1}: dFr/dQr - dFl/dQr   + visc/cond su Qr
                // J_{i,i+2}: dFr/dQrr  (solo se i<=N-3)

                Matrix3 Jmm = -dFl_dQll;                             // i-2
                Matrix3 Jm = dFr_dQl - dFl_dQl;                   // i-1
                Matrix3 Jd = Matrix3::Identity() * (dx / dt)
                    + dFr_dQc - dFl_dQc;                    // i
                Matrix3 Jp = dFr_dQr - dFl_dQr;                   // i+1
                Matrix3 Jpp = dFr_dQrr;                             // i+2

                // Conduction & viscosity (stencil a 3 punti su Ql,Qc,Qr)
                Jm.row(2) += dx * dRcond_dQl(Ql);
                Jd.row(2) += dx * dRcond_dQc(Qc);
                Jp.row(2) += dx * dRcond_dQr(Qr);

                Jm.row(1) += dx * dRviscMom_dQl(Ql);
                Jd.row(1) += dx * dRviscMom_dQc(Qc);
                Jp.row(1) += dx * dRviscMom_dQr(Qr);

                Jm.row(2) += dx * dRviscEn_dQl(Ql, Qc);
                Jd.row(2) += dx * dRviscEn_dQc(Ql, Qc, Qr);
                Jp.row(2) += dx * dRviscEn_dQr(Qc, Qr);

                // Quando il ghost dipende dalla soluzione interna, serve propagare
                // la derivata. Per il ghost destro (cella N-1):
                //   Qr = rightGhostCell(Qc), quindi dQr/dQc = dRightGhost_dQlast(Qc)
                //   I termini Jp vanno corretti: Jd += Jp * dRightGhost_dQlast
                // Per il ghost sinistro: è Dirichlet puro → dQl/dQc = 0, nessuna correzione.
                if (i == N - 1) {
                    Matrix3 dGhost_dQc = dRightGhost_dQlast(Qc);
                    Jd += Jp * dGhost_dQc;
                    // Jp non viene assemblato (i+1 fuori dominio)
                }
                // Analogamente per Qrr = rightGhostCell(Qr) quando i == N-2
                if (i == N - 2) {
                    Matrix3 dGhost_dQr = dRightGhost_dQlast(Qr);
                    Jp += Jpp * dGhost_dQr;
                    // Jpp non viene assemblato (i+2 fuori dominio)
                }

                // ---- Assemblaggio triplets ----
                auto addBlock = [&](int row_cell, int col_cell, const Matrix3& M) {
                    if (col_cell < 0 || col_cell >= N) return;
                    for (int r = 0; r < 3; r++)
                        for (int c = 0; c < 3; c++)
                            triplets.push_back({ 3 * row_cell + r, 3 * col_cell + c, M(r,c) });
                    };

                addBlock(i, i - 2, Jmm);
                addBlock(i, i - 1, Jm);
                addBlock(i, i, Jd);
                if (i < N - 1) addBlock(i, i + 1, Jp);
                if (i < N - 2) addBlock(i, i + 2, Jpp);
            }

            double res_norm = Residual.norm();

            if (res_norm < NEWTON_TOL) break;

            if (!factorized || iter % REFACTOR_EVERY == 0) {
                Eigen::SparseMatrix<double> J(3 * N, 3 * N);
                J.setFromTriplets(triplets.begin(), triplets.end());
                J.makeCompressed();
                lu_solver.compute(J);
                if (lu_solver.info() != Eigen::Success) {
                    std::cerr << "SparseLU failed at t=" << time << " iter=" << iter
                        << " |R|=" << res_norm << "\n";
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
                f_T << T << ", "; f_energy << e << ", ";
            }
            f_rho << "\n"; f_u << "\n"; f_p << "\n"; f_T << "\n"; f_energy << "\n";
            f_rho.flush(); f_u.flush(); f_p.flush(); f_T.flush(); f_energy.flush();
        }

        Q_n = Q_new;
        time += dt;
        step++;

        if (std::abs(time - 0.5) < 0.5 * dt) {
            std::cout << "\n=== T profile at t=0.5 ===\n";
            for (int i = 0; i < N; ++i) {
                Vector3 Q = Q_new.segment<3>(3 * i);
                std::cout << get_T(Q) << ", ";
            }
            std::cout << "\n";
        }
    }

    // =========== FLUX BALANCE DIAGNOSTICS ===========
    // Esegui solo all'ultimo step
    if (time + dt >= t_final) {

        // Flussi alle interfacce (primo ordine per semplicità diagnostica)
        double mass_flux_left = Q_new(1) / AREA;                          // rho*u alla cella 0 (bordo sx)
        double mass_flux_right = Q_new(3 * (N - 1) + 1) / AREA;                  // rho*u alla cella N-1 (bordo dx)

        // Flusso di massa attraverso ogni interfaccia interna
        std::cout << "\n=== FLUX BALANCE AT t_final ===\n";
        std::cout << "rho*u at left  boundary (i=0):   " << mass_flux_left << "\n";
        std::cout << "rho*u at right boundary (i=N-1): " << mass_flux_right << "\n";
        std::cout << "mass flux difference: " << mass_flux_right - mass_flux_left << "\n\n";

        // Bilancio globale: somma dei residui temporali
        double dmass_dt = 0.0, dmom_dt = 0.0, denergy_dt = 0.0;
        for (int i = 0; i < N; ++i) {
            Vector3 dQ = (Q_new.segment<3>(3 * i) - Q_n.segment<3>(3 * i)) / dt;
            dmass_dt += dQ(0);
            dmom_dt += dQ(1);
            denergy_dt += dQ(2);
        }
        std::cout << "d(total_mass)/dt:   " << dmass_dt * dx << "\n";
        std::cout << "d(total_mom)/dt:    " << dmom_dt * dx << "\n";
        std::cout << "d(total_energy)/dt: " << denergy_dt * dx << "\n\n";

        // Profilo del flusso di massa cella per cella
        std::cout << "rho*u per cell:\n";
        for (int i = 0; i < N; ++i) {
            double rhou = Q_new(3 * i + 1) / AREA;
            std::cout << "i=" << i << " rho*u=" << rhou << "\n";
        }

        // Flusso numerico all'interfaccia i+1/2 (Rusanov, primo ordine)
        std::cout << "\nNumerical mass flux at interfaces F_{i+1/2}:\n";
        double eps = 1e-8;
        for (int i = 0; i < N - 1; ++i) {
            Vector3 Uc = Q_new.segment<3>(3 * i);
            Vector3 Ur = Q_new.segment<3>(3 * (i + 1));
            double sp_c = std::abs(get_u(Uc)) + get_sound_speed(Uc);
            double sp_r = std::abs(get_u(Ur)) + get_sound_speed(Ur);
            double nu = eps * std::max(sp_c, sp_r);
            double F_mass = 0.5 * (Q_new(3 * i + 1) + Q_new(3 * (i + 1) + 1)) - 0.5 * nu * (Ur(0) - Uc(0));
            F_mass /= AREA;
            std::cout << "i=" << i << " F_mass_{i+1/2}=" << F_mass << "\n";
        }
    }

    f_rho.close(); f_u.close(); f_p.close(); f_T.close(); f_energy.close();

    std::clock_t cpu_end = std::clock();
    std::cout << "CPU time: " << (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC << " s\n";
    auto wall_end = std::chrono::steady_clock::now();
    std::cout << "Wall clock time: "
        << std::chrono::duration<double>(wall_end - wall_start).count() << " s\n";

    system("pause");
    return 0;
}