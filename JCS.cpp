#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <fstream>

// =========== PARAMETERS
const int N = 100;                      // Number of spatial cells [-]
const double L = 1.0;                   // Length of the domain [m]
const double dx = L / N;                // Axial length of the volumes [m]
const double CFL = 0.8;                 // CFL Number [-]
const double gamma = 1.4;               // Ratio of specific heats (Ideal Gas) [-]
const double gravity_x = 0.0;           // g_x [(Set to -9.81 if vertical) [m/s2]
const double section = 1.0;             // Constant Area A_v [m2]
const double R_vapor = 361.5;           // Sodium vapor constant [J/kgK]

// Newton-Krylov settings
const int max_newton_iters = 20;        // Maximum number of outer iterations [-]
const double newton_tol = 1e-6;         // Maximum non-linear residual tolerated [-]

using Vector3 = Eigen::Vector3d;        // Local state vector (3 x 1)    [rhoA, rhouA, rhoEA]
using Matrix3 = Eigen::Matrix3d;        // Local state matrix (3 x 3)

using VectorGlobal = Eigen::VectorXd;   // Global state vector (3 x N)

// =========== PHYSICS AND STATE HELPERS

// Get pressure value (using EOS)
// p_v * A_v = (gamma - 1) * (rhoEA - 0.5 * (rhouA)^2 / rhoA)
double get_pA(const Vector3& Q) {

    if (Q(0) < 1e-8) return 0.0;
    return (gamma - 1.0) * (Q(2) - 0.5 * Q(1) * Q(1) / Q(0));
}

// Get sound speed (using formula for SS)
// sqrt(gamma * p / rho)
double get_sound_speed(const Vector3& Q) {

    double pA = get_pA(Q);
    double rhoA = Q(0);
    if (rhoA < 1e-8) return 0.0;
    return std::sqrt(gamma * pA / rhoA);
}

// Compute Convective Flux Vector F(Q)
// F = [ rho*u*A,  (rho*u^2 + p)*A,  u*(rho*E + p)*A ]
Vector3 computeFlux(const Vector3& Q) {

    double pA = get_pA(Q);
    double u = Q(1) / Q(0); // u = (rho*u*A) / (rho*A)

    Vector3 F;
    F(0) = Q(1);
    F(1) = Q(1) * u + pA;
    F(2) = u * (Q(2) + pA);
    return F;
}

// Compute Source Terms S(Q) based on your equations
// S = [ Gamma_int*A'_int,  -F_v*A + rho*g*A,  ... ]
Vector3 computeSource(const Vector3& Q) {

    Vector3 S = Vector3::Zero();

    double rhoA = Q(0);
    double u = Q(1) / Q(0);

    // Source parameters
    double Gamma_int = 0.0;                                 // Mass transfer
    double A_prime_int = 0.0;                               // Interfacial area change
    double F_v_friction = 0.02 * u * std::abs(u) * rhoA;    // Simple friction model
    double q_v_int = 0.0;                                   // Heat transfer
    double E_v_int = 0.0;                                   // Energy of transferred mass

    // 1. Mass Conservation Source
    S(0) = Gamma_int * A_prime_int;

    // 2. Momentum Conservation Source: -F_v * A + rho * g_x * A
    S(1) = -F_v_friction * section + rhoA * gravity_x;

    // 3. Energy Conservation Source
    // = rho*u*g*A + (q_int + Gamma*E_int)*A'_int
    S(2) = (rhoA * u * gravity_x) + (q_v_int + Gamma_int * E_v_int) * A_prime_int;

    return S;
}

// =========== JACOBIANS AND LINEARIZATION

// Analytical Jacobian of the Flux dF/dQ
Matrix3 computeFluxJacobian(const Vector3& Q) {

    Matrix3 A;
    double q1 = Q(0); // rho A
    double q2 = Q(1); // rho u A
    double q3 = Q(2); // rho E A

    double u = q2 / q1;
    double u2 = u * u;
    double pA = get_pA(Q);
    double H = (q3 + pA) / q1;  // Total Enthalpy
    double gm1 = gamma - 1.0;

    // Row 1 (Mass)
    A(0, 0) = 0.0;
    A(0, 1) = 1.0;
    A(0, 2) = 0.0;

    // Row 2 (Momentum)
    A(1, 0) = 0.5 * (gm1 - 3.0) * u2;
    A(1, 1) = (3.0 - gamma) * u;
    A(1, 2) = gm1;

    // Row 3 (Energy)
    A(2, 0) = u * (0.5 * gm1 * u2 - H);
    A(2, 1) = H - gm1 * u2;
    A(2, 2) = gamma * u;

    return A;
}

// Analytical Jacobian of the Source dS/dQ
Matrix3 computeSourceJacobian(const Vector3& Q) {

    // For a robust Newton, sources must be discretized
    return Matrix3::Zero();
}

// =========== SOLVER CORE

int main() {

    // Varibles initialization
    VectorGlobal Q_n(3 * N);            // Old Newton state
    VectorGlobal Q_new(3 * N);          // New Newton state

    double p_initial = 10000.0;                                 // 10,000 Pa
    double T_initial = 300.0;                                   // 300 K
    double u_initial = 1.0;                                     // 0 m/s
    double rho_initial = p_initial / (R_vapor * T_initial);     // Density according to EOS

    for (int i = 0; i < N; ++i) {

        double E = p_initial / ((gamma - 1.0) * rho_initial) + 0.5 * u_initial * u_initial;

        // Riempimento Vettore di Stato Q = [rho, rho*u, rho*E] * Area
        Q_n(3 * i + 0) = rho_initial * section;
        Q_n(3 * i + 1) = rho_initial * u_initial * section;
        Q_n(3 * i + 2) = rho_initial * E * section;
    }

    Q_new = Q_n;

    double dt = 0.00000001;      // Time step [s]
    double t_final = 1.0;   // Final time [s]
    double time = 0.0;      // Actual time [s]

    std::cout << "Starting Newton-Krylov FVM Solver..." << std::endl;
    std::cout << "Grid: " << N << " cells. System Size: " << 3 * N << std::endl;

    // 1. APRI IL FILE PRIMA DEL LOOP TEMPORALE
    std::ofstream file("history.csv");
    // Aggiungi la colonna 'time' all'intestazione
    file << "time,x,rho,u,p,T,energy\n";

    // =========== TIME STEPPING LOOP
    int step_counter = 0; // Contatore per decidere ogni quanto salvare

    // =========== TIME STEPPING LOOP
    while (time < t_final) {

        // =========== NEWTON RAPHSON LOOP
        for (int iter = 0; iter < max_newton_iters; ++iter) {

            // Building the Linear System: [J] * deltaQ = -Residual
            // Using Eigen Sparse Matrix
            std::vector<Eigen::Triplet<double>> tripletList;
            tripletList.reserve(N * 3 * 3 * 3); // Approx reservation

            VectorGlobal Residual(3 * N);
            Residual.setZero();

            // Loop over cells to build Residual and Jacobian
            for (int i = 0; i < N; ++i) {

                // =========== BOUNDARY CONDITIONS
                Vector3 Uc = Q_new.segment<3>(3 * i);

                // Recupero primitive interne per calcoli BC
                double rho_in = Uc(0) / section;
                double u_in = Uc(1) / Uc(0);
                double p_in = get_pA(Uc) / section;
                double T_in = p_in / (rho_in * R_vapor);

                // Left ghost cell construction
                Vector3 Ul;

                if (i > 0) {
                    
                    // Standard case, left neighbor
                    Ul = Q_new.segment<3>(3 * (i - 1));
                }
                else {

                    // u=0 (Dirichlet), p=Neumann, T=350 (Dirichlet)

                    // 1. Velocity: symmetrical value to have 0 at the face
                    // u_ghost = -u_internal => u_face = 0.5*(-u + u) = 0
                    double u_b = 1.0;

                    // 2. Pressure: Neumann (copy)
                    double p_b = p_in;

                    // 3. Temperatura: Dirichlet 350 K
                    // Fixed value
                    double T_b = 350.0;

                    // 4. Density calculation from EOS
                    double rho_b = p_b / (R_vapor * T_b);

                    // 5. Rebuilding of left ghost face vector
                    double E_b = p_b / ((gamma - 1.0) * rho_b) + 0.5 * u_b * u_b;

                    Ul(0) = rho_b * section;
                    Ul(1) = rho_b * u_b * section;
                    Ul(2) = rho_b * E_b * section;
                }

                // Right ghost cell construction
                Vector3 Ur;
                if (i < N - 1) {

                    // Standard case, right neighbor
                    Ur = Q_new.segment<3>(3 * (i + 1));
                }
                else {

                    // u=Neumann, p=10000 (Dirichlet), T=Neumann

                    // 1. Velocity: Neumann (copia)
                    double u_b = u_in;

                    // 2. Pressure: Dirichlet 10000 Pa
                    // Fixed value
                    double p_b = 10000.0;

                    // 3. Temperature: Neumann (copy)
                    double T_b = 300;

                    // 4. Density from EOS
                    double rho_b = p_b / (R_vapor * T_b);

                    // 5. Rebuilding of right ghost face vector
                    double E_b = p_b / ((gamma - 1.0) * rho_b) + 0.5 * u_b * u_b;
                    Ur(0) = rho_b * section;
                    Ur(1) = rho_b * u_b * section;
                    Ur(2) = rho_b * E_b * section;
                }

                // → Compute residual
                
                // Term 1: Time derivative (Backward Euler)
                Vector3 time_term = (Uc - Q_n.segment<3>(3 * i)) * (dx / dt);

                // Term 2: Flux Divergence (Central + Dissipation)
                // With scalar artificial dissipation to stabilize the central scheme
                Vector3 Fc = computeFlux(Uc);
                Vector3 Fl = computeFlux(Ul);
                Vector3 Fr = computeFlux(Ur);

                // Artificial Viscosity (Spectral Radius based)
                double spectral_c = std::abs(Uc(1) / Uc(0)) + get_sound_speed(Uc);
                double spectral_l = std::abs(Ul(1) / Ul(0)) + get_sound_speed(Ul);
                double spectral_r = std::abs(Ur(1) / Ur(0)) + get_sound_speed(Ur);

                double eps = 0.5; // Dissipation coefficient
                double nu_l = eps * std::max(spectral_c, spectral_l);
                double nu_r = eps * std::max(spectral_c, spectral_r);

                // Numerical Flux at interfaces, using Jameson-Schmidt-Turkel (JST) scheme
                Vector3 Flux_Right = 0.5 * (Fc + Fr) - 0.5 * nu_r * (Ur - Uc);
                Vector3 Flux_Left = 0.5 * (Fl + Fc) - 0.5 * nu_l * (Uc - Ul);

                Vector3 flux_diff = Flux_Right - Flux_Left;

                // Term 3: Source Terms
                Vector3 Source = computeSource(Uc) * dx;

                // Total Residual for cell i
                Vector3 R_cell = time_term + flux_diff - Source;
                Residual.segment<3>(3 * i) = R_cell;

                // → Compute Jacobian Blocks for [J]
                
                // Diagonal Block (dRi / dQi)
                Matrix3 J_diag = Matrix3::Identity() * (dx / dt); // Time part

                // Flux Jacobian contribution (Central part) is zero for diagonal in pure central,
                // but strictly positive due to Artificial Dissipation part:
                // d(Flux_Right)/dUc = 0.5*A - 0.5*nu*(-I) = 0.5*A + 0.5*nu*I
                // d(Flux_Left)/dUc  = 0.5*A - 0.5*nu*(I)  = 0.5*A - 0.5*nu*I
                // Total Diag = J_diag + (0.5*A + 0.5*nu) - (0.5*A - 0.5*nu) - dSource/dQ
                //            = J_diag + nu * I - dSource/dQ

                Matrix3 dSource = computeSourceJacobian(Uc);
                double nu_total = 0.5 * nu_r + 0.5 * nu_l;          // Average of the artificial dissipation

                J_diag += (nu_total * Matrix3::Identity()) * 1.0;   // Dissipation contribution
                J_diag -= dSource * dx;

                // Off-Diagonal Blocks
                // Right Neighbor (dRi / dQ_{i+1}): From Flux_Right term
                // d/dUr (0.5*Fr - 0.5*nu*Ur) approx 0.5*A_r - 0.5*nu*I
                Matrix3 A_r = computeFluxJacobian(Ur);
                Matrix3 J_right = 0.5 * A_r - 0.5 * nu_r * Matrix3::Identity();

                // Left Neighbor (dRi / dQ_{i-1}): From -Flux_Left term
                // - d/dUl (0.5*Fl - 0.5*nu*(-Ul)) approx - (0.5*A_l + 0.5*nu*I)
                Matrix3 A_l = computeFluxJacobian(Ul);
                Matrix3 J_left = -0.5 * A_l - 0.5 * nu_l * Matrix3::Identity();

                // =========================================================
                // Fill Sparse Matrix Triplets (JACOBIAN ASSEMBLY)
                // =========================================================

                // 1. BLOCCO DIAGONALE (Sempre presente)
                for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++)
                    tripletList.push_back({ 3 * i + r, 3 * i + c, J_diag(r,c) });

                // 2. BLOCCO DESTRO (Interazione con i+1)
                if (i < N - 1) {

                    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++)
                        tripletList.push_back({ 3 * i + r, 3 * (i + 1) + c, J_right(r,c) });
                }

                // 3. BLOCCO SINISTRO (Interazione con i-1)
                if (i > 0) {
                    for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++)
                        tripletList.push_back({ 3 * i + r, 3 * (i - 1) + c, J_left(r,c) });
                }
            }

            // Check Convergence
            double res_norm = Residual.norm();
            if (res_norm < newton_tol) break;

            // =========== KRYLOV SOLVER
            Eigen::SparseMatrix<double> J_global(3 * N, 3 * N);
            J_global.setFromTriplets(tripletList.begin(), tripletList.end());   // Fill Jacobian with triplets

            Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> solver;

            // Preconditioning and solver settings
            solver.preconditioner().setDroptol(0.001); // Drops elements < 0.001 * row_norm
            solver.preconditioner().setFillfactor(7);  // Cap preconditioner size at 7x original matrix
            solver.setMaxIterations(100);              // Maximum iterations for linear solver
            solver.setTolerance(1e-6);                 // Maximum tolerance for linear solver
            solver.compute(J_global);

            if (solver.info() != Eigen::Success) {
                std::cerr << "Decomposition failed!" << std::endl;
                return -1;
            }

            // Solve J * deltaQ = -Residual
            VectorGlobal deltaQ = solver.solve(-Residual);

            if (solver.info() != Eigen::Success) {
                std::cerr << "Solving failed!" << std::endl;
                return -1;
            }

            // Update State
            Q_new += deltaQ;
        }

        std::cout << "Time: " << time << std::endl;

        for (int i = 0; i < N; ++i) {
            Vector3 Q = Q_n.segment<3>(3 * i);

            // Calculating primitive values
            double rho = Q(0) / section;
            double u = 0.0;
            if (Q(0) > 1e-8) u = Q(1) / Q(0);

            double p = get_pA(Q) / section;
            double energy = 0.0;
            if (Q(0) > 1e-8) energy = Q(2) / Q(0);

            double T = 0.0;
            if (rho > 1e-8) {
                T = p / (rho * R_vapor);
            }

            double x = (i + 0.5) * dx;

            file << time << "," << x << "," << rho << "," << u << "," << p << "," << T << "," << energy << "\n";
        }

        file.flush();

        // Advance Time
        Q_n = Q_new;
        time += dt;
        step_counter++;
    }

    file.close();

    std::cout << "Simulation Complete. Data saved in history.csv" << std::endl;

    return 0;
}