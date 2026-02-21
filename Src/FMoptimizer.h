/**
 * @file Shape_Analysis/src/FMoptimizer.h
 * @author Aravindhkumar Samathur Kalimuthu
 * @date   2026-Jan-01
 */

#ifndef FMOPTIMIZER_H
#define FMOPTIMIZER_H
#include <iostream>
#include <Eigen/Core>
#include <LBFGSB.h>
#include "ptr.h"
#include "Utils.h"

namespace  ShapeAnalysis
{
    /**
     * @class FunctionalMapEnergy
     * @brief Energy formulation and optimizer for computing a functional map between two shapes in a truncated Laplace–Beltrami eigenbasis.
     * This class implements a weighted least-squares energy commonly used in  Functional Map frameworks:
     *
     * E(C) =   w_descr * 0.5 || C D₁ − D₂ ||²_F + w_lap * 0.5 || C Λ₁ − Λ₂ C ||²_F + w_descrComm  * 0.5 Σ || C Aᵢ − Bᵢ C ||²_F
     * where:
     *      - C ∈ ℝ^{K×K} is the functional map in spectral domain.
     *      - D₁, D₂ are reduced descriptor matrices expressed in the eigenbases of the source and target shapes.
     *      - Λ₁, Λ₂ are diagonal matrices of Laplace–Beltrami eigenvalues.
     *      - (Aᵢ, Bᵢ) are pairs of descriptor-induced multiplication operators.
     *
     * The optimization is performed using the LBFGS++ solver by minimizing the above energy with respect to C.
     * The class is designed to be used as a callable energy functor compatible with LBFGS++.
     */
    class FunctionalMapEnergy
    {
    public:
        /**
         * @brief Factory method to construct a FunctionalMapEnergy instance.
         * @param wDescr Weight for descriptor preservation term.
         * @param wLap Weight for Laplacian commutativity term.
         * @param wDescrComm Weight for descriptor-operator commutativity terms.
         * @param wOrient  Weight reserved for orientation-based constraints (currently unused)
         * @param descr1Reduced Reduced descriptor matrix for shape 1 expressed in its Laplace–Beltrami eigenbasis (K × M).
         * @param descr2Reduced Reduced descriptor matrix for shape 2 expressed in its Laplace–Beltrami eigenbasis (K × M).
         * @param descrOperatorPairs Pairs of descriptor induced multiplication operators  (Aᵢ, Bᵢ) in the spectral domain.
         * @param orientationOperatorPairs Pairs of orientation operator operators  (Aᵢ, Bᵢ) in the spectral domain.
         * @param eigenvalueSqDiff Precomputed squared eigenvalue differences ( (λ₁ᵢ − λ₂ⱼ)² ), used for efficient Laplacian commutativity evaluation.
         * @return Shared pointer to the constructed FunctionalMapEnergy instance.
         */
        static FunctionalMapEnergySptr create(const OptimizationParameters& params);

        /**
         * class destructor
         */
        ~FunctionalMapEnergy();

        /**
         * @brief Evaluate the functional map energy and its gradient.
         *
         * This operator is called by the LBFGS++ optimizer. The input vector `x`
         * represents the functional map C flattened column-wise into a vector.
         * @param x Flattened functional map matrix (size K²).
         * @param grad Output gradient vector with respect to x (size K²).
         * @return The scalar energy value E(C).
         * @note Internally, x is reshaped into a K × K matrix C using Eigen::Map. The gradient is computed analytically for all enabled terms.
         */
        double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) const;

        /**
         * @brief Solve the functional map optimization problem.
         * Initializes the functional map with an identity matrix and minimizes the energy using the LBFGS++ solver.
         * Upon convergence (or termination), the optimized functional map is stored internally and can be accessed via getMatrixC().
         */
        void solve();

        /**
         * @brief Retrieve the optimized functional map.
         * @return Constant reference to the optimized functional map C (K × K).
         */
        const Eigen::MatrixXd& getMatrixC() const;

    private:
        /**
         * class constructor
         */
        FunctionalMapEnergy(const OptimizationParameters& params);

        /**
         * @brief Orthogonality preservation constraint.
         *
         * For isometric source and target shapes, the optimal functional map C should be orthogonal (i.e. C Cᵀ = I). This term penalizes deviations
         * from orthogonality using the Frobenius norm:
         *          - E(C) = || C Cᵀ − I ||²_F
         * The function returns both the energy and its gradient:
         *          -  ∇E(C) = 4 (C Cᵀ − I) C
         * This regularizer encourages C to remain close to an orthogonal matrix.
         *
         * @warning
         *  - This term should be used with care during optimization. Enforcing strict
         *  - orthogonality can degrade correspondences when the shapes are not close to isometric.
         *  - In practice, orthogonality is typically imposed only during map refinement rather than during the main optimization.
         * @param C Square functional map matrix.
         * @return Pair (energy, gradient).
         */
        std::pair<double, Eigen::MatrixXd> orthogonalEnergyTerm(const Eigen::MatrixXd& C) const;
    private:
        double wDescr_;
        double wLap_;
        double wDescrComm_;
        double wOrient_;
        bool orthogonalEnergy_ = false;

        Eigen::MatrixXd descr1_;
        Eigen::MatrixXd descr2_;
        std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> descrOperatorPairs_;
        std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> orientationOperatorPairs_;
        Eigen::MatrixXd eigenvalueSqDiff_;


    private:
        // Truncated basis dimension.
        int K;

        // Number of energy function evaluations (for diagnostics).
        mutable int functional_calls = 0;

        // Optimized functional map matrix.
        Eigen::MatrixXd C_; // final functional Map
    };
};
#endif
