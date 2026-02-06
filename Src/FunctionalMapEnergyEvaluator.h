#ifndef FUNCTIONALMAPENERGYEVALUATOR_H
#define FUNCTIONALMAPENERGYEVALUATOR_H
#include "ptr.h"
#include "Utils.h"

namespace ShapeAnalysis
{
    /**
     * Computes the multi-term functional map energy used for shape correspondence.
     * The total energy has the form:
     *      wDescr * descriptor preservation +
     *      wLap* Laplacian commutation +
     *      wDescrComm * descriptor operator commutation +
     *      wOrient * orientation operator commutation
     *
     * All terms are evaluated in the reduced spectral basis.
     */
    class FunctionalMapEnergyEvaluator
    {
    public:
        /**
         * Destructor.
         */
        ~FunctionalMapEnergyEvaluator();

        /**
         * Factory method.
         * Constructs an EnergyFunction using the provided optimization parameters.
         * Copies all operators, descriptors, and weights internally.
         * @param parameters
         * @return
         */
        static FunctionalMapEnergyEvaluatorSptr create(const OptimizationParameters& parameters);

        /**
         * Computes the full weighted energy evaluated at the identity map C = I.
         * This is mainly used  for weight normalization and also to estimate baseline energy magnitude.
         * @return Total scalar energy value.
         */
        double computeEnergy();

        /**
         * Computes the operator commutativity energy:
         *      0.5 * sum_i || C A_i - B_i C ||_F^2
         * Each pair (A_i, B_i) represents corresponding operators on the two shapes, and ued for
         *      - descriptor operators.
         *      - orientation operators.
         * @param C  Functional map matrix.
         * @param operatorPairs  operatorPairs List of corresponding operator pairs (A_i, B_i).
         * @return Scalar commutation energy.
         */
        double operatorCommutation(const Eigen::MatrixXd& C, const std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>& operatorPairs);

    private:
        /**
         * Class constructor.
         * @param parameters
         */
        FunctionalMapEnergyEvaluator(const OptimizationParameters& parameters);

        /**
         * Descriptor preservation (data term):
         *      - 0.5 * || C D1 - D2 ||_F^2
         * Encourages mapped descriptors on shape 1 to match descriptors on shape 2.
         * This is the main data-fitting term driving correspondence.
         *
         * @param C Functional map matrix.
         * @param descr1 computed descriptor on the source shape.
         * @param descr2 computed descriptor on the target shape
         * @return Scalar descriptor energy.
         */
        double descriptorPreservation(const Eigen::MatrixXd& C, const Eigen::MatrixXd& descr1, const Eigen::MatrixXd& descr2);

        /**
         * Laplacian commutativity energy:
         *      - 0.5 * || C Λ1 - Λ2 C ||_F^2
         * Uses a precomputed matrix of squared eigenvalue differences for efficiency:
         *      - eigenvalueSqDiff(i,j) = (λ1_i - λ2_j)^2
         * This term enforces intrinsic isometry and smoothness of the map.
         *
         * @param C Functional map matrix.
         * @param eigenvalueSqDiff
         * @return Scalar LB commutative energy.
         */
        double laplacianCommutation(const Eigen::MatrixXd& C, const Eigen::MatrixXd& eigenvalueSqDiff);

    private:
        int K; // Truncated basis dimension.
        double wDescr_;
        double wLap_;
        double wDescrComm_;
        double wOrient_;

        Eigen::MatrixXd descr1_;
        Eigen::MatrixXd descr2_;
        std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> descrOperatorPairs_;
        std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> orientationOperatorPairs_;
        Eigen::MatrixXd eigenvalueSqDiff_;
    };
}

#endif