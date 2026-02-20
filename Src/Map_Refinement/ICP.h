/**
 * @file Shape_Analysis/src/Map_Refinement/ICP.h
 * @author Aravindhkumar Samathur Kalimuthu
 * @date   2026-Feb-10
 */

#ifndef ICP_H
#define ICP_H
#include "ptrMapRefinement.h"
#include "../Utils.h"

namespace ShapeAnalysis
{
    /**
     * @brief Functional map refinement using the Iterative Closest Point (ICP) algorithm.
     * This class refines an initial functional map between two shapes by iteratively enforcing consistency with a point-to-point correspondence.
     *
     * Functional maps computed in a truncated spectral (eigenbasis) domain often contain noise and typically violate the orthogonality constraint: C @ C^T = I.
     * which is desirable for near-isometric shape matching. This implementation applies an ICP style refinement loop that:
     *      1. Converts the current functional map to a pointwise correspondence.
     *      2. Recomputes an updated functional map from that correspondence.
     *      3. Projects the map onto the closest orthogonal matrix using SVD.
     * The process is repeated until convergence or until a maximum number of iterations is reached, producing a more accurate and stable map.
     */
    class IterativeClosestPoint
    {
    public:
        /**
         * Class Destructor.
         */
        ~ IterativeClosestPoint();

        /**
         * @brief Factory method to create an ICP refinement object.
         * @param fMap Initial functional map matrix (square K×K).
         * @param sourceMesh Source MeshProcessor pointer
         * @param targetMesh Target MeshProcessor pointer
         * @param numIter Maximum number of ICP iterations (<=0 uses default).
         * @param tolerance Convergence threshold based on max coefficient change.
         * @param adjointMap If true, use the adjoint map during pointwise conversion.
         * @return class instance.
         */
        static IterativeClosestPointSptr create(const Eigen::MatrixXd& fMap, const MeshProcessorSptr& sourceMesh, const MeshProcessorSptr& targetMesh, int numIter, double tolerance, bool adjointMap);

        /**
         * @brief Run ICP refinement on the functional map.
         * Executes the iterative ICP procedure until convergence or the maximum number of iterations is reached.
         * @return Refined functional map satisfying approximate orthogonality.
         */
        Eigen::MatrixXd refineICP() const;

    private:
        /**
         * @brief Constructor (use create() instead).
         */
        IterativeClosestPoint(const Eigen::MatrixXd& fMap, const MeshProcessorSptr& sourceMesh, const MeshProcessorSptr& targetMesh, int numIter, double tolerance, bool adjointMa);

        /**
         * @brief Perform a single ICP iteration.
         * Steps:
         *      1. Convert functional map ---> pointwise correspondences.
         *      2. Convert correspondences ---> updated functional map.
         *      3. Enforce orthogonality via SVD projection.
         *
         * @param FM Current functional map.
         * @param truncatedSourceEvecs Source truncated eigenvectors.
         * @param truncatedTargetEvecs Target truncated eigenvectors.
         * @return Orthogonalized functional map after one iteration.
         */
        Eigen::MatrixXd iterateICP(const Eigen::MatrixXd& FM, const Eigen::MatrixXd& truncatedSourceEvecs, const Eigen::MatrixXd& truncatedTargetEvecs) const;
    private:
        // Initial/working functional map.
        Eigen::MatrixXd fMap_;

        // Source Mesh.
        MeshProcessorSptr sourceMeshSptr_;

        // Target Mesh.
        MeshProcessorSptr targetMeshSptr_;

        // Maximum ICP iterations.
        int numIter_ = -1;

        // Convergence threshold.
        double tolerance_;

        // Use adjoint map flag.
        bool adjointMap_;
    };
}

#endif