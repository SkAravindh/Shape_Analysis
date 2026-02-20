/**
 * @file Shape_Analysis/src/FunctionalMapping.h
 * @author Aravindhkumar Samathur Kalimuthu
 * @date   2026-Jan-01
 */

#ifndef FUNCTIONALMAP_H
#define FUNCTIONALMAP_H
#include <fstream>
#include <optional>
#include "Utils.h"

namespace ShapeAnalysis
{
    /**
     * @class FunctionalMapping
     * @brief Implements the Functional Map framework for shape correspondence.
     * This class computes a functional map between a source and a target mesh by:
     *  - Computing spectral bases (Laplace–Beltrami eigenfunctions).
     *  - Extracting intrinsic descriptors (HKS / WKS).
     *  - Projecting descriptors into the spectral domain.
     *  - Solving an optimization problem to estimate the functional map matrix.
     *  - Optionally recovering point-to-point correspondences.
     */
    class FunctionalMapping
    {
    public:
        /**
         * @brief Factory method to create a FunctionalMapping instance.
         * @param sourceMesh Shared pointer to the source mesh processor.
         * @param targetMesh Shared pointer to the target mesh processor.
         * @return Shared pointer to the created FunctionalMapping object.
         */
        static FunctionalMappingSptr create(const MeshProcessorSptr& sourceMesh, const MeshProcessorSptr& targetMesh);

        /**
         * @brief Default destructor.
         */
        ~FunctionalMapping() = default;

        /**
         * @brief Preprocesses meshes and computes feature descriptors.
         * Steps:
         *  - Computes Laplace–Beltrami spectra for both meshes.
         *  - Computes intrinsic descriptors (HKS or WKS).
         *  - Optionally augments descriptors using landmark-based features.
         *  - Subsamples and normalizes descriptors.
         * @param params Preprocessing parameters.
         */
        void preprocess(const PreProcessParameters& params);

        /**
         * @brief Estimates the functional map matrix.
         * This Method:
         *  - Projects descriptors onto the spectral bases.
         *  - Constructs descriptor multiplication operators.
         *  - Computes Laplacian commutativity terms.
         *  - Solves the functional map optimization problem.
         * @param params Optimization and regularization parameters.
         */
        void fit(const FitParameters& params);

        /**
         * @brief Computes a point-to-point correspondence from the functional map.
         * The correspondence is obtained via nearest-neighbor search in the spectral embedding induced by the functional map.
         * @return Returns, for each vertex in the target mesh, its corresponding vertex in the source shape and the associated spectral domain distance.
         */
        std::pair<std::vector<size_t>, std::vector<double>> computePointToPoint() const;

        /**
         * @brief Helper method that initiates actual ICP method which further performs the functional map refinement procedure.
         * @return Returns, for each vertex in the target mesh, its corresponding vertex in the source shape and the associated spectral domain distance.
         */
        std::pair<std::vector<size_t>, std::vector<double>> iterativeClosestPointRefinement();

        /**
         * @brief Helper method that initiates actual Zoom Out method which further performs the functional map refinement procedure.
         * @return Returns, for each vertex in the target mesh, its corresponding vertex in the source shape and the associated spectral domain distance.
         */
        std::pair<std::vector<size_t>, std::vector<double>> zoomOutRefinement();

    private:
        /**
         * @brief Constructor (use factory method instead).
         *
         * @param sourceMesh Source mesh processor.
         * @param targetMesh Target mesh processor.
         */
        FunctionalMapping(const MeshProcessorSptr& sourceMesh, const MeshProcessorSptr& targetMesh);
    private:
        /**
         * @brief Reads landmark vertex indices from a file.
         * The file is expected to contain one correspondence per line: (source_vertex_index, target_vertex_index).
         * @param fileName fileName Path to the landmark file.
         * @param expectedCount expectedCount Expected number of landmark pairs.
         */
        void readLandMarkFile(const std::string& fileName, unsigned expectedCount);

        /**
         * @brief Computes paired reduced orientation operators for both meshes.
         * For each of the N descriptor features, computes the reduced orientation operator on source and target meshes, returning them as paired (source, target) matrices.
         * @param reversing Whether to reverse orientation.
         * @param normalize Whether to normalize the operator.
         * @return Vector of N pairs of k×k matrices, where N = features, k = reduced basis size.
         * @example Example: 120 features with 35 eigenfunctions → returns 120 pairs of 35×35 matrices.
         */
         std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> computeOrientationOperator(bool reversing, bool normalize) const;

        /**
         * @brief Computes descriptor multiplication operators.
         * For each descriptor f, this computes the operator:  Φᵀ M (f ⊙ Φ) which encodes pointwise multiplication in the reduced spectral basis.
         * @return std::vector of (source, target) descriptor operators.
         */
        std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> computeDescrOperator() const;

        /**
         * @brief Projects descriptors onto the Laplacian eigenbasis.
         * @param descr descr Descriptor matrix in the spatial domain.
         * @param mesh mesh Identifier for source or target mesh.
         * @param K Optional number of eigenfunctions to use.
         * @return Descriptor coefficients in the spectral domain.
         */
        Eigen::MatrixXd project(const Eigen::MatrixXd& descr, MeshID mesh, std::optional<int> K) const;

        /**
         * @brief Computes squared differences between Laplacian eigenvalues.
         * This matrix is used to enforce Laplacian commutativity in the functional map optimization.
         * @return Matrix of squared eigenvalue differences.
         */
        Eigen::MatrixXd computeEigenvalueSqDiff() const;

        /**
         * @brief A method that creates instance of ICP class and initiates the map refinement process.
         * @param numIter Maximum ICP iterations.
         * @param tolerance Convergence threshold.
         * @param adjointMap Use adjoint map flag.
         * @return Returns, for each vertex in the target mesh, its corresponding vertex in the source shape and the associated spectral domain distance.
         */
        std::pair<std::vector<size_t>, std::vector<double>> iterativeClosestPointRefinement(int numIter, double tolerance, bool adjointMap);

        /**
         * @brief A method that creates instance of Zoom Out class and initiates the map refinement process.
         * @param numIter Maximum Zoom Out iteration.
         * @param stepSize Spectral increment.
         * @return Returns, for each vertex in the target mesh, its corresponding vertex in the source shape and the associated spectral domain distance.
         */
        std::pair<std::vector<size_t>, std::vector<double>> zoomOutRefinement(int numIter, int stepSize);

    private:
        // Non-owning shared references to input meshes
        MeshProcessorSptr sourceMeshSptr;
        MeshProcessorSptr targetMeshSptr;

        // Spectral dimensions (K1: source, K2: target)
        // K1 is for mesh1, whereas K2 is for mesh2
        std::pair<int, int> K1;
        std::pair<int, int> K2;
        int kProcess;

        // Normalized descriptors
        Eigen::MatrixXd sourceDescriptor;
        Eigen::MatrixXd targetDescriptor;

        // Landmark correspondences (source_idx, target_idx)
        Eigen::MatrixXd landMarkPoints;

        // Functional map matrix C
        Eigen::MatrixXd FM;

        // Refined Functional map matrix C using ICP
        Eigen::MatrixXd FM_ICP;

        // Refined Functional map matrix C using ICP
        Eigen::MatrixXd FM_ZoomOut;
    };

}


#endif