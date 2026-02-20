/**
 * @file Shape_Analysis/src/MeshProcessor.h
 * @author Aravindhkumar Samathur Kalimuthu
 * @date   2026-Jan-01
 */
#ifndef MESHPROCESSOR_H
#define MESHPROCESSOR_H
#include <string>
#include <chrono>
#include <filesystem>
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/tufted_laplacian.h"
#include "geometrycentral/numerical/linear_solvers.h"
#include <Eigen/Dense>
#include "ptr.h"
#include "rapidcsv.h"

namespace ShapeAnalysis
{
    using namespace geometrycentral;

    /**
     * @brief MeshProcessor encapsulates geometric and spectral processing of a surface mesh.
     * This class is responsible for:
     *  - Loading and storing a triangle mesh and its geometry.
     *  - Computing discrete Laplace–Beltrami operators and mass matrices.
     *  - Computing or loading Laplacian eigenvalues and eigenfunctions.
     *  - Providing truncated spectral bases for downstream algorithms.
     *  - Performing spectral projections and mass-matrix–induced inner products.
     * The class supports both standard cotangent Laplacians and robust (intrinsic / tufted) Laplacians via Geometry Central.
     * This class serves as the backbone for spectral shape analysis methods such as HKS, WKS, and functional maps.
     */
    class MeshProcessor
    {
    public:
        /**
         * Creates and returns a shared pointer to a MeshProcessor instance.
         *
         * This is the preferred way to construct a MeshProcessor object.
         * @param fileName Path to the input mesh file.
         * @return Shared pointer to the created MeshProcessor.
         */
        static MeshProcessorSptr create(const std::string& fileName);

        /**
         * class destructor.
         */
        ~MeshProcessor();

        /**
         * Preprocesses the mesh by computing geometric quantities and the Laplacian spectrum.
         * The pipeline consists of:
         *  - Optional computation of per-face normals
         *  - Construction of the discrete Laplace–Beltrami operator L and mass matrix M.
         *  - Computation or loading of the generalized eigenproblem:
         *          - L φ_i = λ_i M φ_i
         * If precomputed eigenpairs are available, they are reused and truncated to the requested number of eigenfunctions.
         * @param no_of_ev Number of eigenvalues and eigenfunctions to compute or retain.
         * @param skipNormals If true, skips face normal computation.
         * @param intrinsic If true, uses intrinsic triangulation.
         * @param robust If true, uses a robust (tufted) Laplacian formulation.
         * @param eigenSpecPath Optional path to precomputed eigenvalues and eigenvectors.
         */
        void process(int no_of_ev, bool skipNormals, bool intrinsic, bool robust, const std::string& eigenSpecPath);

        /**
         * Returns the first @p num Laplacian eigenvalues.
         * @param num Number of eigenvalues to return
         * @return Vector of size (num)
         */
        const Eigen::VectorXd getTruncatedEval(int num) const;

        /**
         * Returns the first @p num Laplacian eigenfunctions.
         * @param num Number of eigenfunctions to return
         * @return Matrix of size (num_vertices × num)
         */
        const Eigen::MatrixXd getTruncatedEvec(int num) const;

        /**
         * Returns the discrete Laplace–Beltrami operator.
         *
         * The Laplacian is constructed during the preprocessing stage and may correspond to either the standard cotangent Laplacian
         * or a robust (intrinsic / tufted) formulation.
         * @return Sparse Laplacian matrix L.
         */
        const Eigen::SparseMatrix<double>& getLaplacian() const;

        /**
         * Returns the vertex lumped mass matrix.
         * The mass matrix is used to define inner products on functions over the mesh and appears in the generalized eigenproblem.
         * @return Sparse mass matrix M
         */
        const Eigen::SparseMatrix<double>& getMassMatrix() const;

        /**
         * Computes the mass matrix induced inner product between functions on the mesh.
         * Given functions f and g defined at mesh vertices, the bilinear form is:
         *      - ⟨f, g⟩_M = fᵀ M g
         * This inner product is used to ensure scale-invariant comparisons and is consistent with the Laplace–Beltrami operator.
         * @param func1 First function(s) defined on vertices.
         * @param func2 Second function(s) defined on vertices.
         * @return Vector of inner products, one per function column.
         */
        Eigen::VectorXd computeBilinearForm(const Eigen::MatrixXd& func1, const Eigen::MatrixXd& func2) const;

        /**
         * Projects vertex-based functions onto the Laplacian eigenbasis.
         * Given a function f defined on vertices, the spectral coefficients are:
         *      - f̂ = Φᵀ M f
         *      - where Φ contains the Laplacian eigenfunctions.
         *
         * @param descr Function(s) defined on mesh vertices.
         * @param K Number of eigenfunctions to use (-1 uses all available).
         * @return Spectral coefficients matrix.
         */
        Eigen::MatrixXd project(const Eigen::MatrixXd& descr, int K);

        /**
         * Computes per-face gradients of scalar or vector-valued functions.
         * @note This method computes gradient vector per faces for given number of time steps.
         * @param function Function(s) defined on vertices
         * @param useSymmetry Whether to enforce symmetric gradients
         * @return Each Eigen::Matrix has size (num_times × 3) per face, where num_times denotes the number of diffusion time steps. The gradient is computed independently for each time step.
         */
        std::vector<Eigen::MatrixXd> computeGradient(const Eigen::MatrixXd& function, bool normalize, bool useSymmetry);

        /**
         * @brief Computes the stiffness matrix W for the orientation operator O(g) = (∇f × ∇g) · n.
         * Mathematical forms (all equivalent via scalar triple product):
         *  - O(g) = (∇f × ∇g) · n [Geometric: oriented parallelogram area]
         *  - O(g) = ∇f · (∇g × n) [Rotated gradient form]
         *  - O(g) = ⟨ n × ∇f, ∇g ⟩ [Form used in implementation]
         *  Implementation details:
         *   - Uses FEM with piecewise linear elements (P1).
         *   - ∇f is constant per face (input parameter).
         *   - Computes the stiffness matrix W such that:
         *      - (W * g)[i] = integral of  φ_i(x) · (∇f × ∇g) · n dA
         *      - where φ_i is the barycentric basis function at vertex i.
         * To obtain pointwise vertex values O(g)_i ≈ (∇f × ∇g)·n at vertex i:
         *  - O_pointwise = inv(M) * W * g
         * @param gradF Vector of gradient of f per face. Each element should be 1x3 matrix.
         * @param rotated If false, computes n × ∇f internally. If true, assumes gradF already contains rotated gradients (n × ∇f).
         * @return Sparse matrix O (n_vertices × n_vertices) such that:  (O * g)[i] ≈ (∇f × ∇g) · n evaluated at vertex i.
         */
        Eigen::SparseMatrix<double> computeOrientationOperator(const std::vector<Eigen::MatrixXd>& gradF, bool rotated);

        /**
        * Method that precomputes the barycentric basis vectors for each face. These basis vectors are required to compute the per-face gradient vector
        * and the orientation operator.
         */
        void computeBarycentricBasis();
        const int& getEigenSpectrumSize() const;
    private:
        /**
         * Constructs a MeshProcessor by loading a surface mesh from disk.
         *
         * The mesh topology and vertex positions are loaded using Geometry Central. No preprocessing (normals, Laplacian, or eigen-decomposition) is performed here.
         * @param fileName Path to the input mesh file.
         */
        explicit MeshProcessor(const std::string& fileName);
    private:
        /**
         * Computes per-face normal vectors.
         *
         * Face normals are computed using Geometry Central and stored internally for later geometric operations.
         */
        void computeFaceNormal();

        /**
         * Computes per-face areas of the mesh.
         *
         * Face areas are required for certain geometric computations such as gradient evaluation and normalization.
         */
        void computeFaceArea();

        /**
         * Constructs the discrete Laplace–Beltrami operator and mass matrix.
         * Depending on the flags, this method computes either:
         *  - The standard cotangent Laplacian and vertex-lumped mass matrix, or A robust (tufted) Laplacian based on intrinsic triangulation.
         * The resulting matrices satisfy the generalized eigenproblem:
         *      - L φ = λ M φ
         * @param no_of_ev Number of eigenpairs to be computed later.
         * @param intrinsic Whether to use intrinsic triangulation.
         * @param robust Whether to use a robust Laplacian formulation.
         */
        void laplacianSpectrum(int no_of_ev, bool intrinsic, bool robust);

        /**
         * Computes the Laplacian eigenvalues and eigenfunctions.
         * Solves the generalized eigenvalue problem:
         *      - L φ_i = λ_i M φ_i.
         * using Geometry Central's sparse eigensolver for symmetric positive-definite systems.
         * Eigenvalues are computed via the Rayleigh quotient:
         *      - λ_i = (φ_iᵀ L φ_i) / (φ_iᵀ M φ_i)
         *
         * @param no_of_ev Number of smallest eigenpairs to compute.
         * @param eigenSpecPath Optional path for loading precomputed spectra.
         */
        void eigenSpectrum(int no_of_ev, const std::string& eigenSpecPath);

        /**
         * Reads precomputed Laplacian eigenvalues and eigenvectors from disk.
         * The method expects two CSV files in the specified directory:
         *      - <mesh_name>_evals.csv : Eigenvalues (no_of_ev × 1)
         *      - <mesh_name>_evecs.csv : Eigenvectors (num_vertices × no_of_ev)
         *
         * The loaded eigenpairs replace any previously computed spectra.
         * @param no_of_ev Number of eigenvalues and eigenvectors to read.
         * @param eigenSpecPath  Directory containing the eigen spectrum files.
         */
        void readPreComputedEigenSpectrum(int no_of_ev, const std::string& eigenSpecPath);
    private:
        // Topological representation of the surface mesh.
        std::unique_ptr<surface::SurfaceMesh> mesh;

        // Geometric embedding of the mesh (vertex positions and geometric queries).
        std::unique_ptr<surface::VertexPositionGeometry> geometry;
        std::string meshName;
    private:
        // Per-face normal vectors.
        surface::FaceData<Vector3> faceNormals;

        // Per-face surface areas.
        surface::FaceData<double> faceArea;

        // Discrete Laplace–Beltrami operator (L) and vertex-lumped mass matrix (M).
        Eigen::SparseMatrix<double> L , M; // Weak Laplacian Matrix and lumped mass matrix.

        // Laplacian eigenvalues λ_i (sorted in ascending order).
        Eigen::VectorXd eigenValues;

        // Corresponding Laplacian eigenfunctions φ_i stored column-wise (num_vertices × num_eigenfunctions).
        Eigen::MatrixXd eigenVectors;

        // Container that stores barycentric basis vectors per face.
        // Three basis vectors are stored for each face, so the size is (3 * number of faces).
        std::vector<Vector3> barycentric_basis;

        // Total number of eigenvalues and eigenvectors computed.
        int eigenSpectrumSize;
    };
}



#endif