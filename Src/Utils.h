/**
 * @file Shape_Analysis/src/Utils.h
 * @author Aravindhkumar Samathur Kalimuthu
 * @date   2026-Jan-01
 */
#ifndef UTILS_H
#define UTILS_H
#include <string>
#include <Eigen/Dense>
#include <filesystem>
#include "ptr.h"

/**
 * @brief Utility functions and lightweight data structures used across the shape analysis and functional map pipeline.
 *
 * This file contains helper enums, parameter containers, descriptor manipulation routines, and nearest-neighbor utilities used for
 * spectral embedding and point-to-point correspondence.
 */
namespace ShapeAnalysis
{
    // Type of spectral descriptor used for shape analysis.
    enum class DescriptorType {HKS, WKS};

    // Refinement strategy for functional map optimization.
    enum class RefinementType {Classic, ICP, ZoomOut};

    // Identifier for input meshes.
    enum class MeshID {Mesh1, Mesh2};

    /**
     * @brief Container for preprocessing parameters used in spectral analysis.
     * This structure controls eigen-decomposition size, descriptor computation,  landmark handling, and debugging behavior.
     */
    struct PreProcessParameters
    {
        // Number of eigenvalues/eigenvectors used for each mesh (mesh1, mesh2).
        std::pair<int, int> n_EV = std::make_pair(50,50);

        // Number of descriptor samples (time steps or energy levels).
        int nDescr = 100;

        // Descriptor type (HKS or WKS).
        DescriptorType descrType = DescriptorType::WKS;

        // Path to landmark file (optional).
        std::string landMarkPath = std::string();

        // Path to precomputed eigen spectrum for mesh 1.
        std::string eigenSpectraPathMesh1 = std::string();

        // Path to precomputed eigen spectrum for mesh 2.
        std::string eigenSpectraPathMesh2 = std::string();

        // Number of landmark points.
        int landMarkPointsCount = 0;

        // Subsampling step for descriptors.
        int subSampleStep = 1;

        // Number of parallel processes (if applicable).
        int kProcess = 0;

        // Enables verbose/debug output.
        bool debug = false;
    };

    /**
     *  @brief Parameters controlling functional map fitting.
     */
    struct FitParameters
    {
        // Descriptor preservation weight.
        double w_descr = 1e-1;

        // Laplacian commutativity weight.
        double w_lap = 1e-3;

        // Descriptor commutativity weight.
        double w_dcomm = 1;

        // Orientation consistency weight.
        double w_orient = 0;

        // Allow orientation-reversing correspondences.
        bool orient_reversing = false;

        // Initialization strategy for optimization.
        std::string opt_init = "Zeros";
    };

    /**
     * @brief Dynamic point cloud structure compatible with nanoflann.
     * Stores points in a flat array of size (num_points × dimension) and provides the required interface for KD-tree nearest neighbor queries.
     * @tparam T Scalar type (e.g., double)
     */
    template <typename T>
    struct PointCloudDynamic
    {
        size_t dim;
        std::vector<T> pts; // size = num of points * dim

        // Must return the number of data points
        inline size_t kdtree_get_point_count() const
        {
            return pts.size() / dim;
        }

        // Returns the dim'th component of the idx'th point in the class:
        inline T kdtree_get_pt(size_t idx, size_t d) const
        {
            return pts[(idx * dim) + d];
        }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX&) const
        {
            return false;
        }
    };

    /**
     * @brief Checks whether a file exists on disk.
     * @param fileName Path to the file.
     * @return True if the file exists, false otherwise.
     */
    bool isFileExist(const std::string& fileName);

    /**
     * @brief Horizontally concatenates two descriptor matrices.
     * Both descriptor matrices must have the same number of rows (corresponding to vertices).
     * @param descr1 First descriptor matrix (num_vertices × d1).
     * @param descr2 Second descriptor matrix (num_vertices × d2).
     * @return Concatenated descriptor matrix (num_vertices × (d1 + d2)).
     */
    Eigen::MatrixXd hStackDescriptors(const Eigen::MatrixXd& descr1, const Eigen::MatrixXd& descr2);

    /**
     * @brief Subsamples descriptor columns at a fixed step size.
     * This is typically used to reduce descriptor dimensionality by selecting every k-th column.
     * @param descr Descriptor matrix (num_vertices × num_descriptors).
     * @param sampleStep Subsampling step size.
     * @return Subsampled descriptor matrix.
     */
    Eigen::MatrixXd subSampleDescriptors(const Eigen::MatrixXd& descr, int sampleStep);

    /**
     * @brief Establishes point-to-point correspondence using a functional map.
     * Eigenfunctions of both meshes are projected into a shared spectral embedding using the functional map.
     * Point-to-point correspondence is then computed via nearest neighbor search in the spectral domain.
     * @Note Correspondence is established from mesh 2 to mesh 1.
     *
     * @param FM Functional map from mesh 1 to mesh 2.
     * @param mesh1 Source mesh.
     * @param mesh2 Target mesh.
     * @param adjointMap If true, applies the map to the target eigenbasis, otherwise applies the transpose to the source basis.
     * @return pair of (corresponding vertex indices, squared distances).
     */
    std::pair<std::vector<size_t>, std::vector<double>> meshToP2P(const Eigen::MatrixXd& FM, const MeshProcessorSptr& mesh1, const MeshProcessorSptr& mesh2, bool adjointMap);

    /**
     * @brief Performs nearest neighbor search using a KD-tree.
     * For each point in the target embedding, the nearest point in the source embedding is found using Euclidean distance.
     * @param sourceEmdedding Source point set (num_source × dim).
     * @param targetEmbedding Target point set (num_target × dim).
     * @param k Number of nearest neighbors to query.
     * @return Pair of (indices of nearest neighbors, squared distances).
     */
    std::pair<std::vector<size_t>, std::vector<double>> NearestNeighborSearch(const Eigen::MatrixXd& sourceEmdedding, const Eigen::MatrixXd& targetEmbedding, int k=1);

    /**
     * @brief Converts an Eigen matrix into a dynamic point cloud.
     * Each row of the matrix is interpreted as a point in Euclidean space.
     * @tparam T Scalar type.
     * @param X Input matrix (num_points × dim).
     * @param cloud Output point cloud.
     */
    template <typename T>
    void createPointCloud(const Eigen::MatrixXd& X, PointCloudDynamic<T>& cloud);
}
#endif