/**
 * @file Shape_Analysis/src/Descriptor.h
 * @author Aravindhkumar Samathur Kalimuthu
 * @date   2026-Jan-01
 */

#ifndef DESCRIPTORS_H
#define DESCRIPTORS_H
#include "Utils.h"
namespace ShapeAnalysis
{
    /**
     * @class Descriptors
     * @brief Computes intrinsic spectral descriptors for triangular meshes.
     * This class provides implementations of:
     *  - Heat Kernel Signature (HKS).
     *  - Wave Kernel Signature (WKS).
     * Both descriptors can be computed:
     *  - Per-vertex (local descriptors).
     *  - From a set of landmark vertices to all other vertices (global descriptors).
     * The descriptors are computed using the Laplace–Beltrami eigenvalues and eigenfunctions provided by the MeshProcessor.
     */
    class Descriptors
    {
    public:
        /**
         * @brief Factory method to create a Descriptors instance.
         * @return Shared pointer to a Descriptors object.
         */
        static DescriptorsSptr create();

        /**
         * @brief Destructor.
         */
        ~Descriptors();

        /**
         * @brief Computes the Heat Kernel Signature (HKS) on a mesh.
         * This method computes HKS either:
         *  - Per vertex (if no landmarks are provided) or as landmark-based diffusion features (if landmarks are provided).
         * @param mesh Pointer to the mesh processor.
         * @param nDescr Number of time scales (descriptor dimensions).
         * @param K Number of Laplace–Beltrami eigenfunctions used.
         * @param landMark Optional landmark vertex indices.
         * @param scaled Whether to apply scale normalization.
         * @return HKS descriptor matrix of size (num_vertices × feature_dimension).
         */
        Eigen::MatrixXd computeMeshHKS(const MeshProcessorSptr& mesh, int nDescr, int K, const Eigen::VectorXd& landMark, const std::string& meshID, bool scaled);

        /**
         * @brief Computes the Wave Kernel Signature (WKS) on a mesh.
         * This method computes WKS either:
         *  - Per vertex (if no landmarks are provided), or as landmark-based diffusion features (if landmarks are provided).
         * @param mesh Pointer to the mesh processor.
         * @param nDescr Number of energy levels (descriptor dimensions).
         * @param K Number of Laplace–Beltrami eigenfunctions used.
         * @param landMark Optional landmark vertex indices.
         * @param scaled  scaled Whether to apply scale normalization.
         * @return WKS descriptor matrix of size (num_vertices × feature_dimension).
         */
        Eigen::MatrixXd computeMeshWKS(const MeshProcessorSptr& mesh, int nDescr, int K, const Eigen::VectorXd& landMark, const std::string& meshID, bool scaled);

    private:
        /**
         * @brief Constructor (use factory method).
         */
        Descriptors();
    private:
        /**
         * @brief Generates logarithmically spaced values.
         * This is used to sample diffusion times for HKS on a log scale, which better captures multi-scale geometric information.
         * @param start Minimum value.
         * @param end Maximum value.
         * @param numTime Number of samples.
         * @return  Vector of logarithmically spaced values.
         */
        Eigen::VectorXd geomSpace(double start, double end, int numTime) const;

        /**
         * @brief Computes the per-vertex Heat Kernel Signature (HKS).
         * @param eVals Laplace–Beltrami eigenvalues.
         * @param eVecs Laplace–Beltrami eigenvectors.
         * @param timeList Diffusion times.
         * @param scaled  Whether to normalize the descriptor.
         * @return HKS matrix (num_vertices × num_times).
         */
        Eigen::MatrixXd HKS(const Eigen::VectorXd& eVals, const Eigen::MatrixXd& eVecs, const Eigen::VectorXd& timeList, bool scaled) const;

        /**
         * @brief Computes the per-vertex Wave Kernel Signature (WKS).
         * @param eVals Laplace–Beltrami eigenvalues.
         * @param eVecs Laplace–Beltrami eigenvectors.
         * @param energyList Energy levels (log-scaled).
         * @param sigma Bandwidth parameter.
         * @param scaled Whether to normalize the descriptor.
         * @return WKS matrix (num_vertices × num_energies).
         */
        Eigen::MatrixXd WKS(const Eigen::VectorXd& eVals, const Eigen::MatrixXd& eVecs, const Eigen::VectorXd& energyList, double sigma, bool scaled) const;

        /**
         * @brief Computes landmark-based Heat Kernel Signatures.
         * Each landmark acts as a heat source, and diffusion responses to all vertices are concatenated.
         * @param eVals Laplace–Beltrami eigenvalues.
         * @param eVecs Laplace–Beltrami eigenvectors.
         * @param timeList Diffusion times.
         * @param landMark Landmark vertex indices.
         * @param scaled Whether to normalize the descriptor.
         * @return Landmark-based HKS matrix (num_vertices × (num_times × num_landmarks)).
         *          - The descriptor is constructed by concatenating diffusion responses for each landmark across all time scales.
         *          - Feature ordering: for each landmark p, all diffusion levels are stored contiguously.
         */
        Eigen::MatrixXd HKSLandMark(const Eigen::VectorXd& eVals, const Eigen::MatrixXd& eVecs, const Eigen::VectorXd& timeList, const Eigen::VectorXd& landMark,  bool scaled) const;

        /**
         * @brief Computes landmark-based Wave Kernel Signatures.
         * @param eVals Laplace–Beltrami eigenvalues.
         * @param eVecs Laplace–Beltrami eigenvectors.
         * @param energyList Energy levels.
         * @param landMark Landmark vertex indices.
         * @param sigma Bandwidth parameter.
         * @param scaled Whether to normalize the descriptor.
         * @return Landmark-based WKS matrix (num_vertices × (num_times × num_landmarks)).
         *          - The descriptor is constructed by concatenating wave diffusion responses for each landmark across all time scales.
         *          - Feature ordering: for each landmark p, all energy levels are stored contiguously.
         */
        Eigen::MatrixXd WKSLandMark(const Eigen::VectorXd& eVals, const Eigen::MatrixXd& eVecs, const Eigen::VectorXd& energyList, const Eigen::VectorXd& landMark, double sigma, bool scaled) const;

        /**
         * @brief Computes WKS spectral weighting coefficients.
         * These coefficients define how each eigenfunction contributes to a given energy level.
         * @param eVals Laplace–Beltrami eigenvalues.
         * @param energyList Energy levels.
         * @param sigma Bandwidth parameter.
         * @return Coefficient matrix (num_energies × K). K --> num_of_eigen_values
         */
        Eigen::MatrixXd wksCoefficients(const Eigen::VectorXd& eVals, const Eigen::VectorXd& energyList, double sigma) const;
    };
}

#endif