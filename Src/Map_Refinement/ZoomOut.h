/**
 * @file Shape_Analysis/src/Map_Refinement/ZoomOut.h
 * @author Aravindhkumar Samathur Kalimuthu
 * @date   2026-Feb-18
 */

#ifndef ZOOMOUT_H
#define ZOOMOUT_H
#include "ptrMapRefinement.h"
#include "../Utils.h"

namespace ShapeAnalysis
{
    /**
     * @brief Spectral ZoomOut refinement for functional maps
     * Implements the ZoomOut functional map refinement algorithm from "ZoomOut: Spectral Upsampling for Efficient Shape Correspondence".
     * The method improves an initial low-dimensional functional map by progressively increasing the spectral basis size and recomputing
     * the map in closed form using pointwise correspondences.
     *
     * Typical workflow:
     *      - Compute initial functional map C0 using descriptors.
     *      - Run ZoomOut refinement.
     *      - Use refined map for final correspondence.
     */
    class ZoomOut
    {
    public:

        /**
         * Class destructor.
         */
        ~ ZoomOut();

        /**
         * @brief Factory constructor.
         * @param fMap Initial functional map C₀ (k×k).
         * @param source Source mesh processor.
         * @param target Target mesh processor.
         * @param numIter Number of zoom-out refinement iterations. Each iteration increases spectral dimension.
         * @param stepSize Number of eigenfunctions added per iteration.
         *          Example:
         *              - stepSize = 1  →  35-->36-->37-->...
         *              - stepSize = 5  →  35-->40-->45-->...
         * @param adjointMap If true, uses adjoint functional map during pointwise recovery.
         * @return shared pointer to ZoomOut instance.
         */
        static ZoomOutSptr create(const Eigen::MatrixXd& fMap, const MeshProcessorSptr& source, const MeshProcessorSptr& target, int numIter, int stepSize, bool adjointMap);

        /**
         * @brief Run full ZoomOut refinement.
         * Performs:
         *      for i = 1 … numIter:
         *           C <--- iterateZoomOut(C, stepSize)
         * @return Refined functional map of larger dimension.
         */
        Eigen::MatrixXd refineZoomOut() const;
    private:
        /**
         * Class constructor
         */
        ZoomOut(const Eigen::MatrixXd& fMap, const MeshProcessorSptr& source, const MeshProcessorSptr& target, int numIter, int stepSize, bool adjointMap);

        /**
         * Perform one ZoomOut iteration.
         * Convert current functional map to pointwise indices using spectral nearest neighbors.
         * Increase basis size:
         *    k_new = k_old + stepSize
         * Recompute functional map in closed form:
         *      -  C_new = (Φ_M^{k_new})^+ Π Φ_N^{k_new}.
         *
         * @param FM  Current functional map (k×k).
         * @param stepSize  Spectral increment.
         * @return Updated functional map (k+stepSize × k+stepSize).
         */
        Eigen::MatrixXd iterateZoomOut(const Eigen::MatrixXd& FM, int stepSize) const;
    private:
        // Initial functional map C₀.
        Eigen::MatrixXd fMap_;

        // Source mesh containing eigenbasis Φ_N.
        MeshProcessorSptr sourceMeshSptr_;

        // Target mesh containing eigenbasis Φ_M.
        MeshProcessorSptr targetMeshSptr_;

        // Number of refinement iterations.
        int numIter_;

        // Spectral increment per iteration.
        int stepSize_;

        // Use adjoint map during pointwise conversion.
        bool adjointMap_;
    };
}

#endif