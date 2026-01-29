#include <iostream>
#include "FunctionalMapping.h"
#include "MeshProcessor.h"

int main(int argc, char** argv)
{
    const std::string fileName1 = "../../data/cat-00.off";
    const std::string fileName2 = "../../data/lion-00.off";
    const std::string landMarkPoints = "../../data/landmarks.txt";

    const std::string spectrumPathMesh1 = "../../data/spectrum/cat";
    const std::string spectrumPathMesh2 = "../../data/spectrum/lion";

    assert(ShapeAnalysis::isFileExist(fileName1) && "File Should Exist");
    assert(ShapeAnalysis::isFileExist(fileName2) && "File Should Exist");

    /*
     * Load source and target meshes.
     */
    const ShapeAnalysis::MeshProcessorSptr mesh1 = ShapeAnalysis::MeshProcessor::create(fileName1);
    const ShapeAnalysis::MeshProcessorSptr mesh2 = ShapeAnalysis::MeshProcessor::create(fileName2);

    /*
     * Initialize the functional mapping pipeline.
     */
    const ShapeAnalysis::FunctionalMappingSptr funMap = ShapeAnalysis::FunctionalMapping::create(mesh1, mesh2);

    /*
    * Preprocessing configuration.
    * - Compute Laplace–Beltrami spectra.
    * - Compute intrinsic descriptors (WKS).
    * - Optionally augment descriptors with landmark-based features.
    * - Subsample descriptors to reduce redundancy.
    */
    ShapeAnalysis::PreProcessParameters config;
    config.n_EV = std::make_pair(35, 35);
    config.nDescr = 100;
    config.descrType = ShapeAnalysis::DescriptorType::WKS;
    config.eigenSpectraPathMesh1 = spectrumPathMesh1;
    config.eigenSpectraPathMesh2 = spectrumPathMesh2;
    config.landMarkPath = landMarkPoints;
    config.landMarkPointsCount = 5;
    config.subSampleStep = 5;
    config.kProcess = 0;
    config.debug = false;

    /*
     * Run preprocessing pipeline
     */
    funMap->preprocess(config);

    /*
     * Functional map optimization parameters.
     * - Descriptor preservation.
     * - Laplacian commutativity.
     * - Descriptor commutativity (multiplication operators).
     */
    ShapeAnalysis::FitParameters fitParams;
    fitParams.w_descr = 1e0;
    fitParams.w_lap = 1e-2;
    fitParams.w_dcomm = 1e-1;
    fitParams.w_orient = 0;
    funMap->fit(fitParams);

    /*
     * Recover point-to-point correspondence (Mesh2 → Mesh1).
     */
    std::vector<size_t> indices;
    std::vector<double> distances;
    std::tie(indices, distances) =  funMap->computePointToPoint();
}