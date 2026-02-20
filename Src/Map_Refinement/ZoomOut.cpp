#include "ZoomOut.h"

#include "../MeshProcessor.h"

using namespace ShapeAnalysis;

ZoomOut::ZoomOut(const Eigen::MatrixXd& fMap, const MeshProcessorSptr& source, const MeshProcessorSptr& target, const int numIter, const int stepSize, const bool adjointMap) :
                fMap_(fMap), sourceMeshSptr_(source), targetMeshSptr_(target), numIter_(numIter), stepSize_(stepSize), adjointMap_(adjointMap)
{}

ZoomOut::~ZoomOut()
{}

ZoomOutSptr ZoomOut::create(const Eigen::MatrixXd& fMap, const MeshProcessorSptr& source, const MeshProcessorSptr& target, const int numIter, const int stepSize, const bool adjointMap)
{
    std::shared_ptr<ZoomOut> obj = std::shared_ptr<ZoomOut>(new ZoomOut(fMap, source, target, numIter, stepSize, adjointMap));
    return obj;
}

Eigen::MatrixXd ZoomOut::refineZoomOut() const
{
    assert(fMap_.rows() == fMap_.cols() && "Must be square matrix");
    const int sourceEigenSpectrumSize  = sourceMeshSptr_->getEigenSpectrumSize();
    const int targetEigenSpectrumSize  = targetMeshSptr_->getEigenSpectrumSize();
    assert(sourceEigenSpectrumSize == targetEigenSpectrumSize && "Sizes must match");
    const int diff = sourceEigenSpectrumSize - fMap_.rows();
    const int availableEigenSpectrum = std::max(0, diff);
    const int iteration_range = ( numIter_ > 0 && numIter_ < availableEigenSpectrum ) ? numIter_ : availableEigenSpectrum;

    const bool usedFallback = iteration_range != numIter_;
    std::cout << std::endl;
    if (usedFallback)
    {
        std::cout << "\033[032m" <<"Requested iterations exceed available eigen spectrum. " << "Falling back to maximum allowed iterations: " << "\033[0m" << iteration_range << "\n";
    }
    else
    {
        std::cout << "\033[032m" << "Number of iterations selected for zoom-out refinement: " << "\033[0m" << iteration_range << "\n";
    }

    Eigen::MatrixXd temp_C = fMap_;
    for(int i=1; i<=iteration_range; ++i)
    {
        showProgress(i, iteration_range);
        const Eigen::MatrixXd updated_C = iterateZoomOut(temp_C, stepSize_);
        temp_C = updated_C;
    }
    std::cout << std::endl;
    std::cout << "\033[032m" << "Done !!" << "\033[0m" << std::endl;
    return temp_C;
}

Eigen::MatrixXd ZoomOut::iterateZoomOut(const Eigen::MatrixXd& FM, int stepSize) const
{
    assert(stepSize > 0 && "Step size should be valid");
    // int K2 = FM.rows();
    // int K1 = FM.cols();
    const int K = FM.rows();
    int new_K = K + stepSize;

    std::vector<size_t> indices;
    std::vector<double> distances;
    std::tie(indices, distances) = functionalMapToPointwise(FM, sourceMeshSptr_, targetMeshSptr_, adjointMap_);

    const Eigen::MatrixXd truncatedSourceEvecs = sourceMeshSptr_->getTruncatedEvec(new_K);
    const Eigen::MatrixXd truncatedTargetEvecs = targetMeshSptr_->getTruncatedEvec(new_K);
    Eigen::MatrixXd updated_C = pointwiseToFunctionalMap(indices, truncatedSourceEvecs, truncatedTargetEvecs, targetMeshSptr_->getMassMatrix(), SolverType::WeightedLeastSquaresSolver);
    return updated_C;
}