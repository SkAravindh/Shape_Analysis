#include "ICP.h"
#include "../MeshProcessor.h"

using namespace ShapeAnalysis;

IterativeClosestPoint::IterativeClosestPoint(const Eigen::MatrixXd& fMap, const MeshProcessorSptr& sourceMesh, const MeshProcessorSptr& targetMesh, const int numIter, const double tolerance, const bool adjointMap):
        fMap_(fMap), sourceMeshSptr_(sourceMesh), targetMeshSptr_(targetMesh), numIter_(numIter), tolerance_(tolerance), adjointMap_(adjointMap)
{
}

IterativeClosestPoint::~IterativeClosestPoint()
{}

IterativeClosestPointSptr IterativeClosestPoint::create(const Eigen::MatrixXd& fMap, const MeshProcessorSptr& sourceMesh, const MeshProcessorSptr& targetMesh, const int numIter, const double tolerance, const bool adjointMap)
{
    std::shared_ptr<IterativeClosestPoint> obj = std::shared_ptr<IterativeClosestPoint>(new IterativeClosestPoint(fMap, sourceMesh, targetMesh, numIter, tolerance, adjointMap));
    return obj;
}

Eigen::MatrixXd IterativeClosestPoint::refineICP() const
{
    assert(fMap_.rows() == fMap_.cols() && "Currently, Functional map matrix expected to be square");
    // const int K2 = fMap_.rows();
    // const int K1 = fMap_.cols();
    const int K = fMap_.rows();
    const Eigen::MatrixXd sourceEvec = sourceMeshSptr_->getTruncatedEvec(K);
    const Eigen::MatrixXd targetEvec = targetMeshSptr_->getTruncatedEvec(K);

    int iteration_range = numIter_ > 0 ? numIter_ : 500;
    Eigen::MatrixXd temp_C = fMap_;
    std::cout << std::endl;
    std::cout << "\033[032m" << "Number of iterations chosen for ICP refinement: " << "\033[0m" << iteration_range << " ";
    std::cout << "\033[032m" << "Stopping criterion: " << "\033[0m" << tolerance_ << std::endl;

    for(int i=0; i<=iteration_range; ++i)
    {
        showProgress(i, iteration_range);
        const Eigen::MatrixXd updated_C = iterationICP(temp_C, sourceEvec, targetEvec);
        const double max_co_eff = (updated_C - temp_C).cwiseAbs().maxCoeff();
        if(max_co_eff <= tolerance_)
        {
            std::cout << std::endl << std::endl;
            std::cout << "\033[1m" << "ICP iterations stopped: convergence threshold reached." << "\033[0m" << std::endl;
            break;
        }
        temp_C = updated_C;
    }
    std::cout << std::endl;
    std::cout << "\033[032m" << "Done !!" << "\033[0m" << std::endl;
    return temp_C;
}

Eigen::MatrixXd IterativeClosestPoint::iterationICP(const Eigen::MatrixXd& FM, const Eigen::MatrixXd& truncatedSourceEvecs, const Eigen::MatrixXd& truncatedTargetEvecs) const
{
    std::vector<size_t> indices;
    std::vector<double> distances;
    std::tie(indices, distances) = functionalMapToPointwise(FM, sourceMeshSptr_, targetMeshSptr_, adjointMap_);
    Eigen::MatrixXd updated_C = pointwiseToFunctionalMap(indices, truncatedSourceEvecs, truncatedTargetEvecs, targetMeshSptr_->getMassMatrix());

    /*
     * since FM is not orthogonal, we deliberately make it orthogonal using SVD we use U and V because they are responsible for orthogonality we ignore shear part.
     */
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(updated_C, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // Eigen::MatrixXd identityMatrix(targetEvecs.cols(), sourceEvecs.cols());
    // identityMatrix.setIdentity();
    Eigen::MatrixXd orthogonal_C = svd.matrixU() * svd.matrixV().transpose();
    return orthogonal_C;
}

