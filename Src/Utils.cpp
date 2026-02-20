#include "Utils.h"

#include <nanoflann.hpp>

#include "MeshProcessor.h"
#include "polyscope/point_cloud.h"

namespace ShapeAnalysis
{
    bool isFileExist(const std::string& fileName)
    {
        std::filesystem::path p(fileName);
        if(std::filesystem::exists(p))
        {
            return true;
        }
        return false;
    }

    Eigen::MatrixXd hStackDescriptors(const Eigen::MatrixXd& descr1, const Eigen::MatrixXd& descr2)
    {
        Eigen::MatrixXd hStack = descr1;
        hStack.conservativeResize(Eigen::NoChange, descr1.cols() + descr2.cols());
        hStack.rightCols(descr2.cols()) = descr2;
        assert(hStack.rows() == descr1.rows() && hStack.cols() == (descr1.cols()+descr2.cols()));
        return hStack;
    }

    Eigen::MatrixXd subSampleDescriptors(const Eigen::MatrixXd& descr, const int sampleStep)
    {
        if(sampleStep <= 1)
        {
            return descr;
        }

        const int newCols = (static_cast<int>(descr.cols()) + sampleStep - 1) / sampleStep;
        Eigen::MatrixXd descriptor(descr.rows(), newCols);
        for(int i=0; i<newCols; i++)
        {
            const int colIndex = i * sampleStep;
            if(colIndex < descr.cols())
            {
                descriptor.col(i) = descr.col(colIndex);
            }
            else
            {
                descriptor.col(i).setZero();
            }

        }
        return descriptor;
    }

    Eigen::MatrixXd pointwiseToFunctionalMap(const std::vector<size_t>& indices, const Eigen::MatrixXd& truncatedSourceEvecs, const Eigen::MatrixXd& truncatedTargetEvecs, const Eigen::SparseMatrix<double>& targetShapeMassMatrix, const SolverType solType)
    {
        Eigen::MatrixXd sourceEvecsUsingIndices(indices.size(), truncatedSourceEvecs.cols());
        Eigen::MatrixXd updatedFunctionalMap(truncatedTargetEvecs.cols(), truncatedSourceEvecs.cols());
        // for(int i=0; i<indices.size(); i++)
        // {
        //     sourceEvecsUsingIndices.row(i) = sourceEvecs.row(indices[i]);
        // }
        sourceEvecsUsingIndices = truncatedSourceEvecs(indices, Eigen::all);

        bool isWeightedLeastSquares = solType == SolverType::WeightedLeastSquaresSolver;

        if(isWeightedLeastSquares)
        {
            assert(targetShapeMassMatrix.rows() == truncatedTargetEvecs.rows() && "Dimension should match");
            // (n2 x k2)^T @ ( (n2 x n2) @ (n2 x k1)  ) ---- > (k2 x k1)
            // C=Φ_target^T @ A2 @ Φ_source^(from p2 to p1)
            //weighted least squares problem
            //updatedFunctionalMap.noalias() = targetEvecs.transpose() * (targetShapeArea * sourceEvecsUsingIndices); //(k2 x k1)
            const Eigen::VectorXd mass_vec = targetShapeMassMatrix.diagonal();
            const Eigen::MatrixXd weighted = (sourceEvecsUsingIndices.array().colwise() * mass_vec.array());
            updatedFunctionalMap.noalias() = truncatedTargetEvecs.transpose() * weighted;
        }
        else
        {
            //QR
            updatedFunctionalMap.noalias() = truncatedTargetEvecs.colPivHouseholderQr().solve(sourceEvecsUsingIndices);
        }
        return updatedFunctionalMap;
    }

    //meshToP2P
    std::pair<std::vector<size_t>, std::vector<double>> functionalMapToPointwise(const Eigen::MatrixXd& FM_12, const MeshProcessorSptr& mesh1, const MeshProcessorSptr& mesh2, const bool adjointMap, const bool info)
    {
        if(info)
        {
            std::cout << std::endl;
            std::cout << "\033[032m" << "Establishing point-to-point correspondence using nearest neighbor search in the spectral domain. Note that point-to-point correspondence is "
                         "established from shape 2 to shape 1." << "\033[0m" << std::endl;
        }
        assert(FM_12.cols() == FM_12.rows() && "Should be square matrix");

        // Needed, if we are working with non-square functional map matrix.
        // const int k2 = FM_12.rows();
        // const int k1 = FM_12.cols();

        int dim  = FM_12.cols();
        Eigen::MatrixXd embedded1;
        Eigen::MatrixXd embedded2;
        if(adjointMap)
        {
            embedded1 = mesh1->getTruncatedEvec(dim); // we should use k1 instead of dim, if FM_12 is not a square matrix.
            embedded2 = (mesh2->getTruncatedEvec(dim) * FM_12); // likewise, here we should use k2, if FM_12 is not a square matrix.
        }
        else
        {
            // Φ1(i,:) ∈ ℝ¹ˣ³⁵   row vector of coefficients so C^T is required
            embedded1 = (mesh1->getTruncatedEvec(dim) * FM_12.transpose()); // here we should use k1, if FM_12 is not a square matrix.
            embedded2 = mesh2->getTruncatedEvec(dim); //here we should use k2, if FM_12 is not a square matrix.
        }
        return NearestNeighborSearch(embedded1, embedded2, 1, info);
    }

    std::pair<std::vector<size_t>, std::vector<double>> NearestNeighborSearch(const Eigen::MatrixXd& sourceEmdedding, const Eigen::MatrixXd& targetEmbedding, const int k, const bool info)
    {
        assert(sourceEmdedding.cols() == targetEmbedding.cols());
        int dim = static_cast<int>(sourceEmdedding.cols());

        // construct point cloud
        PointCloudDynamic<double> cloud;
        createPointCloud(sourceEmdedding, cloud);

        // construct a kd-tree index:
        using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloudDynamic<double>>, PointCloudDynamic<double>, -1>;

        my_kd_tree_t index(dim, cloud,  40);
        index.buildIndex();

        const int num_queries = static_cast<int>(targetEmbedding.rows());
        std::vector<size_t> result_indices(num_queries);
        std::vector<double> result_distance(num_queries);
        {
            // Query buffers (reused)
            if(info)
                std::cout << "\033[032m" << "Performing nearest neighbor search using a KD-tree. " << "Number of query points: " << "\033[0m" << num_queries << std::endl;
            std::vector<double> query(dim);
            std::vector<size_t> ret_index(k);
            std::vector<double> out_dis_sqr(k);
            for(size_t i=0; i<num_queries; i++)
            {
                Eigen::MatrixXd::ConstRowXpr vrt = targetEmbedding.row(i);

                for(int d=0; d<dim; d++)
                {
                    query[d] = vrt[d];
                }

                nanoflann::KNNResultSet<double> result_set(k);
                result_set.init(ret_index.data(), out_dis_sqr.data());

                index.findNeighbors(result_set, query.data(), nanoflann::SearchParams(10));

                //assert(ret_index.size() == k &&  ret_index.size() == out_dis_sqr.size());
                result_indices[i] = (ret_index[0]);
                result_distance[i] = (out_dis_sqr[0]);
            }
        }
        if(info)
            std::cout << "\033[032m" << "Done !!" << "\033[0m" << std::endl;
        return {result_indices, result_distance};
    }

    template <typename T>
    void createPointCloud(const Eigen::MatrixXd& X, PointCloudDynamic<T>& cloud)
    {
        int N = static_cast<int>(X.rows());
        cloud.dim  = static_cast<size_t>(X.cols());
        cloud.pts.resize(N * cloud.dim);

        for(int i=0; i<N; i++)
        {
            for(int j=0; j<cloud.dim; j++)
            {
                cloud.pts[(i * cloud.dim) + j] = X(i, j);
            }
        }
        //std::cout << "Total number of elements in the point cloud container :" << cloud.pts.size() << std::endl;
    }

    void showProgress(int current, int total)
    {
        const int width = 50;
        float progress = static_cast<float>(current) / total;

        // Round to nearest integer for both bar and percent
        int filled = static_cast<int>(width * progress + 0.5f);
        int percent = static_cast<int>(progress * 100 + 0.5f);

        // Clamp to valid range (just in case)
        if(filled > width)
            filled = width;

        if(percent > 100)
            percent = 100;

        std::cout << "\r[";   // carriage return to start of line
        for(int i = 0; i < width; ++i)
        {
            std::cout << (i < filled ? '#' : ' ');
        }
        std::cout << "] " << std::setw(3) << percent << '%' << std::flush;
    }

    bool isDiagonal(const Eigen::SparseMatrix<double>& mat)
    {
        // Iterate over the outer dimension (columns by default)
        for(int k = 0; k < mat.outerSize(); ++k)
        {
            // Iterate over the inner indices in the current column
            for(Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it)
            {
                if (it.row() != it.col())
                {
                    return false; // Found an off‑diagonal entry
                }
            }
        }
        return true;
    }
}

