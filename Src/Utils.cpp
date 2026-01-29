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

    std::pair<std::vector<size_t>, std::vector<double>> meshToP2P(const Eigen::MatrixXd& FM_12, const MeshProcessorSptr& mesh1, const MeshProcessorSptr& mesh2, const bool adjointMap)
    {
        std::cout << std::endl;
        std::cout << "\033[032m" << "Establishing point-to-point correspondence using nearest neighbor search in the spectral domain. Note that point-to-point correspondence is "
                     "established from shape 2 to shape 1." << "\033[0m" << std::endl;

        assert(FM_12.cols() == FM_12.rows() && "Should be square matrix");
        int dim  = FM_12.cols();
        Eigen::MatrixXd embedded1;
        Eigen::MatrixXd embedded2;
        if(adjointMap)
        {
            embedded1 = mesh1->getTruncatedEvec(dim);
            embedded2 = (mesh2->getTruncatedEvec(dim) * FM_12);
        }
        else
        {
            // Φ1(i,:) ∈ ℝ¹ˣ³⁵   row vector of coefficients so C^T is required
            embedded1 = (mesh1->getTruncatedEvec(dim) * FM_12.transpose());
            embedded2 = mesh2->getTruncatedEvec(dim);
        }
        return NearestNeighborSearch(embedded1, embedded2, 1);
    }

    std::pair<std::vector<size_t>, std::vector<double>> NearestNeighborSearch(const Eigen::MatrixXd& sourceEmdedding, const Eigen::MatrixXd& targetEmbedding, const int k)
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
}

