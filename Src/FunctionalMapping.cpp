#include "FunctionalMapping.h"
#include <optional>
#include "MeshProcessor.h"
#include "Descriptors.h"
#include "FunctionalMapEnergyEvaluator.h"
#include "FMoptimizer.h"
#include "Map_Refinement/ICP.h"
#include "Map_Refinement/ZoomOut.h"

using namespace ShapeAnalysis;

FunctionalMapping::FunctionalMapping(const MeshProcessorSptr& sourceMesh, const MeshProcessorSptr& targetMesh) : sourceMeshSptr(sourceMesh), targetMeshSptr(targetMesh)
{
    if(!sourceMesh || !targetMesh)
        throw std::invalid_argument("FunctionalMapping: mesh pointers must not be null");
}

FunctionalMappingSptr FunctionalMapping::create(const MeshProcessorSptr& sourceMesh, const MeshProcessorSptr& targetMesh)
{
    std::shared_ptr<FunctionalMapping> obj = std::shared_ptr<FunctionalMapping>(new FunctionalMapping(sourceMesh, targetMesh));
    return obj;
}

void FunctionalMapping::readLandMarkFile(const std::string& fileName, const unsigned expectedCount)
{
    std::cout << "\033[032m" << "Reading landmark file... " << "\033[0m" << std::endl;
    std::ifstream file(fileName);
    if(!file.is_open())
    {
        std::cerr << "Unable to open file " << fileName << "\n";
        throw std::runtime_error("Unable to open landmark file: " + fileName);
    }

    landMarkPoints.resize(expectedCount, 2);
    unsigned count = 0;
    std::string line;
    while(std::getline(file, line) && (count < expectedCount))
    {
        if(line.empty()) continue;

        std::istringstream lineStream(line);
        double x, y;
        if(lineStream >> x >> y)
        {
            landMarkPoints(count, 0) = x;
            landMarkPoints(count, 1) = y;
            count++;
        }
        else
        {
            std::cerr << "Warning: Could not prase line " << count + 1  << ": " << line << std::endl;
        }
    }
    if(count  < expectedCount)
    {
        std::cerr << "Warning: Only read " << count << "points, expected " << expectedCount << std::endl;
        landMarkPoints.conservativeResize(count, 2);
    }
    std::cout << "\033[032m" << "Finished reading landmark file... " << "\033[0m" << std::endl;
}

// Preprocessing pipeline:
// 1. Compute Laplace–Beltrami eigenvalues and eigenfunctions
// 2. Compute intrinsic descriptors (HKS / WKS)
// 3. Optionally augment descriptors using landmark-based features
// 4. Subsample descriptors to reduce redundancy
// 5. Normalize descriptors using the mesh mass matrix
void FunctionalMapping::preprocess(const PreProcessParameters& params)
{
    /*
     * Parameters
     * n_EV : std::pair<int, int>, ---> with the number of Laplacian eigenvalues to consider.
     * nDescr : int ---> number of descriptors to consider
     * descrType : std::string ---> "HKS" | "WKS"
     * landMarkPath : std::stiring ---> landmark file location path
     * subSample : int --->  step with which to subsample the descriptors.
     * kProcess : int --->  number of eigenvalues to compute for the Laplacian spectrum
     * spectrumPathMesh1: std::string ----> Pre computed mesh1 eigen values and vectors
     * spectrumPathMesh2: std::string ----> pre computed mesh2 eigen values and vectors
     */
    if(params.nDescr <=0)
    {
        throw std::invalid_argument("Number of descriptors must be positive");
    }

    K1 = params.n_EV;
    K2 = params.n_EV;
    params.kProcess ? kProcess = params.kProcess : kProcess = 200;

    if(!params.landMarkPath.empty())
    {
        readLandMarkFile(params.landMarkPath, params.landMarkPointsCount);
    }

    const int maxEV = std::max(K1.first, kProcess);
    sourceMeshSptr->process(maxEV, true, false, false, params.eigenSpectraPathMesh1);
    targetMeshSptr->process(maxEV, true, false, false, params.eigenSpectraPathMesh2);

    // computing Mesh Descriptors
    const DescriptorsSptr des = ShapeAnalysis::Descriptors::create();
    Eigen::MatrixXd descr1, descr2;
    Eigen::MatrixXd lmDescr1, lmDescr2;

    std::cout << std::endl;
    std::cout << "\033[32m" <<"Computing intrinsic shape descriptors..."  << "\033[0m" << std::endl;
    if(params.descrType == DescriptorType::HKS)
    {
        std::cout << std::endl;
        std::cout << "\033[32m" << "Computing the Heat Kernel Signature for both meshes." << "\033[0m" << std::endl;
        Eigen::VectorXd lm;
        descr1 = des->computeMeshHKS(sourceMeshSptr, params.nDescr, K1.first, lm, "Mesh 1", true);
        descr2 = des->computeMeshHKS(targetMeshSptr, params.nDescr, K2.first, lm, "Mesh 2", true);
        if(landMarkPoints.size() > 0)
        {
            std::cout << "\033[32m" << "Computing the Heat Kernel Signature from the given landmark vertices to all other vertices in the mesh." << "\033[0m" << std::endl;
            lm = landMarkPoints.col(0);
            lmDescr1 = des->computeMeshHKS(sourceMeshSptr, params.nDescr, K1.first, lm, "Mesh 1", true);

            lm = landMarkPoints.col(1);
            lmDescr2 = des->computeMeshHKS(targetMeshSptr, params.nDescr, K2.first, lm, "Mesh 2", true);
        }
    }
    else if(params.descrType == DescriptorType::WKS)
    {
        std::cout << std::endl;
        std::cout << "\033[32m" << "Computing the Wave Kernel Signature for both meshes." << "\033[0m" << std::endl;
        Eigen::VectorXd lm;
        descr1 = des->computeMeshWKS(sourceMeshSptr, params.nDescr, K1.first, lm, "Mesh 1", true);
        descr2 = des->computeMeshWKS(targetMeshSptr, params.nDescr, K2.first, lm, "Mesh 2", true);
        if(landMarkPoints.size() > 0)
        {
            std::cout << std::endl;
            std::cout << "\033[32m" << "Computing the Wave Kernel Signature from the given landmark vertices to all other vertices in the mesh." << "\033[0m" << std::endl;
            lm = landMarkPoints.col(0);
            lmDescr1 = des->computeMeshWKS(sourceMeshSptr, params.nDescr, K1.first, lm, "Mesh 1", true);

            lm = landMarkPoints.col(1);
            lmDescr2 = des->computeMeshWKS(targetMeshSptr, params.nDescr, K2.first, lm, "Mesh 2", true);
        }
    }
    std::cout << "\033[32m" << "Descriptor computation is complete." << "\033[0m" << std::endl;

    std::cout << std::endl;
    std::cout << "\033[32m" << "Combining global and local features and subsampling from the combined feature set." << "\033[0m" << std::endl;
    std::cout << std::endl;
    // Stack descriptors, (N, (descr1.cols()+lm_descr1.cols()))
    if(landMarkPoints.size()!=0)
    {
        descr1 = hStackDescriptors(descr1, lmDescr1);
        descr2 = hStackDescriptors(descr2, lmDescr2);

        std::cout << "Combined descriptor sizes: descriptor_1 " << "(" << descr1.rows() << " x " << descr1.cols() << ")" << std::endl;
        std::cout << "Combined descriptor sizes: descriptor_2 " << "(" << descr2.rows() << " x " << descr2.cols() << ")" << std::endl;
    }

    // sample both descriptors using given subsample steps.
    Eigen::MatrixXd subSampleDescr1 = subSampleDescriptors(descr1, params.subSampleStep);
    Eigen::MatrixXd subSampleDescr2 = subSampleDescriptors(descr2, params.subSampleStep);

    std::cout << std::endl;
    std::cout << "Subsampled descriptor sizes: descriptor_1 : " << "(" << subSampleDescr1.rows() << " x " << subSampleDescr1.cols() << ")" << std::endl;
    std::cout << "Subsampled descriptor sizes: descriptor_2 : " << "(" << subSampleDescr2.rows() << " x " << subSampleDescr2.cols() << ")" << std::endl;
    std::cout << "\033[32m" << "Feature Subsampling is complete." << "\033[0m" << std::endl;

    // Normalize descriptors with respect to the L2 inner product induced
    // by the mesh mass matrix to ensure scale invariance.
    Eigen::VectorXd norm1 = sourceMeshSptr->computeBilinearForm(subSampleDescr1, subSampleDescr1);
    Eigen::VectorXd norm2 = targetMeshSptr->computeBilinearForm(subSampleDescr2, subSampleDescr2);
    norm1 = norm1.array().sqrt();
    norm2 = norm2.array().sqrt();

    const double eps = 1e-12; // to avoid zero division
    sourceDescriptor = subSampleDescr1.array().rowwise() / (norm1.transpose().array() + eps);
    targetDescriptor = subSampleDescr2.array().rowwise() / (norm2.transpose().array() + eps);
    // for(int i=0; i<subSampleDescr2.cols(); i++)
    // {
    //     subSampleDescr2.col(i) /= norm2(i);
    // }
}

// Estimate the functional map C by minimizing a weighted energy composed of:
//  - Descriptor preservation
//  - Laplacian commutativity
//  - Descriptor commutativity (via multiplication operators)
//  - Optional orientation preservation
void FunctionalMapping::fit(const FitParameters& params)
{
    // Projection Φᵀ·(M·f): Computes Fourier coefficients of descriptor f in the eigenbasis
    std::cout << std::endl;
    std::cout << "\033[032m" << "Projection Φᵀ·(M·f): Computes the Fourier coefficients of the descriptor f on the source and target meshes in their respective eigenbases." << "\033[0m" << std::endl;
    const Eigen::MatrixXd descr1_reduced = project(sourceDescriptor, MeshID::Mesh1, K1.first);
    const Eigen::MatrixXd descr2_reduced = project(targetDescriptor, MeshID::Mesh2, K2.first);
    std::cout << std::endl;

    std::cout << "Reduced descriptor_1 : " << "(" <<descr1_reduced.rows() << " x " << descr1_reduced.cols()  << ")" << std::endl;
    std::cout << "Reduced descriptor_2 : " << "(" << descr2_reduced.rows() << " x " << descr2_reduced.cols() << ")" << std::endl;

    std::cout << "\033[32m" << "Done !!" << "\033[0m" << std::endl;
    //Compute multiplicative operators associated to each descriptor
    std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> descriptorOperatorPairs;
    if(params.w_dcomm > 0.0)
    {
        descriptorOperatorPairs = computeDescrOperator();
    }

    std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> orientationOperatorPairs;
    if(params.w_orient > 0.0)
    {
        orientationOperatorPairs = computeOrientationOperator(false, false);
    }

    // perform pairwise squared difference matrix between the eigenvalues of two meshes.
    // Construct matrix D where D(i,j) = (λ₂ᵢ - λ₁ⱼ)². This penalizes deviation from Laplacian commutativity.
    Eigen::MatrixXd ev_sqdiff = computeEigenvalueSqDiff();

    // normalizing the eigenvalue squared difference matrix by dividing each element by the sum of all elements in the matrix.
    const double total_sum = ev_sqdiff.sum();
    ev_sqdiff.array() /= total_sum;

    // Optimize
    OptimizationParameters optimParams;
    optimParams.wDescr = params.w_descr;
    optimParams.wLap = params.w_lap;
    optimParams.wDescrComm = params.w_dcomm;
    optimParams.wOrient = params.w_orient;
    optimParams.descr1Reduced = descr1_reduced;
    optimParams.descr2Reduced = descr2_reduced;
    optimParams.descrOperatorPairs = descriptorOperatorPairs;
    optimParams.eigenvalueSqDiff = ev_sqdiff;
    optimParams.orienOperatorPairs = orientationOperatorPairs;

    // rescale orientation term
    if(params.w_orient)
    {
        optimParams.wOrient = 0.0;
        FunctionalMapEnergyEvaluatorSptr energy_obj = FunctionalMapEnergyEvaluator::create(optimParams);
        double eval_native = energy_obj->computeEnergy();

        Eigen::MatrixXd C = Eigen::MatrixXd::Identity(K1.first, K2.first);
        double eval_orient = energy_obj->operatorCommutation(C, orientationOperatorPairs);

        if (eval_orient > 1e-12)
        {
            double scale = params.w_orient * (eval_native / eval_orient);
            optimParams.wOrient = scale;
        }
    }

    const FunctionalMapEnergySptr optimizer = FunctionalMapEnergy::create(optimParams);
    optimizer->solve();
    FM = optimizer->getMatrixC();
    std::cout << "\033[032m" << "Optimized Functional Map Matrix : " << "\033[0m" << "(" << optimizer->getMatrixC().rows() << " x "  << optimizer->getMatrixC().cols() << ")" << std::endl;
}

std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> FunctionalMapping::computeOrientationOperator(bool reversing, bool normalize) const
{
    std::cout << std::endl;
    std::cout << "\033[032m" << "Computing orientation operator" << "\033[0m" << std::endl;
    assert(sourceDescriptor.cols() == targetDescriptor.cols() && "Both should have the same size");
    const int num_cols = sourceDescriptor.cols();
    std::vector<std::vector<Eigen::MatrixXd>> grads_source(num_cols);
    std::vector<std::vector<Eigen::MatrixXd>> grads_target(num_cols);

    std::cout << std::endl;
    std::cout << "\033[036m" << "Computing gradient per face: descriptor function with " << "\033[0m" << sourceDescriptor.cols() << "\033[036m" << " feature dimensions on source and target meshes." << "\033[0m" << std::endl;

    // This method computes face quantities once for the given mesh and can be used repeatedly.
    // It must be called before gradient computation and orientation operator computation,
    // as the populated quantities are used by these two methods.
    sourceMeshSptr->computeBarycentricBasis();
    targetMeshSptr->computeBarycentricBasis();

    for(int i=0; i<num_cols; i++)
    {
        grads_source[i] = sourceMeshSptr->computeGradient(sourceDescriptor.col(i), false, false);
        grads_target[i] = targetMeshSptr->computeGradient(targetDescriptor.col(i), false, false);
    }

    const Eigen::MatrixXd truncatedSourceEvecs = sourceMeshSptr->getTruncatedEvec(K1.first);
    const Eigen::MatrixXd truncatedTargetEvecs = targetMeshSptr->getTruncatedEvec(K2.first);

    //const Eigen::MatrixXd inverse_source = truncatedSourceEvecs.transpose() * sourceMeshSptr->getMassMatrix(); // (K x num_vertices)
    //const Eigen::MatrixXd inverse_target = truncatedTargetEvecs.transpose() * targetMeshSptr->getMassMatrix();

    // Faster version
    const Eigen::VectorXd m_src = sourceMeshSptr->getMassMatrix().diagonal();
    const Eigen::VectorXd m_trg = targetMeshSptr->getMassMatrix().diagonal();
    const Eigen::MatrixXd inverse_source = truncatedSourceEvecs.transpose().array().rowwise() * m_src.transpose().array(); // (K x num_vertices)
    const Eigen::MatrixXd inverse_target = truncatedTargetEvecs.transpose().array().rowwise() * m_trg.transpose().array();

    std::cout << std::endl;
    std::cout << "\033[036m" << "Computing reduced operators: "  << "\033[0m" << grads_source.size() << " gradient fields" << std::endl;
    // Compute the operators in reduced basis
    std::vector<Eigen::MatrixXd> operator1(grads_source.size());
    for(int i=0; i<grads_source.size(); i++)
    {
        // (K x num_vertices) * (num_vertices x num_vertices) * (num_vertices x K) ----> (K x K)
        // Projecting orientation operator to reduced basis...
         operator1[i] =  inverse_source * sourceMeshSptr->computeOrientationOperator(grads_source[i], false) * truncatedSourceEvecs;
    }

    std::vector<Eigen::MatrixXd> operator2(grads_target.size());
    if(reversing)
    {
        for(int i=0; i<grads_target.size(); i++)
        {
            // (K x num_vertices) * (num_vertices x num_vertices) * (num_vertices x K) ----> (K x K)
            // Projecting orientation operator to reduced basis...
           operator2[i] = -inverse_target * targetMeshSptr->computeOrientationOperator(grads_target[i], false) * truncatedTargetEvecs;
        }
    }
    else
    {
        for(int i=0; i<grads_target.size(); i++)
        {
            // (K x num_vertices) * (num_vertices x num_vertices) * (num_vertices x K) ----> (K x K)
            // Projecting orientation operator to reduced basis...
             operator2[i] = inverse_target * targetMeshSptr->computeOrientationOperator(grads_target[i], false) * truncatedTargetEvecs;
        }
    }

    assert(operator1.size() == operator2.size() && "Operator vectors must have same size");

    std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> paired_operators;
    paired_operators.reserve(operator1.size());
    for(int i=0; i<operator1.size(); i++)
    {
        paired_operators.emplace_back(operator1[i], operator2[i]);
    }
    std::cout << "\033[032m" << "Orientation operator: Computation completed." << "\033[0m" << std::endl;
    return paired_operators;
}

std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> FunctionalMapping::computeDescrOperator() const
{
    std::cout << std::endl;
    std::cout << "\033[032m" << "Compute the multiplication operators associated with the descriptors." << "\033[0m" << std::endl;
    // Compute the multiplication operators associated with the descriptors.
    // This gives us how each descriptor looks in the reduced eigenbasis.
    /*
        C1_i = Φ₁ᵀ·M₁·(f_i ⊙ Φ₁)
        C2_i = Φ₂ᵀ·M₂·(g_i ⊙ Φ₂)
        ⊙ is element-wise multiplication (Hadamard product)
        Φ₁ = truncated eigenvectors of mesh1 (first K₁)
        Φ₂ = truncated eigenvectors of mesh2 (first K₂)
        M₁, M₂ = mass matrices (for proper integration on surfaces)
     */

    // Extract the truncated eigen functions from both meshes
    const Eigen::MatrixXd Phi1 = sourceMeshSptr->getTruncatedEvec(K1.first); // (N, K)
    const Eigen::MatrixXd Phi2 = targetMeshSptr->getTruncatedEvec(K2.first); // (N, K)

    // Use sparse matrices if available
    const Eigen::SparseMatrix<double>& M1 = sourceMeshSptr->getMassMatrix(); // (N1, N1)
    const Eigen::SparseMatrix<double>& M2 = targetMeshSptr->getMassMatrix(); // (N2, N2)

    // Compute pseudo-inverses
    // const Eigen::MatrixXd pinv1 = Phi1.transpose() * M1; // (N, K)^T * (N,N) ---> (K,N)
    // const Eigen::MatrixXd pinv2 = Phi2.transpose() * M2; // // (N, K)^T * (N,N) ---> (K,N)
    // faster version
    const Eigen::VectorXd m_source = M1.diagonal();
    const Eigen::VectorXd m_target = M2.diagonal();
    const Eigen::MatrixXd pinv1 = Phi1.transpose().array().rowwise() * m_source.transpose().array(); // (N, K)^T * (N,N) ---> (K,N)
    const Eigen::MatrixXd pinv2 = Phi2.transpose().array().rowwise() * m_target.transpose().array(); // // (N, K)^T * (N,N) ---> (K,N)

    const int num_descr = static_cast<int>(sourceDescriptor.cols());
    std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> list_descr;
    list_descr.reserve(num_descr);
    for(int i=0; i<num_descr; i++)
    {
        // Compute: descriptor_i * eigenvectors (element-wise multiplication)
        Eigen::MatrixXd term1 = Phi1.array().colwise() * sourceDescriptor.col(i).array(); // (N, K) * (N) ---> (N, k)
        Eigen::MatrixXd term2 = Phi2.array().colwise() * targetDescriptor.col(i).array(); // (N, K) * (N) ---> (N, k)

        // Multiply by pseudo-inverse
        // Project onto basis: Φᵀ·M·(descriptor ⊙ Φ)
        Eigen::MatrixXd C1_i  = pinv1 * term1; // (K, N) * (N, K) ---> (K, K)
        Eigen::MatrixXd C2_i  = pinv2 * term2;

        list_descr.emplace_back(C1_i , C2_i);
    }
    std::cout << "\033[032m" << "Done !!" << "\033[0m" << std::endl;
    return list_descr;
}

Eigen::MatrixXd FunctionalMapping::computeEigenvalueSqDiff() const
{
    const Eigen::VectorXd ev1 = sourceMeshSptr->getTruncatedEval(K1.first);
    const Eigen::VectorXd ev2 = targetMeshSptr->getTruncatedEval(K2.first);

    // Check they're the same size (as in Python code)
    if(K1.first != K2.first)
        throw std::runtime_error("K1 and K2 must be equal for this computation");

    const int K = K1.first;  // Should be 35
    Eigen::MatrixXd diff(K, K);

    // Fill matrix: diff(i,j) = ev2[i] - ev1[j]
    for(int i=0; i<K; i++)
    {
        for(int j=0; j<K; j++)
        {
            diff(i, j) = ev2(i) - ev1(j);
        }
    }
    // square elementwise
    Eigen::MatrixXd ev_diff = diff.array().square();
    return ev_diff;
}

Eigen::MatrixXd FunctionalMapping::project(const Eigen::MatrixXd& descr, const MeshID mesh, const std::optional<int> K) const
{
    int tempK;
    Eigen::MatrixXd reduced_descr;

    if(K.has_value())
    {
        tempK = K.value();
    }
    else
    {
        tempK = (mesh == MeshID::Mesh1) ? K1.first : K2.first;
    }

    switch(mesh)
    {
        case MeshID::Mesh1:
            reduced_descr = sourceMeshSptr->project(descr, tempK);
            break;
        case MeshID::Mesh2:
            reduced_descr = targetMeshSptr->project(descr, tempK);
            break;
        default:
            throw std::invalid_argument("Invalid mesh index");
            break;
    }
    return reduced_descr;
}

std::pair<std::vector<size_t>, std::vector<double>> FunctionalMapping::computePointToPoint() const
{
    return functionalMapToPointwise(FM, sourceMeshSptr, targetMeshSptr, false, true);
}

std::pair<std::vector<size_t>, std::vector<double>> FunctionalMapping::iterativeClosestPointRefinement()
{
    return iterativeClosestPointRefinement(75, 1e-10, false);
}

std::pair<std::vector<size_t>, std::vector<double>> FunctionalMapping::iterativeClosestPointRefinement(const int numIter, const double tolerance, const bool adjointMap)
{
    std::cout << std::endl;
    std::cout << "\033[032m" << "Performing Iterative Closest Point (ICP) to refine the functional map, " "enforcing orthogonality (C C^T = I) and establishing accurate point-to-point "
                 "correspondences from the target to the source shape." << "\033[0m" << std::endl;

    IterativeClosestPointSptr icp = IterativeClosestPoint::create(FM, sourceMeshSptr, targetMeshSptr, numIter, tolerance, adjointMap);
    FM_ICP =  icp->refineICP();
    std::vector<size_t> indices;
    std::vector<double> distances;
    std::tie(indices, distances) = functionalMapToPointwise(FM_ICP, sourceMeshSptr, targetMeshSptr, false, true);
    //std::cout << "\033[032m" << "Done !!" << "\033[0m" << std::endl;
    return {indices, distances};
}

std::pair<std::vector<size_t>, std::vector<double> > FunctionalMapping::zoomOutRefinement()
{
    return zoomOutRefinement(25, 1);
}

std::pair<std::vector<size_t>, std::vector<double>> FunctionalMapping::zoomOutRefinement(const int numIter, const int stepSize)
{
    ZoomOutSptr zo_obj = ZoomOut::create(FM, sourceMeshSptr, targetMeshSptr, numIter, stepSize, false);
    FM_ZoomOut = zo_obj->refineZoomOut();

    std::cout << std::endl;
    std::cout << "\033[32m" << "Initial functional map matrix size:\033[0m " << "(" << FM.rows() << " x " << FM.cols() << ")\n";
    std::cout << "\033[32m" << "After " << "\033[0m"  << numIter << "\033[032m" << " zoom-out refinement iterations, size:\033[0m " << "(" << FM_ZoomOut.rows() << " x " << FM_ZoomOut.cols() << ")\n";

    std::vector<size_t> indices;
    std::vector<double> distances;
    std::tie(indices, distances) = functionalMapToPointwise(FM_ZoomOut, sourceMeshSptr, targetMeshSptr, false, true);
    //std::cout << "\033[032m" << "Done !!" << "\033[0m" << std::endl;
    return {indices, distances};
}