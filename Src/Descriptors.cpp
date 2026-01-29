#include "Descriptors.h"
#include "MeshProcessor.h"

using namespace ShapeAnalysis;

Descriptors::Descriptors()
{}

Descriptors::~Descriptors()
{}

DescriptorsSptr Descriptors::create()
{
    std::shared_ptr<Descriptors> obj = std::shared_ptr<Descriptors>(new Descriptors());
    return obj;
}

Eigen::VectorXd Descriptors::geomSpace(const double start, const double end, const int numTime) const
{
    // Generates logarithmically spaced diffusion times:
    // t_i = exp((1 - αlpha) * log(start) + αlpha * log(end)),  αlpha = i / (numTime - 1)

    assert(start > 0 && end > start && numTime > 1);
    Eigen::VectorXd result(numTime);
    const double log_start = std::log(start);
    const double log_end = std::log(end);
    for(int i=0; i<numTime; i++)
    {
        const double alpha = static_cast<double>(i) / (numTime - 1);
        result(i) = std::exp( ((1.0-alpha) * log_start) + (alpha * log_end)  );
    }
    return result;
}

/**
 * Computes the Heat Kernel Signature (HKS).

 * The HKS at vertex x and time t is defined as:
 *  - HKS(x, t) = Σ_{i=1}^K exp(-λ_i t) · φ_i(x)^2.

 * where λ_i and φ_i are the eigenvalues and eigenfunctions of the Laplace–Beltrami operator.

 * If scaled == true, a scale-invariant normalization is applied:
 *  - HKS_scaled(x, t) = HKS(x, t) / Σ_{i=1}^K exp(-λ_i t).
 */
Eigen::MatrixXd Descriptors::HKS(const Eigen::VectorXd& eVals, const Eigen::MatrixXd& eVecs, const Eigen::VectorXd& timeList, const bool scaled) const
{
    // Heat kernel coefficients: exp(-λ_i * t_j)
    const Eigen::MatrixXd co_eff = (-eVals * timeList.transpose()).array().exp().matrix();

    Eigen::MatrixXd naturalHKS =  eVecs.array().square().matrix() * co_eff;

    if(scaled)
    {
        // column-wise normalization
        Eigen::RowVectorXd colSum = co_eff.colwise().sum();
        Eigen::MatrixXd scaledHKS = naturalHKS.array().rowwise() * colSum.array().inverse();
        return scaledHKS;
    }
    return naturalHKS;
}

/**
 * Computes landmark-based HKS by treating each landmark as a heat source and measuring diffusion responses to all vertices over multiple time scales.
 * The landmark HKS is defined as:
 *  -  HKS_lm(x, p, t) = Σ_{i=1}^K exp(-λ_i t) · φ_i(x) · φ_i(p)

 * where λ_i and φ_i are the eigenvalues and eigenfunctions of the Laplace–Beltrami operator.
 * If scaled == true, a scale-invariant normalization is applied:
 *  - HKS_lm_scaled(x, p, t) = HKS_lm(x, p, t) / Σ_{i=1}^K exp(-λ_i t)
 *
 * The output matrix has size: (num_vertices × (num_times × num_landmarks)),
 * with features ordered such that all time steps for a landmark are stored contiguously.
 */
Eigen::MatrixXd Descriptors::HKSLandMark(const Eigen::VectorXd& eVals, const Eigen::MatrixXd& eVecs, const Eigen::VectorXd& timeList, const Eigen::VectorXd& landMark, const bool scaled) const
{
    Eigen::MatrixXd temp;
    Eigen::MatrixXd co_eff = (- timeList * eVals.transpose()).array().exp().matrix();
    //std::cout << co_eff.rows() << " " << co_eff.cols() << " " <<  eVal.rows() << std::endl;
    const int N = static_cast<int>(eVecs.rows()); // num of vertices
    const int K = static_cast<int>(eVecs.cols()); // num of eigen functions
    const int T = static_cast<int>(timeList.size()); // num time
    const int P = static_cast<int>(landMark.size()); // number of landmark points

    // Extract landMark eigen vectors (P, K).
    Eigen::MatrixXd landmark_evectors(P, K);
    for(int p=0; p<landMark.rows(); p++)
    {
        const double lm_index = landMark(p);
        landmark_evectors.row(p) =  eVecs.row(static_cast<Eigen::Index>(lm_index));
    }

    // Initialize flat matrix.
    Eigen::MatrixXd landmarks_HKS_flat = Eigen::MatrixXd::Zero(P*T, N);
    for(int t=0; t<T; t++)
    {
        // weighted eigen vectors for time t: (P, K).
        Eigen::MatrixXd weighted_t = landmark_evectors.array().rowwise() * co_eff.row(t).array();

        // multiply by transpose of eigenvectors to get (P, N).
        Eigen::MatrixXd result_t = weighted_t * eVecs.transpose();
        for(int p=0; p<P; p++)
        {
            const int row_idx = p * T + t;
           // std::cout << "row_idx: " << row_idx << std::endl;
            landmarks_HKS_flat.row(row_idx) = result_t.row(p);
        }
    }
    //std::cout << "landmarks_HKS_flat " << landmarks_HKS_flat.rows() << " " << landmarks_HKS_flat.cols() << std::endl;
    if(scaled)
    {
        Eigen::VectorXd inv_scaling = co_eff.rowwise().sum();
        for(int t=0; t< T; t++)
        {
            const double scale = 1.0 / inv_scaling(t);
            for(int p=0; p<P; p++)
            {
                // Store in flat matrix with ordering: for each landmark p, all T energy levels are contiguous
                // Row index = p * T + t  (p varies slow, t varies fast within each p-block)
                const int row_idx = p * T + t;
                landmarks_HKS_flat.row(row_idx) *= scale;
            }
        }
    }
    Eigen::MatrixXd final_result = landmarks_HKS_flat.transpose();
   // std::cout << "final_result " << final_result.rows() << " " << final_result.cols() << std::endl;
    return final_result;
}

Eigen::MatrixXd Descriptors::computeMeshHKS(const MeshProcessorSptr& mesh, const int nDescr, const int K, const Eigen::VectorXd& landMark, const std::string& meshID, const bool scaled)
{
    const Eigen::VectorXd truncatedEvals = mesh->getTruncatedEval(K);
    const Eigen::MatrixXd truncatedEvecs = mesh->getTruncatedEvec(K);
    Eigen::MatrixXd result;
    std::cout << meshID << ": " << "Eigenvalues " << "(" << truncatedEvals.rows() << " x " << truncatedEvals.cols() << ")" << " , ";
    std::cout << "Eigenvectors " << "(" << truncatedEvecs.rows() << " x " << truncatedEvecs.cols() << ")" << std::endl;
   //std::cout << "Truncated Eigen Values: " << "Rows " <<truncatedEvals.size() << " " << "Columns " << truncatedEvals.cols() << " , "  << "Truncated Eigen Vectors: " << "Rows " << truncatedEvecs.rows() << " "  << "Columns " << truncatedEvecs.cols() << std::endl;
    Eigen::VectorXd truncatedEvalAbs = truncatedEvals.cwiseAbs();
    for(int i=0; i<truncatedEvalAbs.size()-1; i++)
    {
        assert(truncatedEvalAbs(i) < truncatedEvalAbs(i+1));
    }
    const double start = (4.0 * std::log(10.0)) / truncatedEvalAbs(K-1);
    const double end = (4.0 * std::log(10.0)) / truncatedEvalAbs(1);
    const Eigen::VectorXd timeList = geomSpace(start, end, nDescr);

    if(landMark.size() == 0)
    {
        result = HKS(truncatedEvalAbs, truncatedEvecs, timeList, scaled);
    }
    else
    {
        result = HKSLandMark(truncatedEvalAbs, truncatedEvecs, timeList, landMark, scaled);
    }
    return result;
}

/**
 * Computes spectral weighting coefficients for the Wave Kernel Signature (WKS).
 * The coefficients are defined by a Gaussian window in log-eigenvalue space:
 *      - w_i(E) = exp( - (log(λ_i) - E)^2 / (2σ^2) )
 * where λ_i are the Laplace–Beltrami eigenvalues and E denotes an energy level.
 * @param eVals Laplace–Beltrami eigenvalues (λ)
 * @param energyList Energy levels at which the WKS is evaluated.
 * @param sigma standard deviation of the Gaussian window (σ > 0).
 * @return Matrix of size (num_energies × num_eigenvalues) containing the spectral weights used in WKS computation.
 */
Eigen::MatrixXd Descriptors::wksCoefficients(const Eigen::VectorXd& eVals, const Eigen::VectorXd& energyList, const double sigma) const
{
    // Compute WKS spectral weights using a Gaussian window in log-eigenvalue space
    assert(sigma > 0.0 && "sigma should be positive");
    // compute log of eigen value
    Eigen::VectorXd log_eval = eVals.array().abs().log();

    const int num_E = static_cast<int>(energyList.size());
    const int K = static_cast<int>(eVals.size());

    // compute diff matrix, slower but clear to understand i.e (e - log(lambda))
    // TODO: this loop version can be improved using eigen's internal methods.
    Eigen::MatrixXd diff(num_E, K);
    for(int i=0; i<num_E; i++)
    {
        for(int j=0; j<K; j++)
        {
            diff(i, j) = energyList(i) - log_eval(j);
        }
    }
    //std::cout << diff.rows() << " " << diff.cols() << "\n";
    const Eigen::MatrixXd squared_diff = diff.array().square().matrix();
    Eigen::MatrixXd co_effs = (-squared_diff.array() / (2.0 * sigma * sigma) ).exp();
    return co_effs;
}

/**
 * Computes the Wave Kernel Signature (WKS) at each vertex of the mesh.
 * The WKS is defined as:
 *      -  WKS(x, E) = Σ_i φ_i(x)^2 · exp( - (log(λ_i) − E)^2 / (2σ^2) )
 * where λ_i and φ_i are the Laplace–Beltrami eigenvalues and eigenfunctions, and E denotes an energy level.
 * Optionally, the descriptor is normalized per energy level:
 *      -  WKS(x, E) = WKS(x, E) / Σ_i C(E, i)
 * @param energyList Energy levels at which the WKS is evaluated.
 * @param sigma standard deviation of the Gaussian window (σ > 0).
 * @param scaled If true, applies per-energy normalization
 * @return WKS matrix of size (num_vertices × num_energy_levels)
 */
Eigen::MatrixXd Descriptors::WKS(const Eigen::VectorXd& eVals, const Eigen::MatrixXd& eVecs, const Eigen::VectorXd& energyList, const double sigma, bool scaled) const
{
    Eigen::MatrixXd co_effs = wksCoefficients(eVals, energyList, sigma);
   // std::cout << co_effs.rows() << " " << co_effs.cols() << "\n";
    Eigen::MatrixXd naturalWKS = eVecs.array().square().matrix() * co_effs.transpose();
    //std::cout << natural_wks.rows() << " " << natural_wks.cols() << "\n";
    if(scaled)
    {
        Eigen::RowVectorXd energy_scaling = co_effs.rowwise().sum().transpose().array().inverse();
        Eigen::MatrixXd scaledWKS = naturalWKS.array().rowwise() * energy_scaling.array();
        return scaledWKS;
    }
    return naturalWKS;
}

/**
 * Computes landmark-based Wave Kernel Signatures (WKS).
 * Each landmark vertex ℓ is treated as a spectral source, and the WKS response is evaluated at all vertices x over multiple energy levels.
 *
 * The landmark WKS is defined as:
 *      -  WKS_ℓ(x, E) = Σ_i φ_i(ℓ) φ_i(x) · exp( - (log(λ_i) − E)^2 / (2σ^2) )
 * Optionally, per-energy normalization is applied:
 *      -  WKS̃_ℓ(x, E) = WKS_ℓ(x, E) / Σ_i C(E, i)
 * The resulting matrix has size:
 *      - (num_vertices × (num_energy_levels × num_landmarks)).
 */
Eigen::MatrixXd Descriptors::WKSLandMark(const Eigen::VectorXd& eVals, const Eigen::MatrixXd& eVecs, const Eigen::VectorXd& energyList, const Eigen::VectorXd& landMark, const double sigma, const bool scaled) const
{
    Eigen::MatrixXd temp;
    Eigen::MatrixXd co_effs = wksCoefficients(eVals, energyList, sigma);
    const int num_vertices = static_cast<int>(eVecs.rows()); // number of vertices
    const int K = static_cast<int>(eVecs.cols()); // number of eigen vectors
    const int T = static_cast<int>(energyList.size()); // number of time step
    const int P = static_cast<int>(landMark.size()); // number of landmark points

    // Extract landMark eigen vectors (P, K)
    Eigen::MatrixXd landmark_evectors(P, K);
    for(int p=0; p<landMark.rows(); p++)
    {
        const double lm_index = landMark(p);
        landmark_evectors.row(p) = eVecs.row(static_cast<Eigen::Index>(lm_index));
    }

    // Intialize flat matrix
    Eigen::MatrixXd landmark_WKS_flat(P*T, num_vertices);
    for(int t=0; t<T; t++)
    {
        // weighted eigen vectors for time t: (P, K)
        Eigen::MatrixXd weight_t = landmark_evectors.array().rowwise() * co_effs.row(t).array();
        //std::cout << "weight_t " << weight_t.rows() << " " << weight_t.cols() << std::endl;

        // multiply by transpose of eigenvectors to get (P, N)
        Eigen::MatrixXd result_t = weight_t * eVecs.transpose();
        for(int p=0; p<P; p++)
        {
            const int row_index = p * T + t;
            landmark_WKS_flat.row(row_index) = result_t.row(p);
        }
    }

    if(scaled)
    {
        Eigen::VectorXd inv_scaling = co_effs.rowwise().sum();
        for(int t=0; t<T; t++)
        {
            const double scale = 1.0 / inv_scaling(t);
            for(int p=0; p<P; p++)
            {
                // Store in flat matrix with ordering: for each landmark p, all T energy levels are contiguous
                // Row index = p * T + t  (p varies slow, t varies fast within each p-block)
                const int row_index = p * T + t;
                landmark_WKS_flat.row(row_index) *= scale;
            }
        }
    }
    Eigen::MatrixXd final_result = landmark_WKS_flat.transpose();
    //std::cout << "final_result " << final_result.rows() << " " << final_result.cols() << std::endl;
    return final_result;
}

Eigen::MatrixXd Descriptors::computeMeshWKS(const MeshProcessorSptr& mesh, const int nDescr, const int K, const Eigen::VectorXd& landMark, const std::string& meshID, const bool scaled)
{
    const Eigen::VectorXd truncatedEvals = mesh->getTruncatedEval(K);
    const Eigen::MatrixXd truncatedEvecs = mesh->getTruncatedEvec(K);
    Eigen::MatrixXd result;
    std::cout << meshID << ": " << "Eigenvalues " << "(" << truncatedEvals.rows() << " x " << truncatedEvals.cols() << ")" << " , ";
    std::cout << "Eigenvectors " << "(" << truncatedEvecs.rows() << " x " << truncatedEvecs.cols() << ")" << std::endl;
    //std::cout << "Truncated Eigen Values: " << "Rows " <<truncatedEvals.size() << " " << "Columns " << truncatedEvals.cols() << " , "  << "Truncated Eigen Vectors: " << "Rows " << truncatedEvecs.rows() << " "  << "Columns " << truncatedEvecs.cols() << std::endl;
    Eigen::VectorXd truncatedEvalAbs = truncatedEvals.cwiseAbs();
    for(int i=0; i<truncatedEvalAbs.size()-1; i++)
    {
        assert(truncatedEvalAbs(i) < truncatedEvalAbs(i+1));
    }

    // Logarithm of eigenvalues:
    double e_min = std::log(truncatedEvalAbs(1));
    double e_max = std::log(truncatedEvalAbs(K-1));

    // Sigma controls the bandwidth of the spectral filters (as in WKS formulation)
    const double sigma = 7 * (e_max - e_min) / nDescr;

    // Energy range adjustment:
    e_min += 2.0 * sigma;
    e_max -= 2.0 * sigma;

    Eigen::VectorXd energy_list(nDescr);
    if(nDescr == 1)
    {
        energy_list(0) = e_min;
    }
    else
    {
        const double step = (e_max - e_min) / (nDescr - 1);
        for(int i=0; i<nDescr; i++)
        {
            energy_list(i) = (e_min + i * step);
        }
    }
    if(landMark.size() == 0)
    {
        result = WKS(truncatedEvalAbs, truncatedEvecs, energy_list, sigma, scaled);
    }
    else
    {
        result = WKSLandMark(truncatedEvalAbs, truncatedEvecs, energy_list, landMark, sigma, scaled);
    }
    return result;
}
