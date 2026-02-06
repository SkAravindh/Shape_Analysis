#include "FMoptimizer.h"

#include "LBFGS.h"

using namespace ShapeAnalysis;

FunctionalMapEnergy::FunctionalMapEnergy(const OptimizationParameters& params)
    : wDescr_(params.wDescr),
    wLap_(params.wLap),
    wDescrComm_(params.wDescrComm),
    wOrient_(params.wOrient),
    descr1_(params.descr1Reduced),
    descr2_(params.descr2Reduced),
    descrOperatorPairs_(params.descrOperatorPairs),
    orientationOperatorPairs_(params.orienOperatorPairs),
    eigenvalueSqDiff_(params.eigenvalueSqDiff)
{
    assert(descr1_.rows() == descr2_.rows() && "Number of rows should match");
    K = static_cast<int>(descr1_.rows());
}

FunctionalMapEnergy::~FunctionalMapEnergy()
{}

FunctionalMapEnergySptr FunctionalMapEnergy::create(const OptimizationParameters& params)
{
    std::shared_ptr<FunctionalMapEnergy> obj =  std::shared_ptr<FunctionalMapEnergy>(new FunctionalMapEnergy(params));
    return obj;
}

const Eigen::MatrixXd& FunctionalMapEnergy::getMatrixC() const
{
    return C_;
}

double FunctionalMapEnergy::operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) const
{
    ++functional_calls;
    const Eigen::Map<const Eigen::MatrixXd> C(x.data(), K, K);
    Eigen::MatrixXd gradC = Eigen::MatrixXd::Zero(K, K);
    double energy = 0.0;

    //******************************************************************************************************************
    // Compute the descriptor preservation constraint
    // 0.5 * || C D1 - D2 ||_F^2
    //******************************************************************************************************************
    {
        Eigen::MatrixXd diff_descr = C * descr1_ - descr2_;
        energy += wDescr_ * 0.5 * (diff_descr).array().square().sum();
        gradC.noalias() +=  wDescr_ * diff_descr * descr1_.transpose();
    }

    //******************************************************************************************************************
    //  Compute the LB commutativity constraint
    // || CΛ_1 - Λ_2C ||^2. or  0.5 * sum(C^2 ⊙ ev_sqdiff)
    //******************************************************************************************************************
    {
        energy += wLap_ * 0.5 * (C.array().square() * eigenvalueSqDiff_.array()).sum();
        gradC.noalias() += wLap_ * (C.array() * eigenvalueSqDiff_.array()).matrix();
    }

    //******************************************************************************************************************
    //  Compute the operator commutativity constraint. Can be used with descriptor multiplication operator
    // 0.5 * || C A - B C ||_F^2
    //******************************************************************************************************************
    {
        for(const auto& ele : descrOperatorPairs_)
        {
            const Eigen::MatrixXd& op1 = ele.first;
            const Eigen::MatrixXd& op2 = ele.second;
            Eigen::MatrixXd Residual = C * op1 - op2 * C;
            energy += wDescrComm_ * 0.5 * Residual.array().square().sum();
            gradC.noalias() += wDescrComm_ * (Residual * op1.transpose() - op2.transpose() * Residual);
        }
    }
    //******************************************************************************************************************
    //  Compute the operator commutativity constraint. Can be used with orientation multiplication operator
    // 0.5 * || C A - B C ||_F^2
    //******************************************************************************************************************
    {
        for(const auto& ele : orientationOperatorPairs_)
        {
            const Eigen::MatrixXd& op1 = ele.first;
            const Eigen::MatrixXd& op2 = ele.second;
            Eigen::MatrixXd Residual(C.rows(), op1.cols());
            Residual.noalias() = C * op1 - op2 * C;
            energy += wOrient_ * 0.5 * Residual.array().square().sum();
            gradC.noalias() += wOrient_ * (Residual * op1.transpose() - op2.transpose() * Residual);
        }
    }

    // write graadient back to vector
    Eigen::Map<Eigen::MatrixXd>(grad.data(), K, K) = gradC;
    return energy;
}

void FunctionalMapEnergy::solve()
{
    std::cout << std::endl;
    std::cout << "\033[032m" << "Optimizing Functional Map using LBFGS++ package " << "\033[0m" << std::endl;
    const int dim = K * K;
    Eigen::MatrixXd C0 = Eigen::MatrixXd::Zero(K, K);
    C0.diagonal().setOnes();
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(C0.data(), dim);

    LBFGSpp::LBFGSParam<double> params;
    params.epsilon = 1e-6;
    params.max_iterations = 200;

    LBFGSpp::LBFGSSolver<double> solver(params);

    double fx = 0.0;
    const int niter = solver.minimize(*this, x, fx);

    // Final Gradient
    Eigen::VectorXd grad(dim);
    (*this)(x, grad);
    const double grad_norm = grad.norm();

    // Final Functional Map
    C_ = Eigen::Map<Eigen::MatrixXd>(x.data(), K, K);
    std::cout << std::endl;
    std::cout << "\033[31m" << "Numerical Optimization Summary " << "\033[0m" << std::endl;
    std::cout << "\033[032m" << "Iteration       :   " << "\033[0m" << niter << std::endl;
    std::cout << "\033[032m" << "Final Energy    :   " << "\033[0m" << fx << std::endl;
    std::cout << "\033[032m" << "Gradient Norm   :   " << "\033[0m" << grad_norm << std::endl;
    std::cout << "\033[032m" << "Functions calls :   " << "\033[0m" << functional_calls << std::endl;

    if(grad_norm <= params.epsilon)
    {
        std::cout << "\033[032m" << "Converged with gradient tolerance" << "\033[0m" << std::endl;
    }
    else if (niter >= params.max_iterations)
    {
        std::cout << "\033[032m" << "Stopped with max iterations" << "\033[0m" << std::endl;
    }
    else
    {
        std::cout << "\033[032m" << "Stopped with (line search / flat energy)" << "\033[0m" << std::endl;
    }
    std::cout << "\033[032m" << "Done !!" << "\033[0m" << std::endl;
}