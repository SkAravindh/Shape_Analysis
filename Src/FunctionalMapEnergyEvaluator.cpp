#include "FunctionalMapEnergyEvaluator.h"


using namespace ShapeAnalysis;

FunctionalMapEnergyEvaluator::FunctionalMapEnergyEvaluator(const OptimizationParameters& parameters) :
            wDescr_(parameters.wDescr),
            wLap_(parameters.wLap),
            wDescrComm_(parameters.wDescrComm),
            wOrient_(parameters.wOrient),
            descr1_(parameters.descr1Reduced),
            descr2_(parameters.descr2Reduced),
            descrOperatorPairs_(parameters.descrOperatorPairs),
            orientationOperatorPairs_(parameters.orienOperatorPairs),
            eigenvalueSqDiff_(parameters.eigenvalueSqDiff)
{
    assert(descr1_.rows() == descr2_.rows() && "Number of rows should match");
    K = static_cast<int>(descr1_.rows());
}

FunctionalMapEnergyEvaluator::~FunctionalMapEnergyEvaluator()
{}

FunctionalMapEnergyEvaluatorSptr FunctionalMapEnergyEvaluator::create(const OptimizationParameters& parameters)
{
    FunctionalMapEnergyEvaluatorSptr obj(new FunctionalMapEnergyEvaluator(parameters));
    return obj;
}

double FunctionalMapEnergyEvaluator::computeEnergy()
{
    Eigen::MatrixXd C = Eigen::MatrixXd::Identity(K, K);
    double energy = 0.0;
    if(wDescr_)
    {
        energy += wDescr_ * descriptorPreservation(C, descr1_, descr2_);
    }
    if(wLap_)
    {
        energy += wLap_ * laplacianCommutation(C, eigenvalueSqDiff_);
    }
    if(wDescrComm_)
    {
        energy += wDescrComm_ *  operatorCommutation(C, descrOperatorPairs_);
    }
    if(wOrient_)
    {
        energy += wOrient_ * operatorCommutation(C, orientationOperatorPairs_);
    }
    return energy;
}

double FunctionalMapEnergyEvaluator::descriptorPreservation(const Eigen::MatrixXd& C, const Eigen::MatrixXd& descr1, const Eigen::MatrixXd& descr2)
{
    Eigen::MatrixXd diff_descr = C * descr1 - descr2;
    double energy = 0.5 * (diff_descr).array().square().sum();
    return energy;
}

double FunctionalMapEnergyEvaluator::laplacianCommutation(const Eigen::MatrixXd& C, const Eigen::MatrixXd& eigenvalueSqDiff)
{
    double energy = 0.5 * (C.array().square() * eigenvalueSqDiff.array()).sum();
    return energy;
}

double FunctionalMapEnergyEvaluator::operatorCommutation(const Eigen::MatrixXd& C, const std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>& operatorPairs)
{
    double energy = 0.0;
    for(const auto& ele : operatorPairs)
    {
        const Eigen::MatrixXd& op1 = ele.first;
        const Eigen::MatrixXd& op2 = ele.second;
        Eigen::MatrixXd Residual(C.rows(), op1.cols());
        Residual.noalias() = C * op1 - op2 * C;
        energy += 0.5 * Residual.array().square().sum();
    }
    return energy;
}