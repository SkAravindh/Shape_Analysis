#ifndef PTRMAPREFINEMENT_H
#define PTRMAPREFINEMENT_H
#include <memory>
namespace ShapeAnalysis
{
    class IterativeClosestPoint;

    typedef std::shared_ptr<IterativeClosestPoint> IterativeClosestPointSptr;
    typedef std::unique_ptr<IterativeClosestPoint> IterativeClosestPointUptr;
}

#endif
