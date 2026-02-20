#ifndef PTRMAPREFINEMENT_H
#define PTRMAPREFINEMENT_H
#include <memory>
namespace ShapeAnalysis
{
    class IterativeClosestPoint;
    class ZoomOut;

    typedef std::shared_ptr<IterativeClosestPoint> IterativeClosestPointSptr;
    typedef std::unique_ptr<IterativeClosestPoint> IterativeClosestPointUptr;

    typedef std::shared_ptr<ZoomOut> ZoomOutSptr;
    typedef std::unique_ptr<ZoomOut> ZoomOutUptr;
}

#endif
