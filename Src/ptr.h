#ifndef SHAPEANALYSIS_PTR_H
#define SHAPEANALYSIS_PTR_H
#include <memory>
namespace ShapeAnalysis
{
    class MeshProcessor;
    class FunctionalMapping;
    class Descriptors;
    class FunctionalMapEnergy;
    class MapRefinement;

    typedef std::shared_ptr<MeshProcessor> MeshProcessorSptr;
    typedef std::unique_ptr<MeshProcessor> MeshProcessorUptr;

    typedef std::shared_ptr<FunctionalMapping> FunctionalMappingSptr;
    typedef std::unique_ptr<FunctionalMapping> FunctionalMappingUptr;

    typedef std::shared_ptr<Descriptors> DescriptorsSptr;
    typedef std::unique_ptr<Descriptors> DescriptorsUptr;

    typedef std::shared_ptr<FunctionalMapEnergy> FunctionalMapEnergySptr;
    typedef std::unique_ptr<FunctionalMapEnergy> FunctionalMapEnergyUptr;

    typedef std::shared_ptr<MapRefinement> MapRefinementSptr;
    typedef std::unique_ptr<MapRefinement> MapRefinementUptr;
}

#endif //SHAPEANALYSIS_PTR_H