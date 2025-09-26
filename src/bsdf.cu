#include "bsdf.h"

#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#include <thrust/random.h>

__device__ thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

inline __device__ Ray SpawnRay(const glm::vec3& pos, const glm::vec3& wi)
{
    Ray r;
    r.origin = pos + wi * 0.001f;
    r.direction = wi;
    return r;
}

inline __device__ glm::vec3 DiffuseBSDF(const glm::vec3& albedo)
{
    return albedo * INV_PI;
}

__device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    glm::vec3 wi = calculateRandomDirectionInHemisphere(normal, rng);
    glm::vec3 bsdf = DiffuseBSDF(m.color);
    //float lambert = glm::abs(glm::dot(wi, normal));
    //float pdf = glm::abs(glm::dot(wi, normal)) * INV_PI;

    glm::vec3 lightTransportResult = bsdf * PI; // Normally (bsdf*lambert)/pdf but this is simplified
    pathSegment.color *= lightTransportResult;
    pathSegment.ray = SpawnRay(intersect, wi);
    pathSegment.remainingBounces--;
}

__global__ void shadeMaterial(   
        int iter
    ,   int num_paths
    ,   ShadeableIntersection* shadeableIntersections
    ,   PathSegment* pathSegments
    ,   Material* materials
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        const PathSegment path = pathSegments[idx];
        int depth = path.remainingBounces;
        if (depth <= 0)
            return; // Retire this thread early if the ray has gone out of bounds or run out of depth.

        ShadeableIntersection intersection = shadeableIntersections[idx];
        
#if STREAM_COMPACTION
        assert(intersection.t > FLT_EPSILON); // Stream compaction has removed rays that didn't hit anything by this point
#else
        if (intersection.t <= 0.0f)
        {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
            return;
        }
#endif 
        // Set up RNG to generate xi's
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);

        Material material = materials[intersection.materialId];
        glm::vec3 materialColor = material.color;

        Ray wo = path.ray;
        glm::vec3 intersect = wo.origin + intersection.t * wo.direction;
        scatterRay(pathSegments[idx], intersect, intersection.surfaceNormal, material, rng);
    }
}

__global__ void shadeMaterialSpecular(
      int iter
    , int num_paths
    , ShadeableIntersection* shadeableIntersections
    , PathSegment* pathSegments
    , Material* materials
)
{

}

__global__ void shadeMaterialEmissive(
    int iter
    , int num_paths
    , ShadeableIntersection* shadeableIntersections
    , PathSegment* pathSegments
    , Material* materials
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        const PathSegment path = pathSegments[idx];
        int depth = path.remainingBounces;
        if (depth <= 0)
            return; // Retire this thread early if the ray has gone out of bounds or run out of depth.

        ShadeableIntersection intersection = shadeableIntersections[idx];

    #if STREAM_COMPACTION
        assert(intersection.t > FLT_EPSILON); // Stream compaction has removed rays that didn't hit anything by this point
    #else
        if (intersection.t <= 0.0f)
        {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
            return;
        }
    #endif 

        Material material = materials[intersection.materialId];
        pathSegments[idx].color *= (material.color * material.emittance);
        pathSegments[idx].remainingBounces = 0; // Mark it for culling later
    }
}