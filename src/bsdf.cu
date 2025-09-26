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

// By convention: MUST match the order of the MaterialType struct
static ShadeKernel sKernels[] =
{
    skDiffuse,
    skSpecular,
    skEmissive,
    skRefractive
};

__host__ ShadeKernel getShadingKernelForMaterial(MaterialType mt)
{
    assert(mt < MT_COUNT);
    return sKernels[mt];
}

__global__ void skDiffuse(ShadeKernelArgs args)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < args.num_paths)
    {
        const PathSegment path = args.pathSegments[idx];
        int depth = path.remainingBounces;
        if (depth <= 0)
            return; // Retire this thread early if the ray has gone out of bounds or run out of depth.

        ShadeableIntersection intersection = args.shadeableIntersections[idx];
        
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
        thrust::default_random_engine rng = makeSeededRandomEngine(args.iter, idx, depth);

        Material material = args.materials[intersection.materialId];
        glm::vec3 materialColor = material.color;

        Ray wo = path.ray;
        glm::vec3 intersect = wo.origin + intersection.t * wo.direction;
        scatterRay(args.pathSegments[idx], intersect, intersection.surfaceNormal, material, rng);
    }
}

__global__ void skSpecular(ShadeKernelArgs args)
{
    // TODO: Using diffuse shading temporarily

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < args.num_paths)
    {
        const PathSegment path = args.pathSegments[idx];
        int depth = path.remainingBounces;
        if (depth <= 0)
            return; // Retire this thread early if the ray has gone out of bounds or run out of depth.

        ShadeableIntersection intersection = args.shadeableIntersections[idx];

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
        thrust::default_random_engine rng = makeSeededRandomEngine(args.iter, idx, depth);

        Material material = args.materials[intersection.materialId];
        glm::vec3 materialColor = material.color;

        Ray wo = path.ray;
        glm::vec3 intersect = wo.origin + intersection.t * wo.direction;
        scatterRay(args.pathSegments[idx], intersect, intersection.surfaceNormal, material, rng);
    }
}

__global__ void skEmissive(ShadeKernelArgs args)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < args.num_paths)
    {
        const PathSegment path = args.pathSegments[idx];
        int depth = path.remainingBounces;
        if (depth <= 0)
            return; // Retire this thread early if the ray has gone out of bounds or run out of depth.

        ShadeableIntersection intersection = args.shadeableIntersections[idx];

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

        Material material = args.materials[intersection.materialId];
        args.pathSegments[idx].color *= (material.color * material.emittance);
        args.pathSegments[idx].remainingBounces = 0; // Mark it for culling later
    }
}

__global__ void skRefractive(ShadeKernelArgs args)
{
    return; // TODO
}
