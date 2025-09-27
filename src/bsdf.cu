#include "bsdf.h"

#include "light.h"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#include <thrust/random.h>

__device__ thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

inline __device__ void coordinateSystem(const glm::vec3& v1, glm::vec3& v2, glm::vec3& v3)
{
    if (glm::abs(v1.x) > glm::abs(v1.y))
        v2 = glm::vec3(-v1.z, 0, v1.x) / glm::sqrt(v1.x * v1.x + v1.z * v1.z);
    else
        v2 = glm::vec3(0, v1.z, -v1.y) / glm::sqrt(v1.y * v1.y + v1.z * v1.z);
    v3 = glm::cross(v1, v2);
}

inline __device__ glm::mat3 LocalToWorld(glm::vec3 nor)
{
    glm::vec3 tan, bit;
    coordinateSystem(nor, tan, bit);
    return glm::mat3(tan, bit, nor);
}

inline __device__ glm::mat3 WorldToLocal(glm::vec3 nor) {
    return glm::transpose(LocalToWorld(nor));
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

}

// By convention: MUST match the order of the MaterialType struct
static ShadeKernel sKernels[] =
{
    skDiffuseDirect,
    skSpecular,
    skEmissive,
    skRefractive
};

__host__ ShadeKernel getShadingKernelForMaterial(MaterialType mt)
{
    assert(mt < MT_COUNT);
    return sKernels[mt];
}

#if STREAM_COMPACTION
#define HANDLE_MISS(idx, intersection, pathSegments) \
        assert((intersection).t > FLT_EPSILON);
#else
#define HANDLE_MISS(idx, intersection, pathSegments)              \
        if ((intersection).t <= 0.0f) {                               \
            (pathSegments)[(idx)].color = glm::vec3(0.0f);            \
            (pathSegments)[(idx)].remainingBounces = 0;               \
            return;                                                   \
        }
#endif

__global__ void skDiffuse(ShadeKernelArgs args)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.num_paths)
        return;

    const PathSegment path = args.pathSegments[idx];
    const ShadeableIntersection intersection = args.shadeableIntersections[idx];
    const Material material = args.materials[intersection.materialId];
        
    HANDLE_MISS(idx, intersection, pathSegments);

    thrust::default_random_engine rng = makeSeededRandomEngine(args.iter, idx, path.remainingBounces);

    glm::vec3 wi = calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng);
    glm::vec3 bsdf = DiffuseBSDF(material.color);
    glm::vec3 lightTransportResult = bsdf * PI; // Normally (bsdf*lambert)/pdf but this is simplified

    args.pathSegments[idx].color *= lightTransportResult;
    args.pathSegments[idx].ray = SpawnRay(path.ray.origin + intersection.t * path.ray.direction, wi);
    args.pathSegments[idx].remainingBounces--;
}

__global__ void skDiffuseDirect(ShadeKernelArgs args)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.num_paths)
        return;

    const PathSegment path = args.pathSegments[idx];
    const ShadeableIntersection intersection = args.shadeableIntersections[idx];
    const Material material = args.materials[intersection.materialId];
    thrust::default_random_engine rng = makeSeededRandomEngine(args.iter, idx, path.remainingBounces);

    HANDLE_MISS(idx, intersection, pathSegments);

    glm::vec3 wiW;
    float pdf;
    glm::vec3 view_point = path.ray.origin + (intersection.t * path.ray.direction);
    glm::vec3 totalDirectLight(0.0f);
    glm::vec3 bsdf = DiffuseBSDF(material.color);
    const int NUM_SAMPLES = 4;
    for (int s = 0; s != NUM_SAMPLES; ++s)
    {
        glm::vec3 liResult = Sample_Li(view_point, intersection.surfaceNormal, args.lights, args.num_lights, rng, wiW, pdf);
        if (pdf < FLT_EPSILON)
        {
            continue;
        }

        Ray shadowRay = SpawnRay(view_point, wiW);

        float lambert = glm::abs(glm::dot(wiW, intersection.surfaceNormal));
        totalDirectLight += bsdf * liResult * lambert / (NUM_SAMPLES * pdf);
    }
    
    
    args.pathSegments[idx].color *= totalDirectLight;
    args.pathSegments[idx].remainingBounces = 0;
}

__global__ void skSpecular(ShadeKernelArgs args)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.num_paths)
        return;

    const PathSegment path = args.pathSegments[idx];
    const ShadeableIntersection intersection = args.shadeableIntersections[idx];
    const Material material = args.materials[intersection.materialId];

    HANDLE_MISS(idx, intersection, pathSegments);

    glm::vec3 wiW = glm::reflect(path.ray.direction, intersection.surfaceNormal);
    args.pathSegments[idx].color *= material.color;
    args.pathSegments[idx].ray = SpawnRay(path.ray.origin + intersection.t*path.ray.direction, wiW);
    args.pathSegments[idx].remainingBounces--;
}

__global__ void skEmissive(ShadeKernelArgs args)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.num_paths)
        return;
    
    const PathSegment path = args.pathSegments[idx];
    const ShadeableIntersection intersection = args.shadeableIntersections[idx];
    const Material material = args.materials[intersection.materialId];

    HANDLE_MISS(idx, intersection, pathSegments);

    args.pathSegments[idx].color *= (material.color * material.emittance);
    args.pathSegments[idx].remainingBounces = 0; // Mark it for culling later
}

__global__ void skRefractive(ShadeKernelArgs args)
{
    return; // TODO
}
