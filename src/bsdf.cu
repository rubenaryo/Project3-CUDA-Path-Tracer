#include "bsdf.h"

#include "light.h"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

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

////////////////////////
// PDF functions

__device__ float squareToHemisphereCosinePDF(const glm::vec3& sampleL)
{
    return sampleL.z * INV_PI;
}

__device__ float Pdf(MaterialType matType, glm::vec3 norW, glm::vec3 woW, glm::vec3 wiW)
{
    glm::vec3 nor = norW;
    glm::mat3 world2Local = WorldToLocal(nor);
    glm::vec3 wo = world2Local * woW;
    glm::vec3 wi = world2Local * wiW;

    switch (matType)
    {
    case MT_DIFFUSE:
        return squareToHemisphereCosinePDF(wi);
    case MT_SPECULAR:
    case MT_REFRACTIVE:
        return 0.0f; // Spec goes directly to the light, and refractive goes into the material.
    }

    return 0.0f; // Unhandled material.
}

////////////////////////
// f / Samplef functions

inline __device__ glm::vec3 f_diffuse(const glm::vec3& albedo)
{
    return albedo * INV_PI;
}

inline __device__ glm::vec3 Sample_f_diffuse(const glm::vec3& albedo, const glm::vec3& norW, thrust::default_random_engine& rng, glm::vec3& out_wiW, float& out_pdf)
{
    out_wiW = calculateRandomDirectionInHemisphere(norW, rng);
    out_pdf = glm::abs(glm::dot(out_wiW, norW)) * INV_PI;
    return f_diffuse(albedo);
}

__device__ glm::vec3 SolveMIS(ShadeableIntersection isect, const SceneData& sd, glm::vec3 view_point, glm::vec3 woW, const Material mat, thrust::default_random_engine& rng)
{
    // Direct Sampling
    int numLights = sd.lights_size;
    thrust::uniform_int_distribution<int> iu0N(0, numLights - 1);
    int randomLightIndex = iu0N(rng);
    const Light chosenLight = sd.lights[randomLightIndex];
    glm::vec3 norW = isect.surfaceNormal;

    glm::vec3 wiW_Li;
    float distToLight_Li;
    float pdf_Li;
    glm::vec3 bsdf_Li = Sample_Li(view_point, norW, chosenLight, numLights, rng, wiW_Li, pdf_Li, distToLight_Li);

    float Li_absDot = glm::abs(glm::dot(wiW_Li, norW));
    glm::vec3 Li_result = bsdf_Li * Li_absDot;

    if (pdf_Li < FLT_EPSILON)
        Li_result = glm::vec3(0.f);
    else
        Li_result /= pdf_Li;

    // BSDF Sampling
    glm::vec3 wiW_bsdf;
    float pdf_bsdf;
    int type_bsdf;
    glm::vec3 bsdf = Sample_f_diffuse(mat.color, norW, rng, wiW_bsdf, pdf_bsdf); // TODO: This should handle non-diffuse as well.

    PathSegment bsdfRay;
    //bsdfRay.ray = SpawnRay(view_point, wiW_bsdf);
    bsdfRay.ray.direction = wiW_bsdf;
    bsdfRay.ray.origin = view_point + 0.001f * wiW_bsdf;

    float bsdf_absDot = glm::abs(glm::dot(wiW_bsdf, norW));

    ShadeableIntersection bsdfIsect;
    sceneIntersect(bsdfRay, sd, bsdfIsect);

    glm::vec3 bsdf_result = bsdfRay.color * bsdf * bsdf_absDot / pdf_bsdf;

    // Cross-method PDF
    float pdf_Li_bsdf = Pdf(GetMaterialTypeFromSortKey(isect.matSortKey), isect.surfaceNormal, woW, wiW_Li); // Li ray with respect to bsdf TODO: This only works for diffuse
    float pdf_bsdf_Li = Pdf_Li(chosenLight, view_point, norW, wiW_bsdf); // bsdf ray with respect to light

    float w = 0.5, wg = 0.5;
    if (pdf_bsdf > 0.001 && pdf_bsdf_Li > 0.001)
        w = PowerHeuristic(1, pdf_bsdf, 1, pdf_bsdf_Li);

    if (pdf_Li > 0.001 && pdf_bsdf_Li > 0.001)
        wg = PowerHeuristic(1, pdf_Li, 1, pdf_Li_bsdf);

    // Assemble final LTE weighted sum
    //return Li_result;
    return (bsdf_result * w) + (Li_result * wg);

    // TODO_MIS: Using resulting wiW, sample the scene and IF the same light was hit as Sample_Li, blend the two together.

    //ShadeableIntersection isect_bsdf;
    //sceneIntersect(bsdfRay, geoms, numGeoms, isect_bsdf);
    //glm::vec3 bsdf_result = isect_bsdf.Le * bsdf * bsdf_absDot;
    //
    //if (pdf_bsdf < 0.001)
    //	bsdf_result = glm::vec3(0.0);
    //else
    //	bsdf_result /= pdf_bsdf;

    // TODO_MIS
    return glm::vec3(0.0f);
}

/////////

// By convention: MUST match the order of the MaterialType struct
static ShadeKernel sKernels[] =
{
    skDiffuseFull,
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
    const Material material = args.materials[GetMaterialIDFromSortKey(intersection.matSortKey)];
        
    HANDLE_MISS(idx, intersection, pathSegments);

    thrust::default_random_engine rng = makeSeededRandomEngine(args.iter, idx, path.remainingBounces);

    glm::vec3 wi = calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng);
    glm::vec3 bsdf = f_diffuse(material.color);
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
    const Material material = args.materials[GetMaterialIDFromSortKey(intersection.matSortKey)];
    thrust::default_random_engine rng = makeSeededRandomEngine(args.iter, idx, path.remainingBounces);

    HANDLE_MISS(idx, intersection, pathSegments);

    Light* lights = args.sceneData.lights;
    int numLights = args.sceneData.lights_size;
    thrust::uniform_int_distribution<int> iu0N(0, numLights - 1);
    glm::vec3 wiW;
    float pdf;
    glm::vec3 view_point = path.ray.origin + (intersection.t * path.ray.direction);
    glm::vec3 totalDirectLight(0.0f);
    glm::vec3 bsdf = f_diffuse(material.color);
    const int NUM_SAMPLES = 4;
    for (int s = 0; s != NUM_SAMPLES; ++s)
    {
        float distToLight;
        int randomLightIndex = iu0N(rng);
        const Light chosenLight = lights[randomLightIndex];

        glm::vec3 liResult = Sample_Li(view_point, intersection.surfaceNormal, chosenLight, numLights, rng, wiW, pdf, distToLight);
        if (pdf < FLT_EPSILON)
            continue;

        PathSegment shadowPath;
        shadowPath.ray = SpawnRay(view_point, wiW);
        ShadeableIntersection shadowTestResult;
        sceneIntersect(shadowPath, args.sceneData, shadowTestResult);

        if (shadowTestResult.t >= 0.0f && shadowTestResult.t < (distToLight - FLT_EPSILON))
            continue;

        float lambert = glm::abs(glm::dot(wiW, intersection.surfaceNormal));
        totalDirectLight += bsdf * liResult * lambert / (NUM_SAMPLES * pdf);
    }
    
    args.pathSegments[idx].color *= totalDirectLight;
    args.pathSegments[idx].remainingBounces = 0;
}

__global__ void skDiffuseFull(ShadeKernelArgs args)
{
    // TODO_MIS
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.num_paths)
        return;

    const PathSegment path = args.pathSegments[idx];
    const ShadeableIntersection intersection = args.shadeableIntersections[idx];
    const Material material = args.materials[GetMaterialIDFromSortKey(intersection.matSortKey)];

    HANDLE_MISS(idx, intersection, pathSegments);

    thrust::default_random_engine rng = makeSeededRandomEngine(args.iter, idx, path.remainingBounces);
    glm::vec3 view_point = path.ray.origin + intersection.t * path.ray.direction;
    glm::vec3 misResult = SolveMIS(intersection, args.sceneData, view_point, path.ray.direction, material, rng);

    args.pathSegments[idx].color *= misResult;
    //args.pathSegments[idx].ray = SpawnRay(path.ray.origin + intersection.t * path.ray.direction, wiW);
    args.pathSegments[idx].remainingBounces = 0;
}

__global__ void skSpecular(ShadeKernelArgs args)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.num_paths)
        return;

    const PathSegment path = args.pathSegments[idx];
    const ShadeableIntersection intersection = args.shadeableIntersections[idx];
    const Material material = args.materials[GetMaterialIDFromSortKey(intersection.matSortKey)];

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
    const Material material = args.materials[GetMaterialIDFromSortKey(intersection.matSortKey)];

    HANDLE_MISS(idx, intersection, pathSegments);

    args.pathSegments[idx].color *= (material.color * material.emittance);
    args.pathSegments[idx].remainingBounces = 0; // Mark it for culling later
}

__global__ void skRefractive(ShadeKernelArgs args)
{
    return; // TODO
}
