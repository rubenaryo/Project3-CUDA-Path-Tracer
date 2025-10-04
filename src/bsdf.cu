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

__host__ __device__ inline bool isnanVec3(const glm::vec3& v) {
    return isnan(v.x) || isnan(v.y) || isnan(v.z);
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

inline __device__ glm::vec4 TextureSample(cudaTextureObject_t texObj, const glm::vec2& uv)
{
    float4 color = tex2D<float4>(texObj, uv.x, uv.y);
    return glm::vec4(color.x, color.y, color.z, color.w);
}

inline __device__ glm::vec4 TryTextureSample(cudaTextureObject_t* texObjArr, int textureId, const glm::vec2& uv, const glm::vec4& fallbackValue)
{
    if (textureId == -1)
        return fallbackValue;

    return TextureSample(texObjArr[textureId], uv);
}

inline __device__ glm::vec3 TryTextureSample(cudaTextureObject_t* texObjArr, int textureId, const glm::vec2& uv, const glm::vec3& fallbackValue)
{
    return glm::vec3(TryTextureSample(texObjArr, textureId, uv, glm::vec4(fallbackValue, 0.0f)));
}

////////////////////////
// PBR Utils (Mostly from 561)

__device__ float pow5(float x) {
    float x2 = x * x;
    return x2 * x2 * x;
}

// Schlick's Fresnel approximation
//__device__ glm::vec3 fresnelSchlick(float cosTheta, const glm::vec3& F0) {
//    return F0 + (glm::vec3(1.0f) - F0) * pow5(glm::clamp(1.0f - cosTheta, 0.0f, 1.0f));
//}

__device__ glm::vec3 fresnelSchlickRoughness(float cosTheta, glm::vec3 F0, float roughness)
{
    return F0 + (glm::max(glm::vec3(1.0f - roughness), F0) - F0) * pow5(glm::clamp(1.0f - cosTheta, 0.0f, 1.0f));
}

// GGX/Trowbridge-Reitz Normal Distribution Function
__device__ float distributionGGX(const glm::vec3& norW, const glm::vec3& whW, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = glm::max(glm::dot(norW, whW), 0.0f);
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = PI * denom * denom;
    denom = glm::max(denom, 0.01f);

    return a2 / denom;
}

__device__ glm::vec3 sampleGGX(const glm::vec3& norW, float roughness, thrust::default_random_engine& rng) {
    
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    float u1 = u01(rng);
    float u2 = u01(rng);
    
    float a = roughness * roughness;
    float a2 = a * a;

    float phi = 2.0f * PI * u1;

    // Spherical coords
    float tanTheta2 = a2 * u2 / (1.0f - u2 + FLT_EPSILON);
    float cosTheta = 1.0f / sqrt(1.0f + tanTheta2);
    float sinTheta = sqrt(glm::max(0.0f, 1.0f - cosTheta * cosTheta));

    assert(!isnan(cosTheta) && !isnan(sinTheta));

    // Cartesian
    glm::vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;

    // Build TBN frame
    glm::vec3 up = abs(norW.z) < 0.999f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 tangent = glm::normalize(glm::cross(up, norW));
    glm::vec3 bitangent = glm::cross(norW, tangent);

    glm::vec3 result = glm::normalize(tangent * H.x + bitangent * H.y + norW * H.z);

    assert(!isnanVec3(result));
    return result;
}

// Smith's Geometry function with GGX (Schlick-GGX)
__device__ float geometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0f);
    float k = (r * r) / 8.0f;

    return NdotV / (NdotV * (1.0f - k) + k);
}

__device__ float geometrySmith(const glm::vec3& norW, const glm::vec3& woW, const glm::vec3& wiW, float roughness) {
    float NdotWo = glm::max(glm::dot(norW, woW), 0.1f);
    float NdotWi = glm::max(glm::dot(norW, wiW), 0.1f);
    float ggx2 = geometrySchlickGGX(NdotWo, roughness);
    float ggx1 = geometrySchlickGGX(NdotWi, roughness);

    return ggx1 * ggx2;
}
////////////////////////
// PDF functions

__device__ float squareToHemisphereCosinePDF(const glm::vec3& sampleL)
{
    return sampleL.z * INV_PI;
}

__device__ float pdfGGX(const glm::vec3& norW, const glm::vec3& whW, const glm::vec3& woW, float roughness) {
    float HdotN  = glm::max(glm::dot(whW, norW), 0.0f);
    float HdotWo = glm::max(glm::dot(whW, woW), 0.01f);
    
    float D = distributionGGX(norW, whW, roughness);


    return (D * HdotN) / (4.0f * HdotWo);
}

__device__ float pdfCookTorrance(const glm::vec3& norW, const glm::vec3& whW, const glm::vec3& woW, const glm::vec3& wiW, float roughness, float metallic) {
    // Mix between diffuse and specular PDF based on Fresnel
    float VdotH = glm::max(glm::dot(woW, whW), 0.01f);
    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), glm::vec3(1.0f), metallic);
    glm::vec3 F = fresnelSchlickRoughness(VdotH, F0, roughness);
    float specularWeight = (F.x + F.y + F.z) / 3.0f;

    float pdfSpec = pdfGGX(norW, whW, woW, roughness);
    float pdfDiff = glm::max(glm::dot(norW, wiW), 0.0f) * INV_PI;

    assert(!isnan(pdfSpec) && !isnan(pdfDiff) && !isnan(specularWeight));

    return glm::mix(pdfDiff, pdfSpec, specularWeight);
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
    case MT_MICROFACET_PBR:
    {
        //glm::vec3 whW = glm::normalize(woW + wiW);
        //return pdfCookTorrance(norW, whW, woW, wiW, 0.1f, 0.1f);
    }
    case MT_SPECULAR:
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

inline __device__ glm::vec3 f_spec(const glm::vec3& albedo, const glm::vec3& wiW, const glm::vec3& norW)
{
    float absCosTheta = glm::abs(glm::dot(wiW, norW));
    if (absCosTheta < FLT_EPSILON)
        return glm::vec3(0.0f);

    return albedo / absCosTheta;
}

inline __device__ glm::vec3 f_cookTorrance(const glm::vec3& albedo, const glm::vec3& norW, const glm::vec3& woW, const glm::vec3& wiW, float roughness, float metallic)
{
    glm::vec3 whW = glm::normalize(wiW + woW);

    float NdotWo = glm::max(glm::dot(norW, woW), 0.0f);
    float NdotWi = glm::max(glm::dot(norW, wiW), 0.0f);

    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), albedo, metallic);

    // Cook-Torrance specular
    float D = distributionGGX(norW, whW, roughness);
    float G = geometrySmith(norW, woW, wiW, roughness);
    glm::vec3 F = fresnelSchlickRoughness((glm::dot(whW, woW)), F0, roughness);

    glm::vec3 numerator = D * G * F;
    float denominator = 4.0f * NdotWi * NdotWo + 0.001f; // Add epsilon to prevent division by zero
    glm::vec3 specular = numerator / denominator;

    // Lambertian component
    // For metals, diffuse is 0 
    glm::vec3 kS = F; // Specular part
    glm::vec3 kD = glm::vec3(1.0f) - kS; // Diffuse part
    kD *= (1.0f - metallic); // Metals have no diffuse

    glm::vec3 diffuse = kD * albedo / PI;

    return (diffuse + specular) * NdotWo;
}

inline __device__ glm::vec3 Sample_f_diffuse(const glm::vec3& albedo, const glm::vec3& norW, thrust::default_random_engine& rng, glm::vec3& out_wiW, float& out_pdf)
{
    out_wiW = calculateRandomDirectionInHemisphere(norW, rng);
    out_pdf = glm::abs(glm::dot(out_wiW, norW)) * INV_PI;
    return f_diffuse(albedo);
}

inline __device__ glm::vec3 Sample_f_specular(const glm::vec3& albedo, const glm::vec3& woW, const glm::vec3& norW, glm::vec3& out_wiW, float& out_pdf)
{
    out_wiW = glm::reflect(woW, norW);
    out_pdf = 1.0f;
    return f_spec(albedo, out_wiW, norW);
}

__device__ glm::vec3 Sample_f_cookTorrance(const glm::vec3& albedo, const glm::vec3& woW, const glm::vec3& norW, float metallic, float roughness, thrust::default_random_engine& rng, glm::vec3& out_wiW, float& out_pdf)
{
    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), albedo, metallic);

    float NdotWo = glm::max(glm::dot(woW, norW), 0.0f);
    glm::vec3 F = fresnelSchlickRoughness(NdotWo, F0, roughness);
    float specWeight = (F.x + F.y + F.z) / 3.0f; // Avg of each component, used to choose between spec and diffuse

    glm::vec3 wiW(0.0f);
    glm::vec3 whW(0.0f); // resulting view vector and half vector

    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    // Choose spec vs diffuse based on random number and spec weight
    float u1 = u01(rng);
    if (u1 < specWeight)
    {
        whW = sampleGGX(norW, roughness, rng);
        wiW = glm::reflect(-woW, whW);

        assert(!isnanVec3(wiW) && !isnanVec3(whW));

        // hemisphere check
        if (glm::dot(wiW, norW) <= 0.0f)
        {
            out_pdf = 1.0f;
            return glm::vec3(1.0f);
        }
    }
    else
    {
        // Just diffuse sampling
        wiW = calculateRandomDirectionInHemisphere(norW, rng);
        whW = glm::normalize(woW + wiW);
    }


    assert(!isnanVec3(wiW) && !isnanVec3(whW));

    float pdf = pdfCookTorrance(norW, whW, woW, wiW, roughness, metallic);
    if (pdf < FLT_EPSILON)
    {
        out_pdf = 1.0f;
        return glm::vec3(1.0f);
    }

    out_wiW = wiW;
    out_pdf = pdf;
    return f_cookTorrance(albedo, norW, woW, wiW, roughness, metallic);
}

__device__ bool SolveDirectLighting(const SceneData& sd, cudaTextureObject_t envMapObj, const glm::vec3& norW, glm::vec3 view_point, thrust::default_random_engine& rng, glm::vec3& out_radiance, glm::vec3& out_wiW, float& out_pdf)
{
    envMapObj = 0;
    int numLights = sd.lights_size;
    //if (numLights <= 0)
    //    return false;

    int randomLightIndex = 0;

    if (numLights > 0)
    {
        thrust::uniform_int_distribution<int> iu0N(0, envMapObj != 0 ? numLights : numLights-1);
        randomLightIndex = iu0N(rng);
    }

    float pdf_Li;
    float distToLight;
    glm::vec3 wiW_Li;
    int ignoreGeomId = -1;
    glm::vec3 liResult;

    if (randomLightIndex == numLights) // This is out of bounds for the array, this represents the environment map.
    {
        // Like diffuse bsdf, just choose a random direction in our local hemisphere.
        wiW_Li = calculateRandomDirectionInHemisphere(norW, rng);
        pdf_Li = glm::abs(glm::dot(wiW_Li, norW)) * INV_PI;

        liResult = sampleEnvironmentMap(envMapObj, wiW_Li);
    }
    else
    {
        const Light chosenLight = sd.lights[randomLightIndex];
        ignoreGeomId = chosenLight.geomId;

        liResult = Sample_Li(view_point, norW, chosenLight, numLights, rng, wiW_Li, pdf_Li, distToLight);
        if (pdf_Li < FLT_EPSILON)
            return false;
    }

    // Test occlusion
    PathSegment shadowPath;
    shadowPath.ray = SpawnRay(view_point, wiW_Li);
    ShadeableIntersection shadowTestResult;
    sceneIntersect(shadowPath, sd, shadowTestResult, nullptr, ignoreGeomId);
    
    if (shadowTestResult.t >= 0.0f && shadowTestResult.t < (distToLight - FLT_EPSILON))
        return false;

    out_radiance = liResult;
    out_wiW = wiW_Li;
    out_pdf = pdf_Li;

    return true;
}

/////////

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
    glm::vec3 view_point = path.ray.origin + intersection.t * path.ray.direction;
    glm::vec3 thisBounceRadiance(0.0f); // Comes from direct lighting only
    
    const glm::vec3 ERROR_COLOR(1.0f, 0.4118f, 0.7059f);
    glm::vec3 wiW_bsdf;
    float pdf_bsdf;
    glm::vec3 bsdf;

    glm::vec3 albedo;
    if (material.diffuseTexId != -1)
    {
        cudaTextureObject_t texObj = args.textures[material.diffuseTexId];
        if (!texObj)
            albedo = ERROR_COLOR;
        else
            albedo = glm::vec3(TextureSample(texObj, intersection.uv));
    }
    else
    {
        albedo = material.color;
    }
    
    // BSDF Sampling
    glm::vec3 norW = intersection.surfaceNormal;
    bsdf = Sample_f_diffuse(albedo, norW, rng, wiW_bsdf, pdf_bsdf);
    if (pdf_bsdf < FLT_EPSILON)
    {
        // Something went wrong, terminate
        args.pathSegments[idx].remainingBounces = 0;
        return;
    }

    glm::vec3 directRadiance;
    glm::vec3 wiW_Li;
    float pdf_Li;

    float lambert = glm::abs(glm::dot(norW, wiW_bsdf));
    args.pathSegments[idx].throughput *= (bsdf/pdf_bsdf) * lambert;
    args.pathSegments[idx].prevBounceSample.pdf = pdf_bsdf;
    args.pathSegments[idx].prevBounceSample.matType = MT_DIFFUSE;
    args.pathSegments[idx].ray = SpawnRay(view_point, wiW_bsdf);
    args.pathSegments[idx].remainingBounces--;

    // Direct Light Sampling
    // Key difference using MIS: Accumulate direct lighting radiance here.
    glm::vec3 throughput = args.pathSegments[idx].throughput;
    cudaTextureObject_t envMapObj = args.envMaps ? args.envMaps[0] : 0;
    if (SolveDirectLighting(args.sceneData, envMapObj, norW, view_point, rng, directRadiance, wiW_Li, pdf_Li))
    {
        float bsdf_pdf = Pdf(material.type, norW, -path.ray.direction, wiW_Li);
        float lambert_Li = glm::abs(glm::dot(norW, wiW_Li));

        // Assemble direct lighting components
        glm::vec3 directLightResult = args.pathSegments[idx].throughput * directRadiance * lambert_Li / pdf_Li;
        thisBounceRadiance += directLightResult * PowerHeuristic(1, pdf_Li, 1, bsdf_pdf);
        args.pathSegments[idx].Lo += thisBounceRadiance;
    }
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

    glm::vec3 view_point = path.ray.origin + intersection.t * path.ray.direction;
    glm::vec3 wiW_bsdf;
    float pdf_bsdf; // for spec materials, this should be 1.0
    glm::vec3 bsdf = Sample_f_specular(material.color, path.ray.direction, intersection.surfaceNormal, wiW_bsdf, pdf_bsdf);

    // Spec bounces don't need to do direct light calculation, since they only reflect light in one direction.
    float lambert = glm::abs(glm::dot(intersection.surfaceNormal, wiW_bsdf));
    args.pathSegments[idx].throughput *= (bsdf * lambert) / pdf_bsdf;
    args.pathSegments[idx].ray = SpawnRay(view_point, wiW_bsdf);
    args.pathSegments[idx].prevBounceSample.pdf = pdf_bsdf;
    args.pathSegments[idx].prevBounceSample.matType = MT_SPECULAR;
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

    assert(material.type == MT_EMISSIVE);
    
    glm::vec3 totalRadiance(0.0f);
    glm::vec3 throughput = args.pathSegments[idx].throughput;
    if (args.depth == 0 || path.prevBounceSample.matType == MT_SPECULAR) // If this is the first bounce or we just came from specular, there is no "previous" data to go off
    {
        totalRadiance = material.color * material.emittance * throughput;
    }
    else
    {
        if (intersection.hitGeomIdx == -1)
        {
            // Error: this intersection should have geometry associated with it.
            args.pathSegments[idx].Lo = glm::vec3(0.0f);
            args.pathSegments[idx].remainingBounces = 0;
            return;
        }
        // TODO: Check that path.previous is not specular
        glm::mat4 geomTfm = args.sceneData.geoms[intersection.hitGeomIdx].transform; // TODO: Maybe just hold the tfm (and type) instead.
        thrust::default_random_engine rng = makeSeededRandomEngine(args.iter, idx, path.remainingBounces);
        thrust::uniform_int_distribution<float> uH(-0.5f, 0.5f);
        glm::vec4 randPosLocal(uH(rng), uH(rng), 0.0f, 1.0f);
        glm::vec3 randPosWorld = glm::vec3(geomTfm * randPosLocal);

        float lightPdf = Pdf_Rect(geomTfm, path.ray.origin, randPosWorld, intersection.surfaceNormal);
        float bsdfPdf = path.prevBounceSample.pdf;
        totalRadiance += (material.color * material.emittance) * throughput * PowerHeuristic(1, bsdfPdf, 1, lightPdf);
    }

    args.pathSegments[idx].Lo += totalRadiance;
    args.pathSegments[idx].remainingBounces = 0; // Mark it for culling later
}

__global__ void skMicrofacetPBR(ShadeKernelArgs args)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.num_paths)
        return;

    const PathSegment path = args.pathSegments[idx];
    const ShadeableIntersection intersection = args.shadeableIntersections[idx];
    const Material material = args.materials[GetMaterialIDFromSortKey(intersection.matSortKey)];

    HANDLE_MISS(idx, intersection, pathSegments);

    thrust::default_random_engine rng = makeSeededRandomEngine(args.iter, idx, path.remainingBounces);
    glm::vec3 view_point = path.ray.origin + intersection.t * path.ray.direction;
    glm::vec3 thisBounceRadiance(0.0f); // Comes from direct lighting only

    const glm::vec4 metallicRoughFallback(material.metallic, material.roughness, 1.0f, 1.0f);

    glm::vec3 woW = -path.ray.direction;

    glm::vec3 albedo = TryTextureSample(args.textures, material.diffuseTexId, intersection.uv, material.color);
    glm::vec4 metallicRough = TryTextureSample(args.textures, material.metallicRoughTexId, intersection.uv, metallicRoughFallback);

    glm::vec3 norW = intersection.surfaceNormal;
    
    if (material.normalTexId != -1)
    {
        glm::vec3 sampledNormal = glm::vec3(TextureSample(args.textures[material.normalTexId], intersection.uv));
        glm::vec3 norT = (2.0f * sampledNormal) - glm::vec3(1.0f);

        glm::vec3 tangent = intersection.tangent;
        glm::vec3 bitangent = glm::normalize(glm::cross(norW, tangent));

        // Ensure right-handed coordinate system
        if (glm::dot(glm::cross(norW, tangent), bitangent) < 0.0f) {
            tangent = -tangent;
        }

        glm::mat3 TBN(tangent, bitangent, norW);
        norW = glm::normalize(TBN * norT);
    }
    
    // This assumes the metallic and roughness always go in the same channels as they should
    // Someone should let the people on sketchfab know....
    float roughness = glm::clamp(metallicRough.g, 0.01f, 1.0f);
    float metallic = glm::clamp(metallicRough.b, 0.01f, 1.0f);

    glm::vec3 wiW;
    float pdf;
    glm::vec3 bsdf = Sample_f_cookTorrance(albedo, woW, norW, metallic, roughness, rng, wiW, pdf);
    if (pdf < FLT_EPSILON)
    {
        // Something went wrong, terminate
        args.pathSegments[idx].remainingBounces = 0;
        return;
    }

    bsdf = glm::clamp(bsdf, glm::vec3(0.0f), glm::vec3(1.0f));

    //assert(bsdf.x > 0 && bsdf.y > 0 && bsdf.z > 0);
    assert(!isnanVec3(bsdf));

    glm::vec3 directRadiance;
    glm::vec3 wiW_Li;
    float pdf_Li;

    float lambert = glm::abs(glm::dot(norW, wiW));
    args.pathSegments[idx].throughput *= (bsdf / pdf);
    args.pathSegments[idx].prevBounceSample.pdf = pdf;
    args.pathSegments[idx].prevBounceSample.matType = MT_MICROFACET_PBR;
    args.pathSegments[idx].ray = SpawnRay(view_point, wiW);
    args.pathSegments[idx].remainingBounces--;

    // Direct Light Sampling
    // Key difference using MIS: Accumulate direct lighting radiance here.
    cudaTextureObject_t envMapObj = args.envMaps ? args.envMaps[0] : 0;
    glm::vec3 throughput = args.pathSegments[idx].throughput;
    if (SolveDirectLighting(args.sceneData, envMapObj, norW, view_point, rng, directRadiance, wiW_Li, pdf_Li))
    {
        glm::vec3& whW = glm::normalize(woW + wiW_Li);
        float bsdf_pdf = pdfCookTorrance(norW, whW, woW, wiW_Li, roughness, metallic);
        float lambert_Li = glm::abs(glm::dot(norW, wiW_Li));

        // Assemble direct lighting components
        glm::vec3 directLightResult = args.pathSegments[idx].throughput * directRadiance * lambert_Li / pdf_Li;
        thisBounceRadiance += directLightResult * PowerHeuristic(1, pdf_Li, 1, bsdf_pdf);
        args.pathSegments[idx].Lo += thisBounceRadiance;
    }
}

#if ONLY_BSDF_SAMPLING
__global__ void skDiffuseSimple(ShadeKernelArgs args)
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

    args.pathSegments[idx].throughput *= lightTransportResult;
    args.pathSegments[idx].ray = SpawnRay(path.ray.origin + intersection.t * path.ray.direction, wi);
    args.pathSegments[idx].remainingBounces--;
}

__global__ void skSpecularSimple(ShadeKernelArgs args)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.num_paths)
        return;

    const PathSegment path = args.pathSegments[idx];
    const ShadeableIntersection intersection = args.shadeableIntersections[idx];
    const Material material = args.materials[GetMaterialIDFromSortKey(intersection.matSortKey)];

    HANDLE_MISS(idx, intersection, pathSegments);

    glm::vec3 wiW = glm::reflect(path.ray.direction, intersection.surfaceNormal);
    args.pathSegments[idx].throughput *= material.color;
    args.pathSegments[idx].ray = SpawnRay(path.ray.origin + intersection.t * path.ray.direction, wiW);
    args.pathSegments[idx].remainingBounces--;
}

#endif

#if DIRECT_SAMPLING
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
        glm::vec3 radiance;
        if (!SolveDirectLighting(args.sceneData, intersection, view_point, rng, radiance, wiW, pdf))
            continue;

        float cosTheta = glm::dot(wiW, intersection.surfaceNormal);
        if (cosTheta < FLT_EPSILON)
            continue;

        totalDirectLight += radiance * cosTheta / (NUM_SAMPLES * pdf);
    }
    totalDirectLight *= numLights;

    args.pathSegments[idx].throughput *= bsdf;
    glm::vec3 throughput = args.pathSegments[idx].throughput;

    args.pathSegments[idx].Lo += throughput * totalDirectLight;
    args.pathSegments[idx].remainingBounces = 0;
}
#endif

#if DIRECT_SAMPLING || ONLY_BSDF_SAMPLING
__global__ void skEmissiveSimple(ShadeKernelArgs args)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= args.num_paths)
        return;

    const PathSegment path = args.pathSegments[idx];
    const ShadeableIntersection intersection = args.shadeableIntersections[idx];
    const Material material = args.materials[GetMaterialIDFromSortKey(intersection.matSortKey)];

    HANDLE_MISS(idx, intersection, pathSegments);

    glm::vec3 throughput = args.pathSegments[idx].throughput;
    args.pathSegments[idx].Lo += (material.color * material.emittance) * throughput;
    args.pathSegments[idx].remainingBounces = 0; // Mark it for culling later
}
#endif

// By convention: MUST match the order of the MaterialType struct

#if MIS_SAMPLING
static ShadeKernel sKernels[] =
{
    skDiffuse,
    skSpecular,
    skEmissive,
    skMicrofacetPBR
};
#elif DIRECT_SAMPLING
static ShadeKernel sKernels[] =
{
    skDiffuseDirect,
    skSpecular,
    skEmissiveSimple,
    skMicrofacetPBR
};
#else ONLY_BSDF_SAMPLING
static ShadeKernel sKernels[] =
{
    skDiffuseSimple,
    skSpecular,
    skEmissiveSimple,
    skMicrofacetPBR
};
#endif

__host__ ShadeKernel getShadingKernelForMaterial(MaterialType mt)
{
    assert(mt < MT_COUNT);
    return sKernels[mt];
}