#include "light.h"

//#include "bsdf.h"
#include "intersections.h"
#include <cmath>

__device__ float PowerHeuristic(int nf, float fPdf, int ng, float gPdf) {
    float f = nf * fPdf;
    float g = ng * gPdf;
    return (f * f) / (f * f + g * g);
}

__device__ bool areaLightIntersect(const Light& chosenLight, Ray r, ShadeableIntersection& out_isect)
{
    out_isect.t = FLT_MAX;
    switch (chosenLight.geomType)
    {
    case GT_RECT:
    {
        glm::vec3 pos(0.0f);
        glm::vec3 nor(0.0f, 0.0f, 1.0f);
        glm::vec2 halfSideLengths = glm::vec2(chosenLight.scale.x, chosenLight.scale.z); // TODO: Maybe get this from the light's scale?
        glm::vec3 toLightLocal;
        glm::vec2 uv;
        float d = rectIntersectionTest(pos, nor,
            halfSideLengths.x, halfSideLengths.y,
            r, chosenLight.inverseTransform, toLightLocal, uv);

        if (d > (FLT_MAX - FLT_EPSILON))
            return false;

        glm::vec3 toLightWorld = multiplyMV(chosenLight.inverseTransform, glm::vec4(toLightLocal, 1.0f));

        out_isect.t = glm::length(toLightWorld); // TODO: This feels wrong, isn't d in local space?
        out_isect.surfaceNormal = glm::vec3(glm::normalize(chosenLight.invTranspose * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f)));
        // out_isect.Le = light.Le;
        // out_isect.obj_ID = light.ID;
    }
        break;
    default:
        // Unsupported
        break;
    }

    return out_isect.t < (FLT_MAX - FLT_EPSILON);
}

//__device__ float Pdf_Rect(const glm::vec3& halfSideLengths, const glm::vec3& view_point, const glm::vec3& light_point)

__device__ float Pdf_Rect(const Light& chosenLight, const glm::vec3& view_point, const glm::vec3& light_point, const glm::vec3& norW)
{
    using namespace glm;

    float scaleX = chosenLight.scale.x;
    float scaleZ = chosenLight.scale.z;

    float surfaceArea = (scaleX) * (scaleZ);
    float areaPDF = 1.0 / surfaceArea;

    vec3 lightToSurface = view_point - light_point;
    vec3 normalizedLightToSurface = normalize(lightToSurface);
    //vec3 norWorld = (chosenLight.transform * vec4(nor, 0.0)).xyz;
    float cosTheta = abs(dot(normalize(norW), normalizedLightToSurface));
    float r = length(lightToSurface);

    //if (cosTheta < 0.01)
    //    return 0.0;

    return (r * r / cosTheta) * areaPDF;
}

__device__ float Pdf_Li(const Light& chosenLight, const glm::vec3& view_point, const glm::vec3& norW, const glm::vec3& wiW)
{
    //Ray ray = SpawnRay(view_point, wiW);
    Ray ray;
    ray.direction = wiW;
    ray.origin = view_point + wiW * 0.001f;

    ShadeableIntersection isect;
    if (!areaLightIntersect(chosenLight, ray, isect))
        return 0.0f; // Didn't hit anything.

    glm::vec3 light_point = ray.origin + isect.t * wiW;

    switch (chosenLight.geomType)
    {
    case GT_RECT:
        return Pdf_Rect(chosenLight, view_point, light_point, norW);
        break;
    }

    return 0.0f;
}

__device__ glm::vec3 DirectSampleAreaLight(glm::vec3 view_point, glm::vec3 view_nor, const Light& chosenLight, int numLights, thrust::default_random_engine& rng, glm::vec3& out_wiW, float& out_pdf, float& out_distToLight)
{
    GeomType gt = chosenLight.geomType;

    switch (gt)
    {
    case GT_RECT:
        {
            thrust::uniform_real_distribution<float> uH(-0.5f, 0.5f);
            glm::vec4 randPosLocal(uH(rng), uH(rng), 0.0f, 1.0f);
            glm::vec4 norLocal(0.0f, 0.0f, 1.0f, 0.0f);

            glm::vec3 randPosWorld(chosenLight.transform * randPosLocal);
            glm::vec3 norWorld(chosenLight.transform * norLocal);

            glm::vec3 edge1 = glm::vec3(chosenLight.transform * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
            glm::vec3 edge2 = glm::vec3(chosenLight.transform * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));

            // Area is magnitude of cross product
            float surfaceArea = glm::length(glm::cross(edge1, edge2));
            if (surfaceArea < FLT_MAX)
            {
                // surface area is too small, early out.
                out_pdf = 0.0f;
                return glm::vec3(0.0f);
            }

            glm::vec3 toLightW = view_point - randPosWorld;
            float distToLightSq = glm::dot(toLightW, toLightW);
            float disttoLight = glm::sqrt(distToLightSq);
            
            if (distToLightSq < FLT_EPSILON)
            {
            	out_pdf = 0.0f;
            	out_wiW = glm::normalize(-toLightW);
            	return glm::vec3(0.0f);
            }

            toLightW *= glm::inversesqrt(distToLightSq); // normalize
            float cosTheta = (glm::dot(norWorld, toLightW));
            if (cosTheta < 0.0f)
            {
                out_pdf = 0.0f;
                return glm::vec3(0.0f);
            }
            
            out_distToLight = disttoLight;
            out_wiW = toLightW;
            out_pdf = distToLightSq / (cosTheta * surfaceArea);

            return numLights * chosenLight.emittance * chosenLight.color;
        }
        break;
    default:
        // TODO
    }

    float Le = chosenLight.emittance;
    return Le * numLights * chosenLight.color;
}

__device__ glm::vec3 Sample_Li(glm::vec3 view_point, glm::vec3 nor, const Light& chosenLight, int numLights, thrust::default_random_engine& rng, glm::vec3& out_wiW, float& out_pdf, float& out_distToLight)
{
    return DirectSampleAreaLight(view_point, nor, chosenLight, numLights, rng, out_wiW, out_pdf, out_distToLight);
}
