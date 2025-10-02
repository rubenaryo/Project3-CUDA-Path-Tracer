#include "light.h"

//#include "bsdf.h"
#include "intersections.h"
#include <cmath>

__device__ float PowerHeuristic(int nf, float fPdf, int ng, float gPdf) {
    float f = nf * fPdf;
    float g = ng * gPdf;
    return (f * f) / (f * f + g * g);
}

__device__ float Pdf_Rect(const glm::mat4& lightTfm, const glm::vec3& view_point, const glm::vec3& light_point, const glm::vec3& norW)
{
    using namespace glm;

    float surfaceArea = getRectArea(lightTfm);
    float areaPDF = 1.0 / surfaceArea;

    vec3 toLight = light_point - view_point;
    float r2 = glm::dot(toLight, toLight);
    vec3 toLightNormalized = normalize(toLight);
    float cosTheta = abs(dot(normalize(norW), -toLightNormalized));
    float r = sqrt(r2);

    if (cosTheta < FLT_EPSILON)
        return 0.0;

    return areaPDF * r2 / cosTheta;
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
            glm::vec3 norWorld(chosenLight.invTranspose * norLocal);

            glm::vec3 edge1 = glm::vec3(chosenLight.transform * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
            glm::vec3 edge2 = glm::vec3(chosenLight.transform * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));

            // Area is magnitude of cross product
            float surfaceArea = glm::length(glm::cross(edge1, edge2));
            if (surfaceArea < FLT_EPSILON)
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

            toLightW /= disttoLight; // normalize
            float cosTheta = (glm::dot(norWorld, -toLightW));
            if (cosTheta < 0.0f)
            {
                out_pdf = 0.0f;
                return glm::vec3(0.0f);
            }
            
            out_distToLight = disttoLight;
            out_wiW = -toLightW;
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
    switch (chosenLight.lightType)
    {
    case LT_AREA:
        return DirectSampleAreaLight(view_point, nor, chosenLight, numLights, rng, out_wiW, out_pdf, out_distToLight);
    }

    return glm::vec3(0.0f);
}
