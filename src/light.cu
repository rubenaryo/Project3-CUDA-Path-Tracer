#include "light.h"

#include "bsdf.h"
#include "intersections.h"
#include <cmath>

__device__ glm::vec3 SolveMIS(ShadeableIntersection isect, const SceneData& sd, glm::vec3 view_point, glm::vec3 woW, const Material mat, thrust::default_random_engine& rng)
{
    // Direct Sampling
    glm::vec3 norW = isect.surfaceNormal;
    glm::vec3 wiW_Li;
    float distToLight_Li;
    float pdf_Li;
    glm::vec3 bsdf_Li = Sample_Li(view_point, norW, sd.lights, sd.lights_size, rng, wiW_Li, pdf_Li, distToLight_Li);

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
    bsdfRay.ray = SpawnRay(view_point, wiW_bsdf);
    float bsdf_absDot = glm::abs(glm::dot(wiW_bsdf, norW));

    ShadeableIntersection bsdfIsect;
    sceneIntersect(bsdfRay, sd, bsdfIsect);

    glm::vec3 bsdf_result = bsdfRay.color * bsdf * bsdf_absDot / pdf_bsdf;

    // Cross-method PDF
    //float pdf_Li_bsdf = squareToHemisphereCosinePDF(wiW_Li); // Li ray with respect to bsdf TODO: This only works for diffuse
    //float pdf_bsdf_Li = Pdf_Li(view_point, nor, wiW_bsdf, lightIdx); // bsdf ray with respect to light
    //
    //float w = 0.5, wg = 0.5;
    //if (pdf_bsdf > 0.001 && pdf_bsdf_Li > 0.001)
    //    w = PowerHeuristic(1, pdf_bsdf, 1, pdf_bsdf_Li);
    //
    //if (pdf_Li > 0.001 && pdf_bsdf_Li > 0.001)
    //    wg = PowerHeuristic(1, pdf_Li, 1, pdf_Li_bsdf);
    //
    //// Assemble final LTE weighted sum
    ////return Li_result;
    //return (bsdf_result * w) + (Li_result * wg);

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

__device__ glm::vec3 DirectSampleAreaLight(glm::vec3 view_point, glm::vec3 view_nor, const Light& chosenLight, int numLights, thrust::default_random_engine& rng, glm::vec3& out_wiW, float& out_pdf, float& out_distToLight)
{
    GeomType gt = chosenLight.geomType;

    switch (gt)
    {
    case GT_RECT:
        {
            float scaleX = chosenLight.scale.x;
            float scaleZ = chosenLight.scale.z;
            thrust::uniform_real_distribution<float> u01(-0.5f, 0.5f);
            glm::vec4 randPosLocal(u01(rng), u01(rng), u01(rng), 0.0f);
            randPosLocal = glm::normalize(randPosLocal);
            randPosLocal.w = 1.0f;
            glm::vec4 norLocal(0.0f, 0.0f, 1.0f, 0.0f);

            glm::vec3 randPosWorld(chosenLight.transform * randPosLocal);
            glm::vec3 norWorld(chosenLight.transform * norLocal);

            float surfaceArea = (scaleX) * (scaleZ);
            float areaPDF = 1.0f / surfaceArea;

            glm::vec3 lightToSurface = view_point - randPosWorld;
            float r2 = glm::dot(lightToSurface, lightToSurface);
            out_distToLight = glm::sqrt(r2);
            
            //if (r2 < FLT_EPSILON)
            //{
            //	out_pdf = 0.0f;
            //	out_wiW = glm::normalize(-lightToSurface);
            //	return glm::vec3(0.0f);
            //}

            lightToSurface *= glm::inversesqrt(r2); // normalize
            out_wiW = -lightToSurface;
            float cosTheta = glm::abs(glm::dot(norWorld, lightToSurface));

            out_pdf = (r2 / cosTheta) * areaPDF;

            return cosTheta * chosenLight.emittance * chosenLight.color;
        }
        break;
    default:
        // TODO
    }

    float Le = chosenLight.emittance;
    return Le * numLights * chosenLight.color;
}

__device__ glm::vec3 Sample_Li(glm::vec3 view_point, glm::vec3 nor, Light* lights, int numLights, thrust::default_random_engine& rng, glm::vec3& out_wiW, float& out_pdf, float& out_distToLight)
{
    thrust::uniform_int_distribution<int> iu0N(0, numLights-1);
    int randomLightIndex = iu0N(rng);
    const Light chosenLight = lights[randomLightIndex];

    return DirectSampleAreaLight(view_point, nor, chosenLight, numLights, rng, out_wiW, out_pdf, out_distToLight);
}
