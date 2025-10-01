#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <thrust/random.h>

//__device__ glm::vec3 SolveMIS(ShadeableIntersection isect, const SceneData& sd, glm::vec3 view_point, glm::vec3 woW, const Material mat, thrust::default_random_engine& rng);
__device__ float PowerHeuristic(int nf, float fPdf, int ng, float gPdf);
__device__ float Pdf_Li(const Light& chosenLight, const glm::vec3& view_point, const glm::vec3& norW, const glm::vec3& wiW);

__device__ glm::vec3 Sample_Li(glm::vec3 view_point, glm::vec3 nor, const Light& chosenLight, int numLights, thrust::default_random_engine& rng, glm::vec3& out_wiW, float& out_pdf, float& out_distToLight);
