#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <thrust/random.h>

__device__ float PowerHeuristic(int nf, float fPdf, int ng, float gPdf);
__device__ float Pdf_Rect(const glm::mat4& lightTfm, const glm::vec3& view_point, const glm::vec3& light_point, const glm::vec3& norW);

__device__ glm::vec3 Sample_Li(glm::vec3 view_point, glm::vec3 nor, const Light& chosenLight, int numLights, thrust::default_random_engine& rng, glm::vec3& out_wiW, float& out_pdf, float& out_distToLight);
