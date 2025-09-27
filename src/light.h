#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <thrust/random.h>

__device__ glm::vec3 Sample_Li(glm::vec3 view_point, glm::vec3 nor, AreaLight* areaLights, int numLights, thrust::default_random_engine& rng, glm::vec3& out_wiW, float& out_pdf);