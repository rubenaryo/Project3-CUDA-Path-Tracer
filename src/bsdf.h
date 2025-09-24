#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <host_defines.h>

__device__ glm::vec3 squareToDiskConcentric(glm::vec2 xi);
__device__ glm::vec3 squareToHemisphereCosine(const glm::vec2& xi);
__device__ Ray SpawnRay(const glm::vec3& pos, const glm::vec3& wi);
__device__ float squareToHemisphereCosinePDF(const glm::vec3& sample);
__device__ void coordinateSystem(glm::vec3 v1, glm::vec3& v2, glm::vec3& v3);
__device__ glm::mat3 LocalToWorld(glm::vec3 nor);

// General shading kernel
__global__ void shadeMaterial(int iter, int num_paths, ShadeableIntersection* shadeableIntersections, PathSegment* pathSegments, Material* materials);