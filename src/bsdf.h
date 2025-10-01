#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <thrust/random.h>

// TODO some of these don't exist anymore
__device__ thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth);
__device__ glm::vec3 squareToDiskConcentric(glm::vec2 xi);
__device__ glm::vec3 squareToHemisphereCosine(const glm::vec2& xi);
__device__ Ray SpawnRay(const glm::vec3& pos, const glm::vec3& wi);
__device__ float squareToHemisphereCosinePDF(const glm::vec3& sample);
__device__ void coordinateSystem(const glm::vec3& v1, glm::vec3& v2, glm::vec3& v3);
__device__ glm::mat3 LocalToWorld(glm::vec3 nor);

// PDF functions
__device__ float Pdf(MaterialType matType, glm::vec3 norW, glm::vec3 woW, glm::vec3 wiW);

// Sample_f / f functions
__device__ glm::vec3 Sample_f_diffuse(const glm::vec3& albedo, const glm::vec3& norW, thrust::default_random_engine& rng, glm::vec3& out_wiW, float& out_pdf);

// General shading kernel
struct ShadeKernelArgs
{
    int iter;
    int num_paths;
    ShadeableIntersection* shadeableIntersections;
    PathSegment* pathSegments;
    Material* materials;
    SceneData sceneData;
};

typedef void(*ShadeKernel)(ShadeKernelArgs args);

__host__ ShadeKernel getShadingKernelForMaterial(MaterialType mt);