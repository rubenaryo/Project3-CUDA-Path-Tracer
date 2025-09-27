#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <thrust/random.h>

// TODO some of these don't exist anymore
__device__ glm::vec3 squareToDiskConcentric(glm::vec2 xi);
__device__ glm::vec3 squareToHemisphereCosine(const glm::vec2& xi);
__device__ Ray SpawnRay(const glm::vec3& pos, const glm::vec3& wi);
__device__ float squareToHemisphereCosinePDF(const glm::vec3& sample);
__device__ void coordinateSystem(const glm::vec3& v1, glm::vec3& v2, glm::vec3& v3);
__device__ glm::mat3 LocalToWorld(glm::vec3 nor);

// General shading kernel
struct ShadeKernelArgs
{
    int iter;
    int num_paths;
    ShadeableIntersection* shadeableIntersections;
    PathSegment* pathSegments;
    Material* materials;
    AreaLight* areaLights;
    int num_lights;
};

typedef void(*ShadeKernel)(ShadeKernelArgs args);

__host__ ShadeKernel getShadingKernelForMaterial(MaterialType mt);

__global__ void skDiffuse(ShadeKernelArgs args);
__global__ void skDiffuseDirect(ShadeKernelArgs args);
__global__ void skSpecular(ShadeKernelArgs args);
__global__ void skEmissive(ShadeKernelArgs args);
__global__ void skRefractive(ShadeKernelArgs args);

__device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng);
