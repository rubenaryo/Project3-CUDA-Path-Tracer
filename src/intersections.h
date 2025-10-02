#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>


/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

__global__ void generateSortKeys(int N, const ShadeableIntersection* isects, Material* mats, MaterialSortKey* flags);

__device__ float getRectArea(const glm::mat4& rectTfm);


// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ inline glm::vec3 getPointOnRay(Ray r, float t)
{
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v)
{
    return glm::vec3(m * v);
}

// Isect between an arbitrary ray and an AABB (for BVH)
__host__ __device__ float intersectAABB(const Ray& ray, const AABB& aabb);

__host__ __device__ float rectIntersectionTest(const Geom& geom, const Ray& ray, glm::vec3& out_isectPoint, glm::vec3& out_normal, glm::vec2& out_uv);
__host__ __device__ float boxIntersectionTest(Geom box, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside);
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside);
__host__ __device__ float triangleIntersectionTest(Geom tri, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside);
__host__ __device__ float meshIntersectionTest(Geom meshGeom, const SceneData& sd, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec2& uv, bool& outside);

__device__ void sceneIntersect(PathSegment& path, const SceneData& sceneData, ShadeableIntersection& result, cudaTextureObject_t* envMaps, int ignoreGeomId = -1);
__device__ void lightsIntersect(PathSegment& path, const Light* lights, int lights_size, ShadeableIntersection& result, LightID& resultId);