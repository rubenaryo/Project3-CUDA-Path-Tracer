#include "bsdf.h"

#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#include <thrust/random.h>

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

inline __device__ glm::vec3 squareToDiskConcentric(glm::vec2 xi) {

    glm::vec2 corrected = 2.0f * xi - glm::vec2(1.0);
    float x = corrected.x;
    float y = corrected.y;

    // Div by zero guard
    if (abs(x) < 0.01 && abs(y) < 0.01)
        return glm::vec3(0);

    float r, theta;
    if (abs(x) > abs(y))
    {
        r = x;
        theta = (PI / 4.0) * (y / x);
    }
    else
    {
        r = y;
        theta = (PI / 2.0) - (PI / 4.0) * (x / y);
    }

    return r * glm::vec3(cos(theta), sin(theta), 0.0);
}


inline __device__ glm::vec3 squareToHemisphereCosine(const glm::vec2& xi)
{
    glm::vec3 diskCoord = squareToDiskConcentric(xi);
    diskCoord.z = glm::sqrt(glm::max(0.0f, (1.0f - diskCoord.x * diskCoord.x - diskCoord.y * diskCoord.y)));
    return diskCoord;
}

inline __device__ Ray SpawnRay(const glm::vec3& pos, const glm::vec3& wi)
{
    Ray r;
    r.origin = pos + wi * 0.001f;
    r.direction = wi;
    return r;
}

inline __device__ float squareToHemisphereCosinePDF(const glm::vec3& sample) {

    return sample.z * (1.0 / PI);
}

// TODO: Evaluate if there is a more performant way to do this..
inline __device__ void coordinateSystem(glm::vec3 v1, glm::vec3& v2, glm::vec3& v3) {
    if (abs(v1.x) > abs(v1.y))
        v2 = glm::vec3(-v1.z, 0, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z);
    else
        v2 = glm::vec3(0, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z);
    v3 = cross(v1, v2);
}

inline __device__ glm::mat3 LocalToWorld(glm::vec3 nor) {
    glm::vec3 tan, bit;
    coordinateSystem(nor, tan, bit);
    return glm::mat3(tan, bit, nor);
}

__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
            // Set up the RNG
            // LOOK: this is how you use thrust's RNG! Please look at
            // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                // Basically Sample_f() from 561
                //float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                //pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                //pathSegments[idx].color *= u01(rng); // apply some noise because why not

                PathSegment path = pathSegments[idx];
                Ray wo = path.ray;

                thrust::default_random_engine rng0 = makeSeededRandomEngine(iter, idx, 0);
                thrust::default_random_engine rng1 = makeSeededRandomEngine(iter, idx, 1);
                glm::vec2 xi = glm::vec2(u01(rng0), u01(rng1));

                glm::vec3 hemisphereCosWi = squareToHemisphereCosine(xi);
                float pdf = squareToHemisphereCosinePDF(hemisphereCosWi);
                pdf = glm::max(pdf, 0.001f); // for safety.

                glm::mat3 local2World = LocalToWorld(intersection.surfaceNormal);
                glm::vec3 wiW = local2World * hemisphereCosWi;
                glm::vec3 bsdf = materialColor / PI;

                pathSegments[idx].color *= (bsdf * glm::abs(glm::dot(wiW, intersection.surfaceNormal))) / pdf;
                pathSegments[idx].ray = SpawnRay(wo.origin + intersection.t * wo.direction, wiW);
                //glm::vec3 intersect = wo.origin + intersection.t * wo.direction;
                //scatterRay(pathSegments[idx], intersect, intersection.surfaceNormal, material, rng);
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}