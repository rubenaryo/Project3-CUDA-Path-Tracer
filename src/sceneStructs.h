#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

#define STREAM_COMPACTION 1
#define MATERIAL_SORT 1

typedef uint32_t LightID;
typedef uint16_t MaterialID;
typedef uint32_t MaterialSortKey;
static const MaterialSortKey SORTKEY_INVALID = UINT32_MAX;

enum MaterialType : uint16_t
{
    MT_DIFFUSE = 0,
    MT_SPECULAR,
    MT_EMISSIVE,
    MT_REFRACTIVE,

    MT_COUNT,
    MT_FIRST = 0,
    MT_LAST = MT_COUNT - 1,
    MT_INVALID = UINT16_MAX,
};

template<MaterialType t>
struct MaterialTypePred
{
    __host__ __device__
        bool operator()(MaterialType type) { return type == t; }
};

template<MaterialType t>
struct NotMaterialTypePred
{
    __host__ __device__
        bool operator()(MaterialType type) { return type != t; }
};

__host__ __device__
static MaterialSortKey BuildSortKey(MaterialType type, MaterialID id)
{
    return (MaterialSortKey)type << 16 | (MaterialSortKey)id;
}

enum GeomType
{
    GT_INVALID,
    GT_SPHERE,
    GT_CUBE,
    GT_RECT
};

enum LightType
{
    LT_INVALID,
    LT_AREA
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    MaterialID materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Light
{
    glm::mat4 transform;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;

    glm::vec3 color = glm::vec3(0.0f);
    //glm::vec2 extents = glm::vec2(0.0f); // For rectangular area lights
    
    float emittance;
    LightID id;
    LightType type;
    GeomType geomType;
};

struct Material
{
    glm::vec3 color;
    struct
    {
        glm::vec3 color;
        float exponent;
    } specular;

    MaterialType type = MT_INVALID;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;

};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  MaterialID materialId;
};
