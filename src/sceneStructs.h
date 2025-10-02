#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

#define STREAM_COMPACTION 1
#define MATERIAL_SORT 1
#define STOCHASTIC_AA 1
#define USE_BVH 1

#define MIS_SAMPLING 1
#define DIRECT_SAMPLING (0 && !MIS_SAMPLING)
#define ONLY_BSDF_SAMPLING (0 && !MIS_SAMPLING && !DIRECT_SAMPLING)

typedef uint32_t LightID;
typedef uint16_t MaterialID;
typedef uint32_t MaterialSortKey;
static const MaterialSortKey SORTKEY_INVALID = UINT32_MAX;
static const MaterialID MATERIALID_INVALID = UINT16_MAX;

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
static inline MaterialSortKey BuildSortKey(MaterialType type, MaterialID id)
{
    return (MaterialSortKey)type << 16 | (MaterialSortKey)id;
}

__host__ __device__
static inline MaterialType GetMaterialTypeFromSortKey(MaterialSortKey key)
{
    return static_cast<MaterialType>(key >> 16);
}

__host__ __device__
static inline MaterialID GetMaterialIDFromSortKey(MaterialSortKey key)
{
    return static_cast<MaterialID>(key & 0xFFFF);
}

__host__ __device__ 
static inline void UnpackSortKey(MaterialSortKey key, MaterialType& out_type, MaterialID& out_id)
{
    out_type = GetMaterialTypeFromSortKey(key);
    out_id = GetMaterialIDFromSortKey(key); 
}


enum GeomType
{
    GT_INVALID,
    GT_SPHERE,
    GT_CUBE,
    GT_RECT,
    GT_TRIANGLE,
    GT_MESH
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
    MaterialSortKey matSortKey;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    int meshId = -1;
};

struct Mesh
{
    glm::vec3*  vtx = nullptr;
    glm::vec3*  nor = nullptr;
    glm::vec2*  uvs = nullptr;
    glm::uvec3* idx = nullptr;

    uint32_t vtx_count = 0;
    uint32_t nor_count = 0;
    uint32_t uvs_count = 0;
    uint32_t tri_count = 0;

    uint32_t bvh_root_idx = 0;
    bool isDevice = false;

    __host__ void allocate(uint32_t v, uint32_t n, uint32_t u, uint32_t t)
    {
        isDevice = false;

        vtx_count = v;
        nor_count = n;
        uvs_count = u;
        tri_count = t;

        if (v) vtx = (glm::vec3*) malloc(sizeof(glm::vec3)  * vtx_count);
        if (n) nor = (glm::vec3*) malloc(sizeof(glm::vec3)  * nor_count);
        if (u) uvs = (glm::vec2*) malloc(sizeof(glm::vec2)  * uvs_count);
        if (t) idx = (glm::uvec3*)malloc(sizeof(glm::uvec3) * tri_count);
    }

    __host__ void cleanup()
    {
        if (vtx) free(vtx);
        if (nor) free(nor);
        if (uvs) free(uvs);
        if (idx) free(idx);
    }

    __host__ void deviceAllocate(uint32_t v, uint32_t n, uint32_t u, uint32_t t)
    {
        isDevice = true;

        vtx_count = v;
        nor_count = n;
        uvs_count = u;
        tri_count = t;

        if (v) cudaMalloc(&vtx, sizeof(glm::vec3)  * vtx_count);
        if (n) cudaMalloc(&nor, sizeof(glm::vec3)  * nor_count);
        if (u) cudaMalloc(&uvs, sizeof(glm::vec2)  * uvs_count);
        if (t) cudaMalloc(&idx, sizeof(glm::uvec3) * tri_count);
    }

    __host__ void deviceCleanup()
    {
        if (vtx) cudaFree(vtx);
        if (nor) cudaFree(nor);
        if (uvs) cudaFree(uvs);
        if (idx) cudaFree(idx);
    }
};

struct Triangle
{
    glm::vec3 v[3];
};

struct AABB
{
    glm::vec3 min = glm::vec3(FLT_MAX);
    glm::vec3 max = glm::vec3(-FLT_MAX);
    glm::vec3 centre = glm::vec3(0.0f);
};

struct BVHNode
{
    AABB bounds;
    int32_t triIndex = -1;
    uint32_t triCount = 0;
    int32_t childIndex = -1;
};

struct Light
{
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;

    glm::vec3 color = glm::vec3(0.0f);
    
    float emittance;
    int geomId = -1;
    LightType lightType;
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

    int diffuseTexId = -1;
    int normalTexId = -1;
};

struct HostTextureHandle
{
    std::string filePath;
    cudaTextureObject_t texObj = 0;
    cudaArray_t cudaArr = nullptr;
    int width = 0;
    int height = 0;
};

struct BSDFSample
{
    float pdf = 0.0f;
    MaterialType matType = MT_INVALID;
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
    glm::vec3 throughput;
    glm::vec3 Lo; // accumulated radiance
    BSDFSample prevBounceSample; // data from the previous bounce (for MIS)
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
  MaterialSortKey matSortKey;
  int hitGeomIdx = -1;
};

struct BVHIntersectResult
{
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
    float t;
    uint32_t triIdx;
};

struct SceneData
{
    Geom* geoms;
    int geoms_size;
    Mesh* meshes;
    int meshes_size;
    Light* lights;
    int lights_size;
    BVHNode* bvhNodes;
    int bvhNodes_size;
};