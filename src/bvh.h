#pragma once

#include "sceneStructs.h"

static const int BVH_MAX_DEPTH = 32;

// Warning: When called from the device, v is typically global memory!
__host__ __device__ inline Triangle GetTriangleFromTriIdx(uint32_t triIndex, const glm::vec3* v)
{
    uint32_t vi = triIndex * 3;
    return Triangle({v[vi],v[vi+1],v[vi+2]});
}

// Note: This reorders the underlying vertex data in Mesh!
__host__ bool BuildBVH(Mesh& mesh, std::vector<BVHNode>& out_nodes);
