#pragma once

#include "sceneStructs.h"

static const int BVH_MAX_DEPTH = 32;

// Note: This reorders the underlying vertex data in Mesh!
__host__ uint32_t BuildBVH(MeshData& meshData, uint32_t startIndex, uint32_t endIndex, std::vector<BVHNode>& allNodes);
