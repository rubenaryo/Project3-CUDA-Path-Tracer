#pragma once

#include "sceneStructs.h"

// Note: This reorders the underlying vertex data in Mesh!
__host__ bool BuildBVH(Mesh& mesh, std::vector<BVHNode>& out_nodes);