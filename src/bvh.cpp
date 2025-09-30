#include "bvh.h"

// References: 
// https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
// https://www.youtube.com/watch?v=C1H4zIiCOaI

inline __host__ glm::vec3 GetTriCentroid(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2)
{
    return (v0 + v1 + v2) / 3.0f;
}

__host__ void GrowAABB(const glm::vec3* vertices, int vtx_count, AABB& aabb)
{
    for (int v = 0; v < vtx_count; ++v)
    {
        const glm::vec3& vtx = vertices[v];
        aabb.max = glm::max(aabb.max, vtx);
        aabb.min = glm::min(aabb.min, vtx);
    }

    aabb.centre = (aabb.min + aabb.max) * 0.5f;
}

#define MIN_TRIS_PER_LEAF 4
static int maxDepthTest = -1;

__host__ void Split(uint32_t parentIdx, std::vector<BVHNode>& allNodes, Mesh& mesh, int depth)
{
    maxDepthTest = std::max(maxDepthTest, depth);
    if (depth >= BVH_MAX_DEPTH)
        return;

    // Choose split axis
    BVHNode parentCopy = allNodes[parentIdx];
    int32_t parentTriIndex = parentCopy.triIndex;
    int32_t parentTriCount = parentCopy.triCount;

    glm::vec3 extent = allNodes[parentIdx].bounds.max - allNodes[parentIdx].bounds.min;
    uint32_t splitAxis = extent.x > glm::max(extent.y, extent.z) ? 0 : extent.y > extent.z ? 1 : 2;
    float splitPos = allNodes[parentIdx].bounds.centre[splitAxis];

    // Create child nodes
    uint32_t childAIdx = allNodes.size();
    uint32_t childBIdx = childAIdx + 1;

    BVHNode childA;
    BVHNode childB;

    childA.triIndex = parentTriIndex;
    childB.triIndex = parentTriIndex;

    // Assign parent's tris to each child
    const uint32_t maxTriIdx = (uint32_t)parentTriIndex + parentTriCount;
    for (uint32_t triIdx = parentTriIndex; triIdx < maxTriIdx; ++triIdx)
    {
        const uint32_t vertIdx = triIdx * 3;
        const glm::vec3 v0 = mesh.vtx[vertIdx];
        const glm::vec3 v1 = mesh.vtx[vertIdx+1];
        const glm::vec3 v2 = mesh.vtx[vertIdx+2];
        const glm::vec3 centroid = GetTriCentroid(v0, v1, v2);

        bool isSideA = centroid[splitAxis] < splitPos;
        BVHNode& child = isSideA ? childA : childB;
        GrowAABB(mesh.vtx + vertIdx, 3, child.bounds);

        child.triCount++;

        if (isSideA)
        {
            int32_t swapTriIdx = childA.triIndex + childA.triCount - 1;
            int32_t swapVertIdx = swapTriIdx * 3;

            // Swap tris to ensure data cohesion per node
            mesh.vtx[vertIdx]   = mesh.vtx[swapVertIdx];
            mesh.vtx[vertIdx+1] = mesh.vtx[swapVertIdx+1];
            mesh.vtx[vertIdx+2] = mesh.vtx[swapVertIdx+2];

            mesh.vtx[swapVertIdx]   = v0;
            mesh.vtx[swapVertIdx+1] = v1;
            mesh.vtx[swapVertIdx+2] = v2;

            childB.triIndex++; // Every time we add to childA, we move over the start of childB
        }
    }

    if (childA.triCount > 0 && childB.triCount > 0)
    {
        allNodes[parentIdx].childIndex = childAIdx;
        allNodes.push_back(childA);
        allNodes.push_back(childB);
        Split(childAIdx, allNodes, mesh, depth + 1);
        Split(childBIdx, allNodes, mesh, depth + 1);
    }
}

__host__ bool BuildBVH(Mesh& mesh, std::vector<BVHNode>& allNodes)
{
    // For sanity, yell if the mesh has a weird number of verts
    assert(mesh.vtx_count % 3 == 0);
    uint32_t totalTriCount = mesh.vtx_count / 3;

    // Assume only one mesh in the collection for now.
    assert(allNodes.empty());
    allNodes.clear();
    allNodes.reserve(totalTriCount * 2 - 1);

    uint32_t rootIdx = allNodes.size();
    mesh.bvh_root_idx = rootIdx;
    allNodes.emplace_back();
    allNodes[rootIdx].triCount = totalTriCount;
    allNodes[rootIdx].triIndex = 0;

    // Root node contains the whole mesh
    GrowAABB(mesh.vtx, mesh.vtx_count, allNodes[rootIdx].bounds);

    // Split the root at first
    int depth = 0;
    Split(mesh.bvh_root_idx, allNodes, mesh, depth);

    allNodes.shrink_to_fit(); // TODO: Remove this once we add support for more than one mesh.
    return true;
}