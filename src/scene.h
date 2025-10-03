#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    __host__ bool loadGLTF(const std::string& relPath, const std::vector<MaterialID>& materialIdsRequested, Geom geomTemplate);

public:
    Scene(std::string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Light> lights;
    std::vector<Material> materials;
    std::vector<BVHNode> bvhNodes;

    // To reduce GPU global memory reads, we hold all vertex/index/UV information in massive lists.
    // Each mesh geometry just has an index range into allIndices of where to start/end reading from.
    MeshData masterMeshData;

    std::vector<HostTextureHandle> textures;
    HostTextureHandle envMapHandle;

    RenderState state;
};