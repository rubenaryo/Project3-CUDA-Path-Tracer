#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(std::string filename);
    ~Scene();

    void InitDeviceMeshes();

    std::vector<Geom> geoms;
    std::vector<Light> lights;
    std::vector<Material> materials;
    std::vector<Mesh> meshes;
    std::vector<Mesh> deviceMeshes;
    std::vector<BVHNode> bvhNodes;
    RenderState state;
};

__host__ int loadGLTF(const std::string& relPath, std::vector<Mesh>& meshes, MaterialID matId);
