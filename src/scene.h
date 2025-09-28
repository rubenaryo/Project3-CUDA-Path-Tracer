#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Light> lights;
    std::vector<Material> materials;
    std::vector<Mesh> meshes;
    RenderState state;
};

bool loadGLTF(const std::string& filename, std::vector<Mesh>& meshes);
