#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"
#include "tinygltf_include.h"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = MT_DIFFUSE;
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
            newMaterial.type = MT_EMISSIVE;
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = MT_SPECULAR;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        Geom newGeom;
        newGeom.materialid = MatNameToID[p["MATERIAL"]];

        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        const auto& type = p["TYPE"];
        if (type == "cube")
        {
            newGeom.type = GT_CUBE;
        }
        else if (type == "sphere")
        {
            newGeom.type = GT_SPHERE;
        }
        else if (type == "mesh")
        {
            newGeom.type = GT_MESH;
            const auto& fileName = p["PATH"];
            bool loadResult = loadGLTF(fileName, meshes);
        }

        geoms.push_back(newGeom);
    }
    const auto& lightsData = data["Lights"];
    for (const auto& p : lightsData)
    {
        const auto& col = p["RGB"];

        Light newLight = {};
        newLight.emittance = p["EMITTANCE"];
        newLight.color = glm::vec3(col[0], col[1], col[2]);

        if (p["TYPE"] == "AREA")
        {
            newLight.type = LT_AREA;
        }

        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];

        newLight.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newLight.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newLight.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newLight.transform = utilityCore::buildTransformationMatrix(
            newLight.translation, newLight.rotation, newLight.scale);

        // CUBE not supported
        const auto& geom = p["GEOMTYPE"];
        if (geom == "RECT")
            newLight.geomType = GT_RECT;
        else if (geom == "SPHERE")
            newLight.geomType = GT_SPHERE;
        else
            newLight.geomType = GT_INVALID;

        newLight.id = lights.size();
        lights.push_back(newLight);
    }

    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

template<typename T>
std::vector<T> getBufferData(const tinygltf::Model& model, int accessorIndex)
{
    if (accessorIndex < 0) return {};

    const auto& accessor = model.accessors[accessorIndex];
    const auto& bufferView = model.bufferViews[accessor.bufferView];
    const auto& buffer = model.buffers[bufferView.buffer];

    const uint8_t* data = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;

    std::vector<T> result;
    result.resize(accessor.count);

    if (bufferView.byteStride == 0 || bufferView.byteStride == sizeof(T)) {
        // Tightly packed data
        std::memcpy(result.data(), data, accessor.count * sizeof(T));
    }
    else {
        // Strided data
        for (size_t i = 0; i < accessor.count; ++i) {
            std::memcpy(&result[i], data + i * bufferView.byteStride, sizeof(T));
        }
    }

    return result;
}

bool loadGLTF(const std::string& filename, std::vector<Mesh>& meshes)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
    if (!ret) {
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
    }

    if (!err.empty()) {
        std::cerr << "glTF error: " << err << std::endl;
    }
    if (!warn.empty()) {
        std::cerr << "glTF warning: " << warn << std::endl;
    }
    if (!ret) {
        std::cerr << "Failed to load glTF file: " << filename << std::endl;
        return false;
    }

    meshes.clear();

    for (const auto& mesh : model.meshes) {
        for (const auto& primitive : mesh.primitives) {
            Mesh meshData;

            if (primitive.attributes.find("POSITION") != primitive.attributes.end()) {
                int posAccessor = primitive.attributes.at("POSITION");
                meshData.vtx = getBufferData<glm::vec3>(model, posAccessor);
            }

            if (primitive.attributes.find("NORMAL") != primitive.attributes.end()) {
                int normalAccessor = primitive.attributes.at("NORMAL");
                meshData.nor = getBufferData<glm::vec3>(model, normalAccessor);
            }

            if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
                int uvAccessor = primitive.attributes.at("TEXCOORD_0");
                meshData.uv = getBufferData<glm::vec2>(model, uvAccessor);
            }

            if (primitive.indices >= 0) {
                const auto& accessor = model.accessors[primitive.indices];

                if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    auto indices16 = getBufferData<uint16_t>(model, primitive.indices);
                    meshData.idx.reserve(indices16.size() / 3);
                    for (size_t i = 0; i < indices16.size(); i += 3) {
                        meshData.idx.push_back(glm::uvec3(indices16[i], indices16[i + 1], indices16[i + 2]));
                    }
                }
                else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                    auto indices32 = getBufferData<uint32_t>(model, primitive.indices);
                    meshData.idx.reserve(indices32.size() / 3);
                    for (size_t i = 0; i < indices32.size(); i += 3) {
                        meshData.idx.push_back(glm::uvec3(indices32[i], indices32[i + 1], indices32[i + 2]));
                    }
                }
            }

            if (meshData.vtx.empty()) {
                std::cerr << "Warning: Mesh has no vertices" << std::endl;
                continue;
            }

            if (meshData.nor.empty() && !meshData.idx.empty()) {
                meshData.nor.resize(meshData.vtx.size(), glm::vec3(0.0f));

                for (const auto& tri : meshData.idx) {
                    glm::vec3 v0 = meshData.vtx[tri.x];
                    glm::vec3 v1 = meshData.vtx[tri.y];
                    glm::vec3 v2 = meshData.vtx[tri.z];

                    glm::vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

                    meshData.nor[tri.x] += normal;
                    meshData.nor[tri.y] += normal;
                    meshData.nor[tri.z] += normal;
                }

                for (auto& normal : meshData.nor) {
                    normal = glm::normalize(normal);
                }
            }

            // Ensure UVs exist
            if (meshData.uv.size() != meshData.vtx.size()) {
                meshData.uv.resize(meshData.vtx.size(), glm::vec2(0.0f));
            }

            meshes.push_back(std::move(meshData));
        }
    }

    std::cout << "Loaded " << meshes.size() << " meshes from " << filename << std::endl;
    return true;
}