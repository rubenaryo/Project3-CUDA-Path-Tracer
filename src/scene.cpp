#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"
#include "tinygltf_include.h"
#include "bvh.h"

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

Scene::~Scene()
{
    int meshCount = meshes.size();
    for (int m = 0; m != meshCount; ++m)
    {
        meshes.at(m).cleanup();
        deviceMeshes.at(m).deviceCleanup();
    }

}

void Scene::InitDeviceMeshes()
{
    int meshCount = meshes.size();
    for (int m = 0; m != meshCount; ++m)
    {
        const Mesh& hostMesh = meshes.at(m);
        Mesh& deviceMesh = deviceMeshes.emplace_back();
        deviceMesh.bvh_root_idx = hostMesh.bvh_root_idx;

        uint32_t v = hostMesh.vtx_count;
        uint32_t n = hostMesh.nor_count;
        uint32_t u = hostMesh.uvs_count;
        uint32_t t = hostMesh.tri_count;

        deviceMesh.deviceAllocate(v, n, u, t);

        if (v) cudaMemcpy(deviceMesh.vtx, hostMesh.vtx, v * sizeof(glm::vec3), cudaMemcpyHostToDevice);
        if (n) cudaMemcpy(deviceMesh.nor, hostMesh.nor, n * sizeof(glm::vec3), cudaMemcpyHostToDevice);
        if (u) cudaMemcpy(deviceMesh.uvs, hostMesh.uvs, u * sizeof(glm::vec2), cudaMemcpyHostToDevice);
        if (t) cudaMemcpy(deviceMesh.idx, hostMesh.idx, t * sizeof(glm::uvec3), cudaMemcpyHostToDevice);
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
        Geom newGeom = {};

        MaterialID matId = MATERIALID_INVALID;
        auto findIt = MatNameToID.find(p["MATERIAL"]);
        if (findIt == MatNameToID.end())
        {
            // Material not found. Just assign the first material.
            matId = 0;
        }
        else
        {
            matId = findIt->second;
            assert(matId < materials.size());
        }

        const Material& mat = materials.at(matId);
        newGeom.matSortKey = BuildSortKey(mat.type, matId);

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
        else if (type == "rect")
        {
            newGeom.type = GT_RECT;
        }
        else if (type == "mesh")
        {
            // Only support one mesh for now
            if (!meshes.empty())
                continue;

            newGeom.type = GT_MESH;
            const auto& relPath = p["PATH"];
            int meshId = loadGLTF(relPath, meshes);

            if (meshId == -1)
                continue; // Mesh loading failed. Don't add it.

            Mesh& mesh = meshes.at(meshId);
            bool bvhSuccess = BuildBVH(mesh, bvhNodes);
            if (!bvhSuccess)
                printf("BuildBVH failure for %s!", std::string(relPath).c_str());

            newGeom.meshId = meshId;
        }

        // This is also an area light.
        if (mat.type == MT_EMISSIVE)
        {
            Light newLight = {};
            newLight.color = mat.color;
            newLight.emittance = mat.emittance;
            
            newLight.transform = newGeom.transform;
            newLight.inverseTransform = newGeom.inverseTransform;
            newLight.invTranspose = newGeom.inverseTransform;
            newLight.translation = newGeom.translation;
            newLight.rotation = newGeom.rotation;
            newLight.scale = newGeom.scale;

            newLight.lightType = LT_AREA;
            lights.push_back(newLight);
        }

        geoms.push_back(newGeom);
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

// Returns mesh id
__host__ int loadGLTF(const std::string& relPath, std::vector<Mesh>& meshes)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    std::string absolutePath;
    std::filesystem::path filePath(relPath);
    absolutePath = std::filesystem::absolute(filePath).string();

    bool result = loader.LoadASCIIFromFile(&model, &err, &warn, absolutePath);
    if (!result) 
        result = loader.LoadBinaryFromFile(&model, &err, &warn, absolutePath);

    if (!result) {
        std::cerr << "Failed to load glTF: " << absolutePath << std::endl;
        return -1;
    }

    // Collect all mesh data
    std::vector<glm::vec3> allVertices, allNormals;
    std::vector<glm::vec2> allUVs;
    std::vector<glm::uvec3> allIndices;
    uint32_t vertexOffset = 0;

    for (const auto& mesh : model.meshes) {
        for (const auto& primitive : mesh.primitives) {
            // Load vertices
            std::vector<glm::vec3> vertices;
            if (primitive.attributes.find("POSITION") != primitive.attributes.end()) {
                vertices = getBufferData<glm::vec3>(model, primitive.attributes.at("POSITION"));
            }
            if (vertices.empty()) continue;

            // Load normals and UVs
            auto normals = getBufferData<glm::vec3>(model,
                primitive.attributes.count("NORMAL") ? primitive.attributes.at("NORMAL") : -1);
            auto uvs = getBufferData<glm::vec2>(model,
                primitive.attributes.count("TEXCOORD_0") ? primitive.attributes.at("TEXCOORD_0") : -1);

            // Add to combined arrays
            allVertices.insert(allVertices.end(), vertices.begin(), vertices.end());

            if (normals.size()) 
                allNormals.insert(allNormals.end(), normals.begin(), normals.end());

            if (uvs.size()) 
                allUVs.insert(allUVs.end(), uvs.begin(), uvs.end());

            // Load and adjust indices
            if (primitive.indices >= 0) {
                const auto& accessor = model.accessors[primitive.indices];
                if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    auto indices16 = getBufferData<uint16_t>(model, primitive.indices);
                    for (size_t i = 0; i < indices16.size(); i += 3) {
                        allIndices.push_back(glm::uvec3(
                            indices16[i] + vertexOffset,
                            indices16[i + 1] + vertexOffset,
                            indices16[i + 2] + vertexOffset));
                    }
                }
                else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                    auto indices32 = getBufferData<uint32_t>(model, primitive.indices);
                    for (size_t i = 0; i < indices32.size(); i += 3) {
                        allIndices.push_back(glm::uvec3(
                            indices32[i] + vertexOffset,
                            indices32[i + 1] + vertexOffset,
                            indices32[i + 2] + vertexOffset));
                    }
                }
            }

            vertexOffset += vertices.size();
        }
    }

    if (allVertices.empty()) return -1;

    int meshId = meshes.size();
    Mesh& mesh = meshes.emplace_back();

    // Allocate host memory and copy data
    uint32_t v = allVertices.size();
    uint32_t n = allNormals.size();
    uint32_t u = allUVs.size();
    uint32_t t = allIndices.size();

    mesh.allocate(v, n, u, t);

    if (v)
    {
        memcpy(mesh.vtx, allVertices.data(), v * sizeof(glm::vec3));
    }

    if (n)
    {
        memcpy(mesh.nor, allNormals.data(), n * sizeof(glm::vec3));
    }
    
    if (u)
    {
        memcpy(mesh.uvs, allUVs.data(), u * sizeof(glm::vec2));
    }

    if (t)
    {
        memcpy(mesh.idx, allIndices.data(), t * sizeof(glm::uvec3));
    }

    return meshId;
}