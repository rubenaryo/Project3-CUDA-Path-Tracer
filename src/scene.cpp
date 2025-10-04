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
    for (HostTextureHandle& h : textures)
    {
        if (h.texObj)
            cudaDestroyTextureObject(h.texObj);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;

    auto itFindEnv = data.find("Environment");
    if (itFindEnv != data.end())
    {
        std::string relPath = itFindEnv.value();
        std::filesystem::path filePath(relPath);
        std::string absoluteStr = std::filesystem::absolute(filePath).string();
        std::filesystem::path absolutePath(relPath);

        if (std::filesystem::exists(absolutePath))
        {
            // The env map exists. hold onto the absolute path for later.
            envMapHandle.filePath = std::move(absoluteStr);
        }
        else
        {
            printf("Error: Environment map path does not exist for %s!\n", absoluteStr.c_str());
        }
    }

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
        else if (p["TYPE"] == "MicrofacetPBR")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = MT_MICROFACET_PBR;

            newMaterial.roughness = glm::clamp((float)p["ROUGHNESS"], MIN_ROUGHNESS, 1.0f);
            newMaterial.metallic  = glm::clamp((float)p["METALLIC"], MIN_METALLIC, 1.0f);
        }

        auto TryLoadAssignTexture = [p](const char* attrName, std::vector<HostTextureHandle>& out_handleArr, bool sRGB, int& out_id)
        {
            if (p.find(attrName) != p.end())
            {
                std::string relPath = p[attrName];
                std::filesystem::path filePath(relPath);
                std::string absoluteStr = std::filesystem::absolute(filePath).string();
                std::filesystem::path absolutePath(relPath);

                if (!std::filesystem::exists(absolutePath))
                {
                    printf("Path does not exist for %s!\n", absoluteStr.c_str());
                    return false; // We needed this, but it doesn't exist.
                }

                out_id = out_handleArr.size();
                HostTextureHandle& handle = out_handleArr.emplace_back();
                handle.filePath = std::move(absoluteStr);
                handle.sRGB = sRGB;
            }
            return true; // We either don't need it, or do and it exists.
        };

        if (!TryLoadAssignTexture("DIFFUSE", textures, false, newMaterial.diffuseTexId))
            continue;

        if (!TryLoadAssignTexture("NORMAL", textures, false, newMaterial.normalTexId))
            continue;

        if (!TryLoadAssignTexture("METALLIC_ROUGHNESS", textures, false, newMaterial.metallicRoughTexId))
            continue;

        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        Geom newGeom = {};
        int geomId = geoms.size();

        MaterialID matId = 0;

        if (!p["MATERIAL"].is_array()) // Legacy default case, just a raw string
        {
            auto findIt = MatNameToID.find(p["MATERIAL"]);
            if (findIt != MatNameToID.end())
            {
                matId = findIt->second;
                assert(matId < materials.size());
            }
        }

        // If a mesh has multiple materials, this will be redone later
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
            newGeom.type = GT_MESH;
            const auto& relPath = p["PATH"];

            std::vector<std::string> materialsRequested = p.at("MATERIAL").get<std::vector<std::string>>();
            std::vector<MaterialID> materialIdsRequested;
            materialIdsRequested.reserve(materialsRequested.size());
            for (const std::string& matStr : materialsRequested)
            {
                MaterialID result = 0;
                auto findIt = MatNameToID.find(matStr);
                if (findIt != MatNameToID.end())
                {
                    result = findIt->second;
                    assert(result < materials.size());
                }
                materialIdsRequested.push_back(result);
            }

            bool loadSuccess = loadGLTF(relPath, materialIdsRequested, newGeom);
            if (!loadSuccess)
            {
                std::string relPathStr = relPath;
                printf("Critical Error: Failed to load GLTF at %s!\n", relPathStr.c_str());
                return;
            }

            // Mesh shaped lights are not supported. We're done here.
            // The resulting Geoms from this should have been created inside loadGLTF.
            continue;
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
            newLight.geomId = geomId;

            newLight.geomType = newGeom.type;
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
__host__ bool Scene::loadGLTF(const std::string& relPath, const std::vector<MaterialID>& materialIdsRequested, Geom geomTemplate)
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
        return false;
    }

    assert(materialIdsRequested.size() <= model.materials.size());

    int numMaterials = materialIdsRequested.size();

    std::vector<MeshData> materialToMeshData;
    materialToMeshData.resize(numMaterials);

    uint32_t vertexOffset = 0;

    // Grading Note: Used AI Help for this part
    for (const auto& mesh : model.meshes) {
        for (const auto& primitive : mesh.primitives) {

            if (primitive.material >= numMaterials) 
                continue; // This material index is greater than what we requested.

            MeshData& meshData = materialToMeshData.at(primitive.material);

            std::vector<glm::vec3>&  allVertices = meshData.vertices;
            std::vector<glm::vec3>&  allNormals = meshData.normals;
            std::vector<glm::vec3>&  allTangents = meshData.tangents;
            std::vector<glm::vec2>&  allUVs = meshData.uvs;
            std::vector<glm::uvec3>& allIndices = meshData.indices;

            // Load vertices
            std::vector<glm::vec3> vertices;
            if (primitive.attributes.find("POSITION") != primitive.attributes.end()) {
                vertices = getBufferData<glm::vec3>(model, primitive.attributes.at("POSITION"));
            }
            if (vertices.empty()) continue;

            // Load normals and UVs
            auto normals = getBufferData<glm::vec3>(model,
                primitive.attributes.count("NORMAL") ? primitive.attributes.at("NORMAL") : -1);
            auto tangents = getBufferData<glm::vec3>(model,
                primitive.attributes.count("TANGENT") ? primitive.attributes.at("TANGENT") : -1);
            auto uvs = getBufferData<glm::vec2>(model,
                primitive.attributes.count("TEXCOORD_0") ? primitive.attributes.at("TEXCOORD_0") : -1);

            // Add to combined arrays
            allVertices.insert(allVertices.end(), vertices.begin(), vertices.end());

            if (normals.size()) 
                allNormals.insert(allNormals.end(), normals.begin(), normals.end());

            if (tangents.size())
                allTangents.insert(allTangents.end(), tangents.begin(), tangents.end());

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
    
    // For building the bvh later
    std::vector<glm::uvec2> startEndIndices;
    startEndIndices.resize(materialToMeshData.size());

    uint32_t geomsStart = geoms.size();

    for (int m = 0; m != numMaterials; ++m)
    {
        // For each material in this file, build a bvh and record as a separate geometry
        MeshData& meshData = materialToMeshData.at(m);
        glm::uvec2& startEndIndex = startEndIndices.at(m);
        MaterialID matId = materialIdsRequested.at(m);
        const Material& mat = materials.at(matId);

        if (meshData.indices.empty()) // Mesh only has vertices.
        {
            meshData.indices.reserve(meshData.vertices.size()/3);
            
            for (int i = 0; (i+3) < meshData.vertices.size();)
            {
                meshData.indices.emplace_back(i++, i++, i++);
            }
        }

        bool hasNormals = true;
        if (meshData.normals.size() < meshData.vertices.size()) // no normals, or not as many
        {
            hasNormals = false;
            meshData.normals.resize(meshData.vertices.size(), glm::vec3(0.0f, 0.0f, 1.0f));
        }

        bool hasTangents = true;
        if (meshData.tangents.size() < meshData.vertices.size()) // no normals, or not as many
        {
            hasTangents = false;
            meshData.tangents.resize(meshData.vertices.size(), glm::vec3(0.0f, 1.0f, 0.0f));
        }

        bool hasUVs = true;
        if (meshData.uvs.size() < meshData.vertices.size())
        {
            hasUVs = false;
            meshData.uvs.resize(meshData.vertices.size(), glm::vec2(-1.0f, -1.0f));
        }
        
        // Add the mesh data to the master lists
        uint32_t startIdx = masterMeshData.indices.size();
        uint32_t endIdx = startIdx + meshData.indices.size();
        startEndIndex.x = startIdx;
        startEndIndex.y = endIdx;
        masterMeshData.Append(meshData);

        Geom& newGeom = geoms.emplace_back(geomTemplate);
        newGeom.matSortKey = BuildSortKey(mat.type, matId);
        newGeom.hasNormals = hasNormals;
        newGeom.hasTangents = hasTangents;
        newGeom.hasUVs = hasUVs;
    }

    // Build the BVH in a second pass so that we know all the vertices are loaded.
    for (int m = 0; m != numMaterials; ++m)
    {
        Geom& geom = geoms.at(geomsStart + m);
        glm::uvec2& startEndIdx = startEndIndices.at(m);

        geom.bvhRootIdx = BuildBVH(masterMeshData, startEndIdx.x, startEndIdx.y, bvhNodes);
    }

    return true;
}