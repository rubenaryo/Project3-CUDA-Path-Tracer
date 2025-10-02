#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "bsdf.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static Light* dev_lights = NULL;
static Mesh* dev_meshes = NULL;
static BVHNode* dev_bvhNodes = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static MaterialSortKey* dev_sortKeys = NULL;  // Parallel array of flags to mark material type.

// TODO: static variables for device memory, any extra info you need, etc
// ...

// 1D block for path tracing
static const int BLOCK_SIZE_1D = 128;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Light));
    cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Light), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_meshes, scene->deviceMeshes.size() * sizeof(Mesh)); // Only support one mesh for now.
    cudaMemcpy(dev_meshes, scene->deviceMeshes.data(), scene->deviceMeshes.size() * sizeof(Mesh), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_bvhNodes, scene->bvhNodes.size() * sizeof(BVHNode));
    cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(), scene->bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_sortKeys, pixelcount * sizeof(MaterialSortKey));
    thrust::fill(thrust::device, dev_sortKeys, dev_sortKeys + pixelcount, SORTKEY_INVALID);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_lights);
    cudaFree(dev_meshes);
    cudaFree(dev_bvhNodes);
    cudaFree(dev_intersections);
    cudaFree(dev_sortKeys);

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= cam.resolution.x || y >= cam.resolution.y)
        return;

    int index = x + (y * cam.resolution.x);
    PathSegment& segment = pathSegments[index];

    segment.ray.origin = cam.position;
    segment.throughput = glm::vec3(1.0f, 1.0f, 1.0f);
    segment.Lo = glm::vec3(0.0f);

#if STOCHASTIC_AA
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
    thrust::uniform_real_distribution<float> uH(-0.5f, 0.5f);

    segment.ray.direction = glm::normalize(cam.view
        - cam.right * cam.pixelLength.x * ((float)x + uH(rng) - (float)cam.resolution.x * 0.5f)
        - cam.up * cam.pixelLength.y * ((float)y + uH(rng) - (float)cam.resolution.y * 0.5f)
    );
#else
    segment.ray.direction = glm::normalize(cam.view
        - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
        - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
    );
#endif

    segment.pixelIndex = index;
    segment.remainingBounces = traceDepth;
}

__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    const SceneData sceneData,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index >= num_paths)
        return;
    
    PathSegment& path = pathSegments[path_index];
    ShadeableIntersection& result = intersections[path_index];
    sceneIntersect(path, sceneData, result);
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];

        const glm::vec3 RADIANCE_UPPER_BOUND(1000000000000.0f);
        assert(glm::all(glm::lessThan(iterationPath.Lo, RADIANCE_UPPER_BOUND)));
        image[iterationPath.pixelIndex] += glm::clamp(iterationPath.Lo, glm::vec3(0.0f), RADIANCE_UPPER_BOUND);
    }
}

//struct MaterialIdComp {
//    
//    using PathIsectTuple = thrust::tuple<PathSegment, ShadeableIntersection>;
//    
//    __host__ __device__
//    bool operator()(const PathIsectTuple& a, const PathIsectTuple& b) const {
//        return thrust::get<1>(a).materialId < thrust::get<1>(b).materialId;
//    }
//};
struct NonTerminated {
    __host__ __device__
        bool operator()(PathSegment& ps) {
        return ps.remainingBounces > 0;
    }
};

struct Terminated {
    __host__ __device__
        bool operator()(const PathSegment& ps) const {
        return ps.remainingBounces == 0;
    }
};

__global__ void testKernel(int N, PathSegment* paths, ShadeableIntersection* isects, MaterialSortKey* sortKeys)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    MaterialSortKey thisMat = sortKeys[idx];

    const PathSegment path = paths[idx];
    paths[idx] = path;

    

    MaterialSortKey* dummy = sortKeys + N;
}

/// Sorts the PathSegment and Intersection arrays by material type (dev_sortKeys)
/// Returns the new number of paths after discarding the non-intersections at the end of the array
__host__ int sortByMaterialType(int num_paths)
{
    typedef thrust::zip_iterator<cuda::std::tuple<PathSegment*, ShadeableIntersection*>> ZipIterator;

    // Sort both arrays together as a zip_iterator
    // This also sorts the keys (dev_sortKeys)
    ZipIterator zip_it = thrust::make_zip_iterator(thrust::make_tuple(dev_paths, dev_intersections));
    thrust::sort_by_key(thrust::device, dev_sortKeys, dev_sortKeys + num_paths, zip_it);

    int numBlocks = utilityCore::divUp(num_paths, BLOCK_SIZE_1D);

    // Binary search to find the partition point after which all paths are invalid
    MaterialSortKey* firstInvalid_it = thrust::lower_bound(thrust::device,
        dev_sortKeys, dev_sortKeys + num_paths, SORTKEY_INVALID);

    // Return the new number of paths
    return firstInvalid_it - dev_sortKeys;
}


// Note: Assumes dev_sortKeys has been sorted already by sortByMaterialType
__host__ void shadeByMaterialType(int num_paths, int iter, int depth, const SceneData& sd)
{
    using utilityCore::divUp;

    // For every material type, launch its corresponding kernel

    // These args stay the same across all rays.
    ShadeKernelArgs skArgs;
    skArgs.iter = iter;
    skArgs.depth = depth;
    skArgs.materials = dev_materials;
    skArgs.sceneData = sd;

    void* cudaKernelArgs[] = { &skArgs };

    dim3 numBlocks;
    int prev_end = 0;
    for (unsigned int m = MT_FIRST; m < MT_COUNT; ++m)
    {
        // Find the index range for this material
        MaterialSortKey maxKey = BuildSortKey((MaterialType)m, UINT16_MAX);
        int mt_end = thrust::upper_bound(thrust::device, dev_sortKeys + prev_end, dev_sortKeys + num_paths, maxKey) - dev_sortKeys;
        int mt_start = prev_end;
        int mt_count = mt_end - mt_start;
        
        // If there are rays for this material type, dispatch them all together in the same kernel
        if (mt_count)
        {
            skArgs.num_paths = mt_count;
            skArgs.pathSegments = dev_paths + mt_start;
            skArgs.shadeableIntersections = dev_intersections + mt_start;

            numBlocks.x = divUp(mt_count, BLOCK_SIZE_1D);
            ShadeKernel sk = getShadingKernelForMaterial((MaterialType)m);
            cudaLaunchKernel(sk, numBlocks, BLOCK_SIZE_1D, cudaKernelArgs, 0, nullptr);
            checkCUDAError("cudaLaunchKernel: shadingKernel");
        }

        prev_end = mt_end;
    }
}

__host__ void shadeLegacy(int num_paths, int iter, int depth)
{
    // TODO: compare between directly shading the path segments and shading
    // path segments that have been reshuffled to be contiguous in memory.

    ShadeKernelArgs skArgs = {
          iter
        , num_paths
        , depth
        , dev_intersections
        , dev_paths
        , dev_materials
        // TODO: Need to pass SceneData here
    };

    dim3 numblocksPathSegmentTracing = (num_paths + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;
    skDiffuseSimple<<<numblocksPathSegmentTracing, BLOCK_SIZE_1D >>>(skArgs);
}

__host__ int cullTerminatedPaths(int num_paths)
{
    auto dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, NonTerminated());
    //auto dev_path_end = thrust::remove_if(thrust::device, dev_paths, dev_paths + num_paths, Terminated());
    return dev_path_end - dev_paths;
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int maxDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    SceneData sd;
    sd.geoms = dev_geoms;
    sd.geoms_size = hst_scene->geoms.size();
    sd.lights = dev_lights;
    sd.lights_size = hst_scene->lights.size();
    sd.meshes = dev_meshes;
    sd.meshes_size = hst_scene->deviceMeshes.size();
    sd.bvhNodes = dev_bvhNodes;
    sd.bvhNodes_size = hst_scene->bvhNodes.size();

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, maxDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    int num_paths = pixelcount;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;
        computeIntersections<<<numblocksPathSegmentTracing, BLOCK_SIZE_1D>>> (
            depth,
            num_paths,
            dev_paths,
            sd,
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

    #if STREAM_COMPACTION || MATERIAL_SORT
        // Flag intersections by material type. We will use this to sort the path and isect arrays
        generateSortKeys<<<numblocksPathSegmentTracing, BLOCK_SIZE_1D>>>(num_paths, dev_intersections, dev_materials, dev_sortKeys);
        checkCUDAError("generateSortKeys");
    #endif

    #if MATERIAL_SORT
        int new_num_paths = sortByMaterialType(num_paths);
        checkCUDAError("sortByMaterialType");
        // TODO_WAVEFRONT: use new_num_paths to determine how many rays to regenerate.
        num_paths = new_num_paths;

        if (num_paths == 0)
            break;
    #endif

    #if MATERIAL_SORT
        shadeByMaterialType(num_paths, iter, depth, sd);
        checkCUDAError("shadeByMaterialType");
    #else
        shadeLegacy(num_paths, iter, depth);
    #endif

    #if STREAM_COMPACTION
        num_paths = cullTerminatedPaths(num_paths);
        // TODO_WAVEFRONT: Regenerate paths
    #endif
        iterationComplete = depth >= maxDepth || num_paths == 0;
        depth++;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;
    finalGather<<<numBlocksPixels, BLOCK_SIZE_1D>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
