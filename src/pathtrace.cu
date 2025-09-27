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
static PathSegment* dev_paths = NULL;
static PathSegment* dev_opaquePaths = NULL;
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
    cudaMalloc(&dev_opaquePaths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

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

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
            pathSegments[path_index].color = glm::vec3(0.0f); // This gmem read might be really bad.
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

struct MaterialIdComp {
    
    using PathIsectTuple = thrust::tuple<PathSegment, ShadeableIntersection>;
    
    __host__ __device__
    bool operator()(const PathIsectTuple& a, const PathIsectTuple& b) const {
        return thrust::get<1>(a).materialId < thrust::get<1>(b).materialId;
    }
};
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
__host__ void shadeByMaterialType(int num_paths, int iter)
{
    using utilityCore::divUp;

    // For every material type, launch its corresponding kernel

    ShadeKernelArgs skArgs;
    skArgs.iter = iter;
    skArgs.materials = dev_materials;

    void* cudaKernelArgs[] = { &skArgs };

    dim3 numBlocks;
    int prev_end = 0;
    for (unsigned int m = MT_FIRST; m < MT_COUNT; ++m)
    {
        MaterialSortKey maxKey = BuildSortKey((MaterialType)m, UINT16_MAX);
        int mt_end = thrust::upper_bound(thrust::device, dev_sortKeys + prev_end, dev_sortKeys + num_paths, maxKey) - dev_sortKeys;
        int mt_start = prev_end;
        int mt_count = mt_end - mt_start;
        
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

__host__ void shadeLegacy(int num_paths, int iter)
{
    // TODO: compare between directly shading the path segments and shading
    // path segments that have been reshuffled to be contiguous in memory.

    ShadeKernelArgs skArgs = {
          iter
        , num_paths
        , dev_intersections
        , dev_paths
        , dev_materials
    };

    dim3 numblocksPathSegmentTracing = (num_paths + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;
    skDiffuse<<<numblocksPathSegmentTracing, BLOCK_SIZE_1D >>>(skArgs);
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
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
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
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

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
        shadeByMaterialType(num_paths, iter);
        checkCUDAError("shadeByMaterialType");
    #else
        shadeLegacy(num_paths, iter);
    #endif

    #if STREAM_COMPACTION
        num_paths = cullTerminatedPaths(num_paths);
        // TODO_WAVEFRONT: Regenerate paths
    #endif
        iterationComplete = depth >= traceDepth || num_paths == 0; // TODO: should be based off stream compaction results.

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
