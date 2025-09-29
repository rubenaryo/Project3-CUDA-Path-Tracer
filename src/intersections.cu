#include "intersections.h"

#include "bvh.h"

/// Kernel to label each intersection with additional information to be used for ray sorting and discarding
__global__ void generateSortKeys(int N, const ShadeableIntersection* isects, Material* mats, MaterialSortKey* sortKeys)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const ShadeableIntersection isect = isects[idx];
    if (isect.t > FLT_EPSILON)
    {
        const Material mat = mats[isect.materialId];
        sortKeys[idx] = BuildSortKey(mat.type, isect.materialId);
    }
    else
    {
        sortKeys[idx] = SORTKEY_INVALID;
    }
}

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ bool intersectAABB(const Ray& ray, const AABB& aabb, float& tMin, float& tMax)
{
    glm::vec3 invDir = 1.0f / ray.direction;

    // Intersect with each axis aligned slab
    glm::vec3 t0 = (aabb.min - ray.origin) * invDir;
    glm::vec3 t1 = (aabb.max - ray.origin) * invDir;

    // Find the closest one
    glm::vec3 tNear = glm::min(t0, t1);
    glm::vec3 tFar = glm::max(t0, t1);

    // Within that find the largest intersection t
    tMin = fmaxf(fmaxf(tNear.x, tNear.y), tNear.z);
    tMax = fminf(fminf(tFar.x, tFar.y), tFar.z);

    return tMax >= tMin && tMax >= 0.0f;;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = glm::min(t1, t2);
        outside = true;
    }
    else
    {
        t = glm::max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float triangleIntersectionTest(Geom tri, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside)
{
    //bool result = glm::intersectRayTriangle(r.origin, r.direction, )
    return 0;
}


// From CIS 561
__host__ __device__ bool intersectRayTriangle_MollerTrumbore(const Ray& ray,
    const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
    BVHIntersectResult& isectResult)
{
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 h = glm::cross(ray.direction, edge2);
    float a = glm::dot(edge1, h);

    if (a > -FLT_EPSILON && a < FLT_EPSILON) 
        return false;

    float f = 1.0f / a;
    glm::vec3 s = ray.origin - v0;
    float u = f * glm::dot(s, h);

    if (u < 0.0f || u > 1.0f)
        return false;

    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(ray.direction, q);

    if (v < 0.0f || u + v > 1.0f)
        return false;

    float t = f * glm::dot(edge2, q);

    if (t > FLT_EPSILON)
    {
        isectResult.pos = ray.origin + ray.direction * t;
        isectResult.uv = glm::vec2(u, v);
        isectResult.normal = glm::normalize(glm::cross(edge1, edge2));
        isectResult.t = t;
        return true;
    }

    return false;
}

__host__ __device__ bool intersectRayTriangle_MollerTrumbore(const Ray& ray,
const Triangle& tri, BVHIntersectResult& isectResult)
{
    return intersectRayTriangle_MollerTrumbore(ray, tri.v[0], tri.v[1], tri.v[2], isectResult);
}

__host__ __device__  float bvhIntersectionTest(const Ray& r, uint32_t bvhRootIndex, const BVHNode* bvhNodes, const glm::vec3* vertices, BVHIntersectResult& isectResult)
{
    uint32_t nodeStack[BVH_MAX_DEPTH];
    uint32_t stackIdx = 0;
    nodeStack[stackIdx++] = bvhRootIndex;
    
    isectResult.t = FLT_MAX;
    BVHIntersectResult tmpIsectResult;

    // DFS - Keep going until there's no nodes left in the stack
    while (stackIdx > 0)
    {
        uint32_t nodeIdx = nodeStack[--stackIdx];
        const BVHNode node = bvhNodes[nodeIdx]; //gmem access
        float tMinAABB, tMaxAABB;
        if (!intersectAABB(r, node.bounds, tMinAABB, tMaxAABB))
            continue;
        
        if (node.childIndex == -1) // Leaf: No more children to test
        {
            // Test all the tris
            const uint32_t maxTriIdx = node.triIndex + node.triCount;
            for (uint32_t triIdx = node.triIndex; triIdx < maxTriIdx; ++triIdx)
            {
                const Triangle tri = GetTriangleFromTriIdx(triIdx, vertices);
                if (intersectRayTriangle_MollerTrumbore(r, tri, tmpIsectResult) && 
                    tmpIsectResult.t < isectResult.t)
                {
                    // This is the closest hit tri that we've tested so far
                    isectResult = tmpIsectResult;
                }
            }
        }
        else
        {
            nodeStack[stackIdx++] = node.childIndex + 1;
            nodeStack[stackIdx++] = node.childIndex + 0;
        }
    }

    return isectResult.t;
}

__host__ __device__  float meshIntersectionTest(Geom meshGeom, const Mesh mesh, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec2& uv, bool& outside)
{
    glm::vec3 ro = multiplyMV(meshGeom.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(meshGeom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray testRay;
    testRay.origin = ro;
    testRay.direction = rd;

    outside = true;
    bool hitAnything = false;
    int closestIdx = -1;

    BVHIntersectResult isectResult;
    isectResult.t = FLT_MAX;

    for (uint32_t i = 0; (i+2) < mesh.vtx_count; i += 3)
    {
        //const Triangle tri = GetTriangleFromTriIdx(i, mesh.vtx);
        Triangle tri;
        tri.v[0] = mesh.vtx[i];
        tri.v[1] = mesh.vtx[i + 1];
        tri.v[2] = mesh.vtx[i + 2];

        BVHIntersectResult tmpIsectResult;
        //bool hit = glm::intersectRayTriangle<glm::vec3>(r.origin, r.direction, v0, v1, v2, baryPosition);
        bool hit = intersectRayTriangle_MollerTrumbore(testRay, tri, tmpIsectResult);
        if (hit && tmpIsectResult.t < isectResult.t)
        {
            hitAnything = true;
            closestIdx = i;
            isectResult = tmpIsectResult;
        }
    }

    if (hitAnything)
    {
        intersectionPoint = glm::vec3(meshGeom.transform * glm::vec4(isectResult.pos, 1.0f));
        normal = glm::vec3(meshGeom.invTranspose * glm::vec4(isectResult.normal, 0.0f));
        normal = glm::normalize(normal);

        uv = glm::vec2(0.f);
        if (0 < closestIdx && closestIdx < (mesh.uvs_count - 2))
        {
            float u = isectResult.uv.x;
            float v = isectResult.uv.y;
            float w = 1.0f - u - v;

            // Barycentric interp to get uv coords
            uv = w * mesh.uvs[closestIdx] +
                 u * mesh.uvs[closestIdx + 1] +
                 v * mesh.uvs[closestIdx + 2];
        }

        return glm::length(r.origin - intersectionPoint);
    }

    return -1.0f;
}

__device__ void sceneIntersect(PathSegment& path, const SceneData& sceneData, ShadeableIntersection& result)
{
    float t;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    MaterialID hitMaterial = -1;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;

    const PathSegment pathCopy = path;
    
    int geoms_size = sceneData.geoms_size;
    const Geom* geoms = sceneData.geoms;

    for (int i = 0; i < geoms_size; i++)
    {
        const Geom geom = geoms[i];

        if (geom.type == GT_CUBE)
        {
            t = boxIntersectionTest(geom, pathCopy.ray, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == GT_SPHERE)
        {
            t = sphereIntersectionTest(geom, pathCopy.ray, tmp_intersect, tmp_normal, outside);
        }
        else if (geom.type == GT_MESH)
        {
            const Mesh mesh = sceneData.meshes[geom.meshId];
            glm::vec2 uv;
            t = meshIntersectionTest(geom, mesh, pathCopy.ray, tmp_intersect, tmp_normal, uv, outside);
        }
        // TODO: add more intersection tests here... triangle? metaball? CSG?

        // Compute the minimum t from the intersection tests to determine what
        // scene geometry object was hit first.
        if (t > 0.0f && t_min > t)
        {
            t_min = t;
            hit_geom_index = i;
            hitMaterial = geom.materialid;
            intersect_point = tmp_intersect;
            normal = tmp_normal;
        }
    }

    if (hit_geom_index == -1)
    {
        result.t = -1.0f;
        path.color = glm::vec3(0.0f); // This gmem read might be really bad.
    }
    else
    {
        // The ray hits something
        result.t = t_min;
        result.materialId = hitMaterial;
        result.surfaceNormal = normal;
    }
}

__device__ void lightsIntersect(PathSegment& path, const Light* lights, int lights_size, ShadeableIntersection& result, LightID& resultId)
{
    // TODO_MIS
}
