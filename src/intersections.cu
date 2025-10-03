#include "intersections.h"

#include "bvh.h"
#include "light.h"
#include "utilities.h"

/// Kernel to label each intersection with additional information to be used for ray sorting and discarding
__global__ void generateSortKeys(int N, const ShadeableIntersection* isects, Material* mats, MaterialSortKey* sortKeys)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const ShadeableIntersection isect = isects[idx];
    if (isect.t > FLT_EPSILON)
    {
        // TODO: maybe instead of using the sortkey buffer we can just sort directly off of this???
        sortKeys[idx] = isect.matSortKey;
    }
    else
    {
        sortKeys[idx] = SORTKEY_INVALID;
    }
}

// Fallback option for creating TBN if the uvs are degenerate
__host__ __device__ void createCoordinateSystem(const glm::vec3& normal, glm::vec3& tangent, glm::vec3& bitangent)
{
    if (fabsf(normal.x) > fabsf(normal.y)) {
        tangent = glm::normalize(glm::vec3(-normal.z, 0.0f, normal.x));
    }
    else {
        tangent = glm::normalize(glm::vec3(0.0f, normal.z, -normal.y));
    }
    bitangent = glm::cross(normal, tangent);
}

// Get the area of the rectangle in world space from its tfm
__device__ float getRectArea(const glm::mat4& rectTfm)
{
    glm::vec3 edge1 = glm::vec3(rectTfm * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
    glm::vec3 edge2 = glm::vec3(rectTfm * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f));

    return glm::length(glm::cross(edge1, edge2));
}

__host__ __device__ float intersectAABB(const Ray& ray, const AABB& aabb)
{
    glm::vec3 invDir = 1.0f / ray.direction;

    // Intersect with each axis aligned slab
    glm::vec3 t0 = (aabb.min - ray.origin) * invDir;
    glm::vec3 t1 = (aabb.max - ray.origin) * invDir;

    // Find the closest one
    glm::vec3 tNear = glm::min(t0, t1);
    glm::vec3 tFar = glm::max(t0, t1);

    // Within that find the largest intersection t
    float tMin = fmaxf(fmaxf(tNear.x, tNear.y), tNear.z);
    float tMax = fminf(fminf(tFar.x, tFar.y), tFar.z);

    bool hit = tMax >= tMin && tMax >= 0.0f;
    return hit ? tMin : FLT_MAX;
}

__host__ __device__ float rectIntersectionTest(const Geom& geom, const Ray& ray, glm::vec3& out_isectPoint, glm::vec3& out_normal, glm::vec2& out_uv)
{
    glm::vec3 ro = glm::vec3(geom.inverseTransform * glm::vec4(ray.origin, 1.0f));
    glm::vec3 rd = glm::vec3(geom.inverseTransform * glm::vec4(ray.direction, 0.0f));

    if (fabsf(rd.z) < FLT_EPSILON) {
        return FLT_MAX;  // Ray is parallel, no intersection
    }

    float t = -ro.z / rd.z;

    if (t < 0.0f) {
        return FLT_MAX;
    }

    // local isect point
    glm::vec3 objPoint = ro + t * rd;

    // Check if point is inside unit square bounds [-0.5, 0.5] x [-0.5, 0.5]
    if (objPoint.x >= -0.5f && objPoint.x <= 0.5f &&
        objPoint.y >= -0.5f && objPoint.y <= 0.5f) {
        
        // Transform back to world space
        out_isectPoint = glm::vec3(geom.transform * glm::vec4(objPoint, 1.0f));

        out_uv = glm::vec2(objPoint.x + 0.5f, objPoint.y + 0.5f);

        // Transform normal to world space
        glm::vec3 objNormal = rd.z < 0.0f ? glm::vec3(0.0f, 0.0f, 1.0f)
            : glm::vec3(0.0f, 0.0f, -1.0f);

        out_normal = glm::normalize(glm::vec3(geom.invTranspose * glm::vec4(objNormal, 0.0f)));

        return t;
    }

    return FLT_MAX;
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

// TODO: potentially broken after mesh refactor
__host__ __device__ bool testAllTrianglesForMesh(const Ray& r, uint32_t meshVtxIdx, const glm::vec3* vertices, uint32_t vtx_count, BVHIntersectResult& isectResult)
{
    bool hit = false;
    isectResult.t = FLT_MAX;
    for (uint32_t i = 0; (i + 2) < vtx_count; i += 3)
    {
        glm::vec3 v0 = vertices[i];
        glm::vec3 v1 = vertices[i + 1];
        glm::vec3 v2 = vertices[i + 2];

        BVHIntersectResult tmpIsectResult;
        bool triHit = intersectRayTriangle_MollerTrumbore(r, v0, v1, v2, tmpIsectResult);
        if (triHit && tmpIsectResult.t < isectResult.t)
        {
            hit = true;
            tmpIsectResult.triIdx = i;
            isectResult = tmpIsectResult;
        }
    }

    return hit;
}

__host__ __device__  bool bvhIntersectionTest(const Ray& r, uint32_t bvhRootIndex, const SceneData& sd, BVHIntersectResult& isectResult)
{
    const BVHNode* bvhNodes = sd.bvhNodes;
    const glm::vec3* vertices = sd.vertices;
    const glm::uvec3* indices = sd.indices;
    int bvhNodes_size = sd.bvhNodes_size;
    int vertices_size = sd.vertices_size;
    int indices_size = sd.indices_size;

    uint32_t nodeStack[BVH_MAX_DEPTH];
    uint32_t stackIdx = 0;
    nodeStack[stackIdx++] = bvhRootIndex;
    
    bool hitAnything = false;
    isectResult.t = FLT_MAX;
    BVHIntersectResult tmpIsectResult;

    // DFS - Keep going until there's no nodes left in the stack
    while (stackIdx > 0)
    {
        uint32_t nodeIdx = nodeStack[--stackIdx];
        const BVHNode node = bvhNodes[nodeIdx]; //gmem access
        
        if (node.childIndex == -1) // Leaf: No more children to test
        {
            // Test all the tris
            const uint32_t maxTriIdx = node.triIndex + node.triCount;
            for (uint32_t triIdx = node.triIndex; triIdx < maxTriIdx; ++triIdx)
            {
                glm::uvec3 triIndices = indices[triIdx];
                glm::vec3 v0 = vertices[triIndices.x];
                glm::vec3 v1 = vertices[triIndices.y];
                glm::vec3 v2 = vertices[triIndices.z];

                if (intersectRayTriangle_MollerTrumbore(r, v0, v1, v2, tmpIsectResult) && 
                    tmpIsectResult.t < isectResult.t)
                {
                    hitAnything = true;
                    // This is the closest hit tri that we've tested so far
                    isectResult = tmpIsectResult;
                    isectResult.triIdx = triIdx;
                }
            }
        }
        else
        {
            uint32_t idxA = node.childIndex + 0;
            uint32_t idxB = node.childIndex + 1;

            BVHNode childA = bvhNodes[idxA];
            BVHNode childB = bvhNodes[idxB];

            float tA = intersectAABB(r, childA.bounds);
            float tB = intersectAABB(r, childB.bounds);

            if (tA > tB)
            {
                if (tA < isectResult.t) nodeStack[stackIdx++] = idxA;
                if (tB < isectResult.t) nodeStack[stackIdx++] = idxB;
            }
            else
            {
                if (tB < isectResult.t) nodeStack[stackIdx++] = idxB;
                if (tA < isectResult.t) nodeStack[stackIdx++] = idxA;
            }
        }
    }

    return hitAnything;
}

__host__ __device__  float meshIntersectionTest(Geom meshGeom, const SceneData& sd, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec3& tangent, glm::vec2& uv, bool& outside)
{
    Ray localRay;
    localRay.origin = multiplyMV(meshGeom.inverseTransform, glm::vec4(r.origin, 1.0f));
    localRay.direction = glm::normalize(multiplyMV(meshGeom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    outside = true;
    const BVHNode* bvhNodes = sd.bvhNodes;

    BVHIntersectResult isectResult;

#if USE_BVH
    bool hit = bvhIntersectionTest(localRay, meshGeom.bvhRootIdx, sd, isectResult);
#else
    bool hit = testAllTrianglesForMesh(localRay, 0, mesh.vtx, mesh.vtx_count, isectResult);
#endif

    if (hit)
    {
        intersectionPoint = glm::vec3(meshGeom.transform * glm::vec4(isectResult.pos, 1.0f));
        normal = glm::vec3(meshGeom.invTranspose * glm::vec4(isectResult.normal, 0.0f));
        normal = glm::normalize(normal);

        float u = isectResult.uv.x;
        float v = isectResult.uv.y;
        float w = 1.0f - u - v;
        
        glm::uvec3 triIndices = sd.indices[isectResult.triIdx];

        // Compute UVs and tangents
        glm::vec3 v0 = sd.vertices[triIndices.x];
        glm::vec3 v1 = sd.vertices[triIndices.y];
        glm::vec3 v2 = sd.vertices[triIndices.z];

        glm::vec2 uv0 = sd.uvs[triIndices.x];
        glm::vec2 uv1 = sd.uvs[triIndices.y];
        glm::vec2 uv2 = sd.uvs[triIndices.z];

        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;

        glm::vec2 deltaUV1 = uv1 - uv0;
        glm::vec2 deltaUV2 = uv2 - uv0;
        
        float det = deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x;

        if (fabsf(det) < FLT_EPSILON)
        {
            glm::vec3 bitangent;
            createCoordinateSystem(normal, tangent, bitangent);
        }
        else
        {
            float invDet = 1.0f / (det);
            tangent = invDet * (deltaUV2.y * edge1 - deltaUV1.y * edge2);
            tangent = glm::normalize(tangent - normal * glm::dot(normal, tangent)); // correction
        }


        // Barycentric interp to get final uv coords
        uv = w * uv0 +
             u * uv1 +
             v * uv2;

        return glm::length(r.origin - intersectionPoint);
    }

    return -1.0f;
}

__device__ glm::vec3 sampleEnvironmentMap(cudaTextureObject_t envMap, const glm::vec3& direction) {
    glm::vec3 dir = glm::normalize(direction);

    // Convert direction to lat-long UV
    float phi = atan2f(dir.z, dir.x);
    float theta = acosf(glm::clamp(dir.y, -1.0f, 1.0f));

    float u = (phi + PI) / (2.0f * PI);
    float v = theta / PI;
    
    float4 color = tex2D<float4>(envMap, u, v);
    return glm::vec3(color.x, color.y, color.z);
}

__device__ void sceneIntersect(PathSegment& path, const SceneData& sceneData, ShadeableIntersection& result, cudaTextureObject_t* envMaps, int ignoreGeomId)
{
    float t;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    glm::vec3 tangent;
    glm::vec2 uv;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    GeomType hit_geom_type = GT_INVALID;
    MaterialSortKey hitMaterialKey = SORTKEY_INVALID;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
    glm::vec3 tmp_tangent(-1.0f); // some isect functions don't support tangent
    glm::vec2 tmp_uv(-1.0f); // some isect functions don't support UV
    glm::vec3 ToLight_Local; // TODO: This works for rect only right now

    const PathSegment pathCopy = path;
    
    int geoms_size = sceneData.geoms_size;
    const Geom* geoms = sceneData.geoms;

    for (int i = 0; i < geoms_size; i++)
    {
        const Geom geom = geoms[i];
        if (i == ignoreGeomId)
            continue;

        if (geom.type == GT_CUBE)
        {
            t = boxIntersectionTest(geom, pathCopy.ray, tmp_intersect, tmp_normal, outside);
            tmp_uv = glm::vec2(-1.0f);
        }
        else if (geom.type == GT_SPHERE)
        {
            t = sphereIntersectionTest(geom, pathCopy.ray, tmp_intersect, tmp_normal, outside);
            tmp_uv = glm::vec2(-1.0f);
        }
        else if (geom.type == GT_MESH)
        {
            t = meshIntersectionTest(geom, sceneData, pathCopy.ray, tmp_intersect, tmp_normal, tmp_tangent, tmp_uv, outside);
        }
        else if (geom.type == GT_RECT)
        {
            t = rectIntersectionTest(geom, pathCopy.ray, tmp_intersect, tmp_normal, tmp_uv);
        }

        // Compute the minimum t from the intersection tests to determine what
        // scene geometry object was hit first.
        if (t > 0.0f && t_min > t)
        {
            t_min = t;
            hit_geom_index = i;
            hitMaterialKey = geom.matSortKey;
            hit_geom_type = geom.type;
            intersect_point = tmp_intersect;
            normal = tmp_normal;
            tangent = tmp_tangent;
            uv = tmp_uv;
        }
    }

    if (hit_geom_index == -1)
    {
        result.t = -1.0f;
        //path.throughput = glm::vec3(0.0f); // This gmem read might be really bad.
        if (envMaps)
        {
            path.throughput *= sampleEnvironmentMap(envMaps[0], pathCopy.ray.direction);
            path.Lo += path.throughput;
        }
        else
        {
            path.throughput *= glm::vec3(0.0f);
            //path.Lo += path.throughput;
        }
    }
    else
    {
        MaterialType matType = GetMaterialTypeFromSortKey(hitMaterialKey);
        if (matType == MT_EMISSIVE) 
        {
            assert(hit_geom_type == GT_RECT);

            // We happened to hit a light.
            //float rectPDF = 
        }

        // The ray hits something
        result.t = t_min;
        result.matSortKey = hitMaterialKey;
        result.surfaceNormal = normal;
        result.tangent = tangent;
        result.uv = uv;
        result.hitGeomIdx = hit_geom_index; // TODO: evaluate whether this is necessary (maybe just copy the geom transform?)
    }
}
