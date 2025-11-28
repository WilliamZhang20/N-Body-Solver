// Reducing Asymptotic Time Complexity with a raw CUDA Bounding Volume Hierarchy
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "timer.h"
#include "files.h"
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/execution_policy.h>

#define SOFTENING 1e-9f

#define CHECK_LAST_CUDA_ERROR()                                           \
do {                                                                      \
    cudaError_t err = cudaGetLastError();                                 \
    if (err != cudaSuccess) {                                             \
        fprintf(stderr,                                                   \
            "CUDA ERROR at %s:%d — %s\n",                                 \
            __FILE__, __LINE__, cudaGetErrorString(err));                 \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
    err = cudaDeviceSynchronize();                                        \
    if (err != cudaSuccess) {                                             \
        fprintf(stderr,                                                   \
            "CUDA SYNC ERROR at %s:%d — %s\n",                            \
            __FILE__, __LINE__, cudaGetErrorString(err));                 \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while(0)

// Constants
#define BLOCK_SIZE 256

__global__ void integrate_leapfrog(
    float* const* pos,
    float* const* vel,
    const float* const* force,
    float dt,
    int nBodies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nBodies) return;

    float ax = force[0][i];
    float ay = force[1][i];
    float az = force[2][i];

    // v += a*dt
    vel[0][i] += ax * dt;
    vel[1][i] += ay * dt;
    vel[2][i] += az * dt;

    // p += v*dt  (velocity-verlet style)
    pos[0][i] += vel[0][i] * dt;
    pos[1][i] += vel[1][i] * dt;
    pos[2][i] += vel[2][i] * dt;
}

__device__ __forceinline__
float3 body_body_interaction(float3 ai, float4 bi, float4 bj)
{
    float3 r = make_float3(bj.x - bi.x, bj.y - bi.y, bj.z - bi.z);

    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + SOFTENING;
    float invDist = rsqrtf(distSqr);
    float invDistCube = invDist * invDist * invDist;

    float s = bj.w * invDistCube;   // bj.w = mass (1.0f)

    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;
    return ai;
}

__device__ __forceinline__
bool node_needs_opening(const float3 pos, const float3 node_com,
                        const float3 node_min, const float3 node_max,
                        float node_mass_total, float theta)
{
    if (node_mass_total <= 0.0f) return true;

    float3 d = make_float3(
        fmaxf(0.0f, fabsf(pos.x - node_com.x) - (node_max.x - node_min.x) * 0.5f),
        fmaxf(0.0f, fabsf(pos.y - node_com.y) - (node_max.y - node_min.y) * 0.5f),
        fmaxf(0.0f, fabsf(pos.z - node_com.z) - (node_max.z - node_min.z) * 0.5f)
    );

    float distSqr = d.x * d.x + d.y * d.y + d.z * d.z;
    float halfsize = fmaxf(node_max.x - node_min.x,
                          fmaxf(node_max.y - node_min.y,
                                node_max.z - node_min.z)) * 0.5f;

    return (halfsize * halfsize / distSqr) > (theta * theta);
}

__global__ void barnes_hut_traverse(
    const float* const* sorted_pos,      // [3] x,y,z of sorted particles
    const float* const* sorted_vel,      // not used here (but kept for symmetry)
    float* const* sorted_force,          // [3] output forces (accumulated)
    const int* __restrict__ left_child,
    const int* __restrict__ right_child,
    float* const* bbox_min,
    float* const* bbox_max,
    const float* __restrict__ node_mass,
    float* const* center,
    int nBodies,
    float theta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nBodies) return;

    float4 my_pos_mass = make_float4(
        sorted_pos[0][i],
        sorted_pos[1][i],
        sorted_pos[2][i],
        1.0f);  // unit mass

    float3 acc = make_float3(0.0f, 0.0f, 0.0f);

    // Start traversal from root
    int root = nBodies;  // root is always at index nBodies

    // Simple stack-less traversal (depth-first via loop)
    int stack[32];
    int stack_size = 0;
    stack[stack_size++] = root;

    while (stack_size > 0) {
        int node_idx = stack[--stack_size];

        // Decode children
        int l = left_child[node_idx];
        int r = right_child[node_idx];

        // Process left child
        if (l != -1) {
            bool is_leaf = (l < nBodies);
						int child_idx = is_leaf ? ~l : l;

            if (is_leaf) {
                if (child_idx != i) {  // skip self-interaction
                    float4 bj = make_float4(
                        sorted_pos[0][child_idx],
                        sorted_pos[1][child_idx],
                        sorted_pos[2][child_idx],
                        1.0f);
                    acc = body_body_interaction(acc, my_pos_mass, bj);
                }
            } else {
                float3 node_com = make_float3(center[0][child_idx],
                                             center[1][child_idx],
                                             center[2][child_idx]);
                float3 node_min = make_float3(bbox_min[0][child_idx],
                                             bbox_min[1][child_idx],
                                             bbox_min[2][child_idx]);
                float3 node_max = make_float3(bbox_max[0][child_idx],
                                             bbox_max[1][child_idx],
                                             bbox_max[2][child_idx]);
                float node_m = node_mass[child_idx];

                if (node_needs_opening(make_float3(my_pos_mass.x, my_pos_mass.y, my_pos_mass.z),
                                       node_com, node_min, node_max, node_m, theta)) {
                    stack[stack_size++] = child_idx;
                } else {
                    // Approximate: treat node as single body at CoM
                    float4 bj = make_float4(node_com.x, node_com.y, node_com.z, node_m);
                    acc = body_body_interaction(acc, my_pos_mass, bj);
                }
            }
        }

        // Process right child (same logic)
        if (r != -1) {
            bool is_leaf = (r < 0);
            int child_idx = is_leaf ? ~r : r;

            if (is_leaf) {
                if (child_idx != i) {
                    float4 bj = make_float4(
                        sorted_pos[0][child_idx],
                        sorted_pos[1][child_idx],
                        sorted_pos[2][child_idx],
                        1.0f);
                    acc = body_body_interaction(acc, my_pos_mass, bj);
                }
            } else {
                float3 node_com = make_float3(center[0][child_idx],
                                             center[1][child_idx],
                                             center[2][child_idx]);
                float3 node_min = make_float3(bbox_min[0][child_idx],
                                             bbox_min[1][child_idx],
                                             bbox_min[2][child_idx]);
                float3 node_max = make_float3(bbox_max[0][child_idx],
                                             bbox_max[1][child_idx],
                                             bbox_max[2][child_idx]);
                float node_m = node_mass[child_idx];

                if (node_needs_opening(make_float3(my_pos_mass.x, my_pos_mass.y, my_pos_mass.z),
                                       node_com, node_min, node_max, node_m, theta)) {
                    stack[stack_size++] = child_idx;
                } else {
                    float4 bj = make_float4(node_com.x, node_com.y, node_com.z, node_m);
                    acc = body_body_interaction(acc, my_pos_mass, bj);
                }
            }
        }
    }

    // Write result
    sorted_force[0][i] = acc.x;
    sorted_force[1][i] = acc.y;
    sorted_force[2][i] = acc.z;
}

__global__ void refit_bvh_bottom_up(
    const int* __restrict__ left_child,
    const int* __restrict__ right_child,
    float* const* bbox_min,       // [3] arrays: bbox_min[0][node] = min_x, etc.
    float* const* bbox_max,       // [3]
    float* const* center,         // [3] center of mass
    float* node_mass,
    int nBodies)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of internal nodes = nBodies - 1
    // They are stored in indices [nBodies, 2*nBodies-2]
    int internal_idx = nBodies + idx;
    if (idx >= nBodies - 1) return;

    // Get children
    int l = left_child[internal_idx];
    int r = right_child[internal_idx];

    // Decode whether child is leaf or internal
    bool l_is_leaf = (l < 0);
    bool r_is_leaf = (r < 0);

    int child_l_idx = l_is_leaf ? ~l : l;
    int child_r_idx = r_is_leaf ? ~r : r;

    // Load left child's data
    float3 l_min = make_float3(bbox_min[0][child_l_idx],
                               bbox_min[1][child_l_idx],
                               bbox_min[2][child_l_idx]);
    float3 l_max = make_float3(bbox_max[0][child_l_idx],
                               bbox_max[1][child_l_idx],
                               bbox_max[2][child_l_idx]);
    float3 l_com = make_float3(center[0][child_l_idx],
                               center[1][child_l_idx],
                               center[2][child_l_idx]);
    float  l_mass = node_mass[child_l_idx];

    // Load right child's data
    float3 r_min = make_float3(bbox_min[0][child_r_idx],
                               bbox_min[1][child_r_idx],
                               bbox_min[2][child_r_idx]);
    float3 r_max = make_float3(bbox_max[0][child_r_idx],
                               bbox_max[1][child_r_idx],
                               bbox_max[2][child_r_idx]);
    float3 r_com = make_float3(center[0][child_r_idx],
                               center[1][child_r_idx],
                               center[2][child_r_idx]);
    float  r_mass = node_mass[child_r_idx];

    // === Merge AABBs ===
    float3 node_min = fminf(l_min, r_min);
    float3 node_max = fmaxf(l_max, r_max);

    // === Compute total mass and center of mass ===
    float total_mass = l_mass + r_mass;
    float inv_total = (total_mass > 0.0f) ? 1.0f / total_mass : 0.0f;

    float3 node_com = make_float3(
        (l_com.x * l_mass + r_com.x * r_mass) * inv_total,
        (l_com.y * l_mass + r_com.y * r_mass) * inv_total,
        (l_com.z * l_mass + r_com.z * r_mass) * inv_total
    );

    // === Write back to this internal node ===
    bbox_min[0][internal_idx] = node_min.x;
    bbox_min[1][internal_idx] = node_min.y;
    bbox_min[2][internal_idx] = node_min.z;

    bbox_max[0][internal_idx] = node_max.x;
    bbox_max[1][internal_idx] = node_max.y;
    bbox_max[2][internal_idx] = node_max.z;

    center[0][internal_idx] = node_com.x;
    center[1][internal_idx] = node_com.y;
    center[2][internal_idx] = node_com.z;

    node_mass[internal_idx] = total_mass;
}

__device__ int delta(int i, int j, const uint64_t* morton_codes, int n)
{
    // Return LCP (longest common prefix) between morton_codes[i] and morton_codes[j]
    // Return -1 if out of bounds (ensures those comparisons lose)
    if (j < 0 || j >= n) return -1;
    
    uint64_t key_i = morton_codes[i];
    uint64_t key_j = morton_codes[j];
    
    if (key_i == key_j) {
        // Identical keys - use indices as tiebreaker
        // Protect against i==j (which would produce __clzll(0) - undefined)
        if (i == j) {
            // Same index & same key -> treat as a very long common prefix
            return 127; // large positive constant
        }
        return 64 + __clzll((uint64_t)i ^ (uint64_t)j);
    }
    
    return __clzll(key_i ^ key_j);
}

__device__ int sign(int x)
{
    return (x > 0) - (x < 0);
}

__device__ int2 determine_range(const uint64_t* morton_codes, int n, int i)
{
    // Determine direction by comparing LCP with neighbors
    int d_left = delta(i, i - 1, morton_codes, n);
    int d_right = delta(i, i + 1, morton_codes, n);
    
    int d = (d_right > d_left) ? 1 : -1;  // direction
    
    // Compute minimum LCP with neighbor in chosen direction
    int delta_min = delta(i, i - d, morton_codes, n);
    
    // Binary search to find the other end of the range
    int l_max = 2;
    while (delta(i, i + l_max * d, morton_codes, n) > delta_min) {
        l_max *= 2;
    }
    
    // Binary search for exact range end
    int l = 0;
    int t = l_max;
    while (t > 1) {
        t /= 2;
        if (delta(i, i + (l + t) * d, morton_codes, n) > delta_min) {
            l += t;
        }
    }
    
    int j = i + l * d;  // the other end of the range
    
    // Return [first, last] in sorted order
    int2 range;
    range.x = min(i, j);  // first
    range.y = max(i, j);  // last
    return range;
}

__device__ int find_split(int first, int last, const uint64_t* morton_codes, int n)
{
    // Find the split position using binary search
    int delta_node = delta(first, last, morton_codes, n);
    
    // Binary search for the split point
    int s = 0;
    int t = last - first;
    
    while (t > 1) {
        t = (t + 1) / 2;  // Divide and round up
        int mid = first + s + t;
        
        if (mid < last && delta(first, mid, morton_codes, n) > delta_node) {
            s += t;
        }
    }
    
    return first + s;
}

__global__ void build_lbvh_radix_tree(
    const uint64_t* __restrict__ morton_codes,
    int* parent,
    int* left_child,
    int* right_child,
    int nBodies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nBodies - 1) return;  // Only need N-1 internal nodes
    
    // Internal node index
    int internal_idx = nBodies + i;
    
    // 1. Determine range [first, last] that this internal node covers
    int2 range = determine_range(morton_codes, nBodies, i);
    int first = range.x;
    int last = range.y;
    
    // 2. Find split point
    int split = find_split(first, last, morton_codes, nBodies);
    
    // 3. Assign children
    int child_L, child_R;
    
    // Left child
    if (split == first) {
        // Leaf node
        child_L = split;
    } else {
        // Internal node: map particle index to internal node index
        child_L = nBodies + split;
    }
    
    // Right child
    if (split + 1 == last) {
        // Leaf node
        child_R = last;
    } else {
        // Internal node
        child_R = nBodies + split + 1;
    }
    
    // 4. Write children
    left_child[internal_idx] = child_L;
    right_child[internal_idx] = child_R;
    
    // 5. Set parent pointers
    if (child_L < nBodies) {
        // Leaf node
        parent[child_L] = internal_idx;
    } else {
        // Internal node
        parent[child_L] = internal_idx;
    }
    
    if (child_R < nBodies) {
        // Leaf node
        parent[child_R] = internal_idx;
    } else {
        // Internal node
        parent[child_R] = internal_idx;
    }
    
    // Root node (internal node 0 = index nBodies) has no parent
    if (i == 0) {
        parent[internal_idx] = -1;
    }
}

__global__ void compute_morton_codes(
    const float* __restrict__ x,
    const float* __restrict__ y,
    const float* __restrict__ z,
    uint64_t* morton,
    float min_x, float max_x,
    float min_y, float max_y,
    float min_z, float max_z,
    int nBodies)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nBodies) return;

    // --- Normalize coordinates to [0, 1) ---
    // Protect against degenerate bounding boxes (max == min)
    float range_x = max_x - min_x;
    float range_y = max_y - min_y;
    float range_z = max_z - min_z;

    float nx, ny, nz;
    const float SMALL_RANGE = 1e-12f;
    if (range_x <= SMALL_RANGE) nx = 0.5f; else nx = (x[idx] - min_x) / range_x;
    if (range_y <= SMALL_RANGE) ny = 0.5f; else ny = (y[idx] - min_y) / range_y;
    if (range_z <= SMALL_RANGE) nz = 0.5f; else nz = (z[idx] - min_z) / range_z;

    // Clamp to [0, 1) to be safe (in case of floating-point overshoot)
    nx = fminf(fmaxf(nx, 0.0f), 0.99999994f);  // ~1 - epsilon
    ny = fminf(fmaxf(ny, 0.0f), 0.99999994f);
    nz = fminf(fmaxf(nz, 0.0f), 0.99999994f);

    // Convert to 21-bit fixed-point integers (21×3 = 63 bits → fits in 64-bit)
    // 2^21 = 2097152
    // Use 64-bit temporaries to avoid truncation when shifting by >=32 bits
    uint64_t xx = (uint64_t)(nx * 2097152.0f);
    uint64_t yy = (uint64_t)(ny * 2097152.0f);
    uint64_t zz = (uint64_t)(nz * 2097152.0f);

    // --- Expand bits (spread 21 bits → 63 bits, 3 bits per coordinate) ---
    // This is the classic "bit interleaving" for Z-order (Morton) curve

    xx = (xx | (xx << 32)) & 0x1F00000000FFFFULL;   // bits 0-4  → 0-9,   32-36 → 41-45
    xx = (xx | (xx << 16)) & 0x1F0000FF0000FFULL;
    xx = (xx | (xx << 8))  & 0x100F00F00F00F00FULL;
    xx = (xx | (xx << 4))  & 0x10C30C30C30C30C3ULL;
    xx = (xx | (xx << 2))  & 0x1249249249249249ULL;

    yy = (yy | (yy << 32)) & 0x1F00000000FFFFULL;
    yy = (yy | (yy << 16)) & 0x1F0000FF0000FFULL;
    yy = (yy | (yy << 8))  & 0x100F00F00F00F00FULL;
    yy = (yy | (yy << 4))  & 0x10C30C30C30C30C3ULL;
    yy = (yy | (yy << 2))  & 0x1249249249249249ULL;

    zz = (zz | (zz << 32)) & 0x1F00000000FFFFULL;
    zz = (zz | (zz << 16)) & 0x1F0000FF0000FFULL;
    zz = (zz | (zz << 8))  & 0x100F00F00F00F00FULL;
    zz = (zz | (zz << 4))  & 0x10C30C30C30C30C3ULL;
    zz = (zz | (zz << 2))  & 0x1249249249249249ULL;

    // Combine into final 64-bit Morton code
    morton[idx] = ((uint64_t)xx << 2) | ((uint64_t)yy << 1) | (uint64_t)zz;
}

void lbvh_timestep(
    float* pos[3], float* vel[3],
    uint64_t* morton, int* idx,
    int* parent, int* left, int* right,
    float* bbox_min[3], float* bbox_max[3],
    float* node_mass, float* center[3],
    float* sorted_pos[3], float* sorted_vel[3], float* sorted_force[3],
    int nBodies, float dt)
{
    const float theta = 0.5f;
    dim3 block(BLOCK_SIZE);
    dim3 grid((nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // === 1. Compute global bounding box ===
    thrust::device_ptr<float> dpx(pos[0]), dpy(pos[1]), dpz(pos[2]);
    auto minmax_x = thrust::minmax_element(thrust::device, dpx, dpx + nBodies);
    auto minmax_y = thrust::minmax_element(thrust::device, dpy, dpy + nBodies);
    auto minmax_z = thrust::minmax_element(thrust::device, dpz, dpz + nBodies);

    // Fetch to host
    float min_x = *(minmax_x.first),  max_x = *(minmax_x.second);
    float min_y = *(minmax_y.first),  max_y = *(minmax_y.second);
    float min_z = *(minmax_z.first),  max_z = *(minmax_z.second);

    // Expand bounds slightly to prevent 1.0f overflow in Morton code
    float eps = 1e-5f;
    max_x += (max_x - min_x) * eps; min_x -= (max_x - min_x) * eps;
    max_y += (max_y - min_y) * eps; min_y -= (max_y - min_y) * eps;
    max_z += (max_z - min_z) * eps; min_z -= (max_z - min_z) * eps;

    // === 2. Compute Morton codes ===
    compute_morton_codes<<<grid, block>>>(
        pos[0], pos[1], pos[2], morton,
        min_x, max_x, min_y, max_y, min_z, max_z, nBodies);
    CHECK_LAST_CUDA_ERROR();
    
    // === 3. Sort particles by Morton code ===
    // Reset indices
    thrust::sequence(thrust::device, thrust::device_pointer_cast(idx), thrust::device_pointer_cast(idx) + nBodies);
    // Sort keys and move indices
    thrust::sort_by_key(thrust::device, thrust::device_pointer_cast(morton), thrust::device_pointer_cast(morton) + nBodies, thrust::device_pointer_cast(idx));

    // === 4. Gather: Reorder particle data into scratch buffers ===
    for (int d = 0; d < 3; ++d) {
        // Reset forces to 0
        cudaMemset(sorted_force[d], 0, nBodies * sizeof(float));

        thrust::gather(thrust::device, thrust::device_pointer_cast(idx), thrust::device_pointer_cast(idx) + nBodies,
                       thrust::device_pointer_cast(pos[d]), thrust::device_pointer_cast(sorted_pos[d]));
        
        thrust::gather(thrust::device, thrust::device_pointer_cast(idx), thrust::device_pointer_cast(idx) + nBodies,
                       thrust::device_pointer_cast(vel[d]), thrust::device_pointer_cast(sorted_vel[d]));
    }

    // === 5. Build tree on sorted particles ===
    build_lbvh_radix_tree<<<grid, block>>>(morton, parent, left, right, nBodies);
		CHECK_LAST_CUDA_ERROR();

    // === 6. Leaf AABBs and center of mass ===
    // Assuming unit mass (1.0f)
    compute_leaf_aabbs_and_com<<<grid, block>>>(sorted_pos, nullptr, bbox_min, bbox_max, center, node_mass, nBodies);
		CHECK_LAST_CUDA_ERROR();

    // === 7. Refit internal nodes ===
    refit_bvh_bottom_up<<<grid, block>>>(left, right, bbox_min, bbox_max, center, node_mass, nBodies);
		CHECK_LAST_CUDA_ERROR();

    // === 8. Force calculation (on sorted particles) ===
    barnes_hut_traverse<<<grid, block>>>(
        sorted_pos, sorted_vel, sorted_force,
        left, right, bbox_min, bbox_max, node_mass, center,
        nBodies, theta);
		CHECK_LAST_CUDA_ERROR();

    // === 9. Integration (on sorted particles) ===
    integrate_leapfrog<<<grid, block>>>(sorted_pos, sorted_vel, sorted_force, dt, nBodies);
		CHECK_LAST_CUDA_ERROR();

    // === 10. Scatter: Put data back into original order ===
    for (int d = 0; d < 3; ++d) {
        thrust::scatter(thrust::device,
                        thrust::device_pointer_cast(sorted_pos[d]),
                        thrust::device_pointer_cast(sorted_pos[d]) + nBodies,
                        thrust::device_pointer_cast(idx),
                        thrust::device_pointer_cast(pos[d]));

        thrust::scatter(thrust::device,
                        thrust::device_pointer_cast(sorted_vel[d]),
                        thrust::device_pointer_cast(sorted_vel[d]) + nBodies,
                        thrust::device_pointer_cast(idx),
                        thrust::device_pointer_cast(vel[d]));
    }
}

void allocate_bvh_arrays(
    int** parent, int** left, int** right,
    float* bbox_min[3], float* bbox_max[3],
    float** node_mass, float* center[3],
    int nBodies)
{
    // Binary tree with N leaves has N-1 internal nodes = 2N-1 total nodes
    int nNodes = 2 * nBodies - 1;
    int node_bytes = nNodes * sizeof(int);
    int node_bytes_float = nNodes * sizeof(float);
    
    // Tree topology
    cudaMalloc(parent, node_bytes);
    cudaMalloc(left, node_bytes);
    cudaMalloc(right, node_bytes);
    
    // Bounding boxes (min and max for x, y, z)
    for (int d = 0; d < 3; d++) {
        cudaMalloc(&bbox_min[d], node_bytes_float);
        cudaMalloc(&bbox_max[d], node_bytes_float);
    }
    
    // Center of mass (x, y, z)
    for (int d = 0; d < 3; d++) {
        cudaMalloc(&center[d], node_bytes_float);
    }
    
    // Total mass at each node
    cudaMalloc(node_mass, node_bytes_float);
    
    // Initialize parent array to -1 (convention: root has no parent)
    cudaMemset(*parent, -1, node_bytes);
}

int main(const int argc, const char** argv) {
    int nBodies = 2 << 11;
    if (argc > 1) nBodies = 2 << atoi(argv[1]);

    const char * initialized_values;
    const char * solution_values;
    if (nBodies == 2 << 11) {
        initialized_values = "09-nbody/files/initialized_4096";
        solution_values = "09-nbody/files/solution_4096";
    } else {
        initialized_values = "09-nbody/files/initialized_65536";
        solution_values = "09-nbody/files/solution_65536";
    }
    if (argc > 2) initialized_values = argv[2];
    if (argc > 3) solution_values = argv[3];

    const float dt = 0.01f;
    const int nIters = 10;

    int bytes = nBodies * sizeof(float);
    
    // Host pinned memory (SoA layout)
    float *h_pos[3], *h_vel[3];
    for(int d = 0; d < 3; d++) {
        cudaMallocHost(&h_pos[d], bytes);
        cudaMallocHost(&h_vel[d], bytes);
    }

    // Load initial data from file
    float *tmp = (float*)malloc(nBodies * 6 * sizeof(float));
    read_values_from_file(initialized_values, tmp, nBodies * 6 * sizeof(float));

    for (int i = 0; i < nBodies; i++) {
        h_pos[0][i] = tmp[i * 6 + 0];  // x
        h_pos[1][i] = tmp[i * 6 + 1];  // y
        h_pos[2][i] = tmp[i * 6 + 2];  // z
        h_vel[0][i] = tmp[i * 6 + 3];  // vx
        h_vel[1][i] = tmp[i * 6 + 4];  // vy
        h_vel[2][i] = tmp[i * 6 + 5];  // vz
    }
    free(tmp);

    // Device arrays (SoA layout)
    float *d_pos[3], *d_vel[3];
    for(int d = 0; d < 3; d++) {
        cudaMalloc(&d_pos[d], bytes);
        cudaMalloc(&d_vel[d], bytes);
    }
    
    float *d_sorted_pos[3], *d_sorted_vel[3], *d_sorted_force[3];
    for(int d = 0; d < 3; d++) {
        cudaMalloc(&d_sorted_pos[d], bytes);
        cudaMalloc(&d_sorted_vel[d], bytes);
        cudaMalloc(&d_sorted_force[d], bytes);
    }

    // Additional device buffers we'll need
    uint64_t *d_morton;     cudaMalloc(&d_morton, nBodies * sizeof(uint64_t));
    int      *d_idx;        cudaMalloc(&d_idx,    nBodies * sizeof(int));
   
    // BVH node arrays (2N-1 nodes total)
    int    *d_parent, *d_left, *d_right;
    float  *d_bbox_min[3], *d_bbox_max[3];
    float  *d_mass, *d_center[3];
    allocate_bvh_arrays(&d_parent, &d_left, &d_right,
                        d_bbox_min, d_bbox_max, d_mass, d_center, nBodies);

    // Copy initial data to device
    for(int d = 0; d < 3; d++) {
        cudaMemcpy(d_pos[d], h_pos[d], bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel[d], h_vel[d], bytes, cudaMemcpyHostToDevice);
    }

    double totalTime = 0.0;
    for(int iter = 0; iter < nIters; iter++) {
        StartTimer();

        // FULL LBVH PIPELINE PER TIMESTEP
        lbvh_timestep(
            d_pos, d_vel,
            d_morton, d_idx,
            d_parent, d_left, d_right,
            d_bbox_min, d_bbox_max,
            d_mass, d_center,
            d_sorted_pos, d_sorted_vel, d_sorted_force,
            nBodies, dt
        );
        
        cudaDeviceSynchronize();
        totalTime += GetTimer() / 1000.0;
    }

    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / (totalTime / nIters);
    printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

    // Copy back final state
    for(int d = 0; d < 3; d++) {
        cudaMemcpy(h_pos[d], d_pos[d], bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vel[d], d_vel[d], bytes, cudaMemcpyDeviceToHost);
    }

    // Write results to file
    float *result = (float*)malloc(nBodies * 6 * sizeof(float));
    for (int i = 0; i < nBodies; i++) {
        result[i * 6 + 0] = h_pos[0][i];  // x
        result[i * 6 + 1] = h_pos[1][i];  // y
        result[i * 6 + 2] = h_pos[2][i];  // z
        result[i * 6 + 3] = h_vel[0][i];  // vx
        result[i * 6 + 4] = h_vel[1][i];  // vy
        result[i * 6 + 5] = h_vel[2][i];  // vz
    }

    write_values_to_file(solution_values, result, nBodies * 6 * sizeof(float));
    free(result);

    // Cleanup device memory
    for(int d = 0; d < 3; d++) {
        cudaFree(d_pos[d]);
        cudaFree(d_vel[d]);
    }
    cudaFree(d_morton);
    cudaFree(d_idx);
    
    // Free BVH arrays
    cudaFree(d_parent);
    cudaFree(d_left);
    cudaFree(d_right);
    for(int d = 0; d < 3; d++) {
        cudaFree(d_bbox_min[d]);
        cudaFree(d_bbox_max[d]);
        cudaFree(d_center[d]);
    }
    cudaFree(d_mass);

    // Cleanup host memory
    for(int d = 0; d < 3; d++) {
        cudaFreeHost(h_pos[d]);
        cudaFreeHost(h_vel[d]);
    }
    
    for(int d=0; d<3; d++) { 
	    cudaFree(d_sorted_force[d]); // Free the scratch force
	    cudaFree(d_sorted_vel[d]);
	    cudaFree(d_sorted_pos[d]);
    }
	
    return 0;
}