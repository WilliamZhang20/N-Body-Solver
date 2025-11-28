# N-Body Problem Solver

Problem: plot the motion of $n$ unit masses in 3D space due to their inter-gravitational forces as fast as possible. With CUDA!

- Initial iteration: `nbody_naive_kernels.cu`
- Coalescing global memory access: `nbody_coalesced.cu`
- Optimizing shared memory: `nbody_sharedMem.cu`
- Leveraging a Bounding Volume Hierarchy (BVH): `nbody_bvh.cu`
