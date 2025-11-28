# N-Body Problem Solver

Problem: plot the motion of $n$ unit masses in 3D space due to their inter-gravitational forces as fast as possible. With CUDA!

- Initial iteration: `nbody_naive_kernels.cu`
- Coalescing global memory access: `nbody_coalesced.cu`
- Optimizing shared memory: `nbody_sharedMem.cu`
- Leveraging a Bounding Volume Hierarchy (BVH): `nbody_bvh.cu`

The BVH implementation is by far the hardest, longest, but most interesting and theory-heavy implementation.

Closely mirrors radix-tree implementation in [this](https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf) 2012 paper by Tero Karras.