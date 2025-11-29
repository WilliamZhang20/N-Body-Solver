# N-Body Problem Solver

Problem: plot the motion of $n$ unit masses in 3D space due to their inter-gravitational forces as fast as possible. With CUDA!

- Initial iteration: `nbody_naive_kernels.cu`
- Coalescing global memory access: `nbody_coalesced.cu`
- Optimizing shared memory: `nbody_sharedMem.cu`
- Leveraging a Bounding Volume Hierarchy (BVH): `nbody_bvh.cu`
- Particle Mesh Method with CuFFT: `nbody_particle_mesh.cu`

The BVH implementation is by far the hardest, longest, but most interesting and theory-heavy implementation.

Closely mirrors radix-tree implementation in [this](https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf) 2012 paper by Tero Karras.

## Getting Started

To compile, run:
```Bash
nvcc -std=c++14 -o nbody 09-nbody/01-nbody.cu
```

To debug in Jupyter, run:
```Bash
%%writefile gdb_commands.txt
# Set breakpoints at the start of your kernel and immediately before launch
break refit_bvh_bottom_up
break 09-nbody/01-nbody.cu:586
handle SIGSEGV stop noprint
handle SIGABRT stop noprint
run
continue
continue
# bt
quit
```

with:

```Bash
nvcc -g -lineinfo -std=c++14 -o nbody 09-nbody/01-nbody.cu
```

and:

```Bash
cuda-gdb -batch -x gdb_commands.txt --args ./nbody
```

To profile, run:
```Bash
nsys profile --stats=true --force-overwrite=true -o nbody-report ./nbody
```