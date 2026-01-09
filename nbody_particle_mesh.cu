#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include "timer.h"
#include "files.h"
#define SOFTENING 1e-9f // Must match reference implementation exactly
#define G_CONSTANT 1.0f // Gravitational constant (implicit in reference code)
typedef struct { float x, y, z, vx, vy, vz; } Body;
#define CHECK_CUDA(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
      fprintf(stderr, "CUDA ERROR %s:%d: %s\n", \
              __FILE__, __LINE__, cudaGetErrorString(_e)); \
      cleanup(d_p, d_grid, d_green, d_fx, d_fy, d_fz, pHost, plan); \
      exit(1); \
    } \
} while(0)
#define CHECK_CUFFT(call) do { \
    cufftResult _r = (call); \
    if (_r != CUFFT_SUCCESS) { \
      fprintf(stderr, "CUFFT ERROR %s:%d: %d\n", \
              __FILE__, __LINE__, _r); \
      cleanup(d_p, d_grid, d_green, d_fx, d_fy, d_fz, pHost, plan); \
      exit(1); \
    } \
} while(0)
// Algorithm parameters
#define GRID_FACTOR 4.0f // Increased for better accuracy (was 2.0)
#define MIN_GRID 64 // Increased minimum (was 32)
#define MAX_GRID 512 // Increased maximum (was 256)
#define EPSILON 1e-10f
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
// Helper: next power of two
static int next_pow2(int v) {
    if (v <= 0) return 1;
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}
// Helper: check if a number is a power of two
static int is_power_of_two(int v) {
    return (v > 0) && ((v & (v - 1)) == 0);
}
// Cleanup function
static void cleanup(Body *d_p, cufftComplex *d_grid, cufftComplex *d_green,
                    float *d_fx, float *d_fy, float *d_fz, Body *pHost, cufftHandle plan) {
    if (d_p) cudaFree(d_p);
    if (d_grid) cudaFree(d_grid);
    if (d_green) cudaFree(d_green);
    if (d_fx) cudaFree(d_fx);
    if (d_fy) cudaFree(d_fy);
    if (d_fz) cudaFree(d_fz);
    if (pHost) cudaFreeHost(pHost);
    cufftDestroy(plan);
}
// Kernel: zero complex grid
__global__ void kernel_zero_grid(cufftComplex *grid, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        grid[idx].x = 0.0f;
        grid[idx].y = 0.0f;
    }
}
// Kernel: scale complex grid
__global__ void kernel_scale_complex(cufftComplex *data, float scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    data[idx].x *= scale;
    data[idx].y *= scale;
}
// Kernel: deposit mass to grid using Cloud-In-Cell (CIC)
__global__ void kernel_deposit_CIC(const Body *p, int n, cufftComplex *grid,
                                   int Nx, int Ny, int Nz,
                                   float xmin, float ymin, float zmin,
                                   float inv_dx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = p[i].x;
    float y = p[i].y;
    float z = p[i].z;
    // Map to grid coords
    float gx = (x - xmin) * inv_dx;
    float gy = (y - ymin) * inv_dx;
    float gz = (z - zmin) * inv_dx;
    int ix = floorf(gx);
    int iy = floorf(gy);
    int iz = floorf(gz);
    float dx = gx - ix;
    float dy = gy - iy;
    float dz = gz - iz;
    // Wrap indices for periodic boundaries
    int ix0 = (ix ) & (Nx - 1);
    int iy0 = (iy ) & (Ny - 1);
    int iz0 = (iz ) & (Nz - 1);
    int ix1 = (ix + 1) & (Nx - 1);
    int iy1 = (iy + 1) & (Ny - 1);
    int iz1 = (iz + 1) & (Nz - 1);
    // CIC weights (assuming unit mass per particle)
    float w000 = (1-dx)*(1-dy)*(1-dz);
    float w100 = dx*(1-dy)*(1-dz);
    float w010 = (1-dx)*dy*(1-dz);
    float w001 = (1-dx)*(1-dy)*dz;
    float w110 = dx*dy*(1-dz);
    float w101 = dx*(1-dy)*dz;
    float w011 = (1-dx)*dy*dz;
    float w111 = dx*dy*dz;
    #define IDX(ix,iy,iz) ((ix) + (iy)*Nx + (iz)*Nx*Ny)
    // Atomic add to real part (density)
    atomicAdd(&grid[IDX(ix0,iy0,iz0)].x, w000);
    atomicAdd(&grid[IDX(ix1,iy0,iz0)].x, w100);
    atomicAdd(&grid[IDX(ix0,iy1,iz0)].x, w010);
    atomicAdd(&grid[IDX(ix0,iy0,iz1)].x, w001);
    atomicAdd(&grid[IDX(ix1,iy1,iz0)].x, w110);
    atomicAdd(&grid[IDX(ix1,iy0,iz1)].x, w101);
    atomicAdd(&grid[IDX(ix0,iy1,iz1)].x, w011);
    atomicAdd(&grid[IDX(ix1,iy1,iz1)].x, w111);
    #undef IDX
}
// Kernel: multiply in k-space by Green's function
__global__ void kernel_multiply_green(cufftComplex *grid_k, const cufftComplex *green_k, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    cufftComplex a = grid_k[idx];
    cufftComplex g = green_k[idx];
    // Complex multiply
    cufftComplex out;
    out.x = a.x * g.x - a.y * g.y;
    out.y = a.x * g.y + a.y * g.x;
    grid_k[idx] = out;
}
// Kernel: compute forces on grid from potential using central differences
__global__ void kernel_compute_force_from_potential(const cufftComplex *pot,
                                                   float *fx, float *fy, float *fz,
                                                   int Nx, int Ny, int Nz,
                                                   float inv_dx)
{
    int idx3 = blockIdx.x * blockDim.x + threadIdx.x;
    int N = Nx*Ny*Nz;
    if (idx3 >= N) return;
    int ix = idx3 % Nx;
    int iy = (idx3 / Nx) % Ny;
    int iz = idx3 / (Nx*Ny);
    // Periodic neighbor indices
    int ixm = (ix - 1 + Nx) & (Nx - 1);
    int ixp = (ix + 1) & (Nx - 1);
    int iym = (iy - 1 + Ny) & (Ny - 1);
    int iyp = (iy + 1) & (Ny - 1);
    int izm = (iz - 1 + Nz) & (Nz - 1);
    int izp = (iz + 1) & (Nz - 1);
    int idx_xm = ixm + iy*Nx + iz*Nx*Ny;
    int idx_xp = ixp + iy*Nx + iz*Nx*Ny;
    int idx_ym = ix + iym*Nx + iz*Nx*Ny;
    int idx_yp = ix + iyp*Nx + iz*Nx*Ny;
    int idx_zm = ix + iy*Nx + izm*Nx*Ny;
    int idx_zp = ix + iy*Nx + izp*Nx*Ny;
    // Potential stored in .x
    float phi_xm = pot[idx_xm].x;
    float phi_xp = pot[idx_xp].x;
    float phi_ym = pot[idx_ym].x;
    float phi_yp = pot[idx_yp].x;
    float phi_zm = pot[idx_zm].x;
    float phi_zp = pot[idx_zp].x;
    // Gradient via central difference
    float gx = (phi_xp - phi_xm) * 0.5f * inv_dx;
    float gy = (phi_yp - phi_ym) * 0.5f * inv_dx;
    float gz = (phi_zp - phi_zm) * 0.5f * inv_dx;
    // Force is negative gradient
    fx[idx3] = -gx;
    fy[idx3] = -gy;
    fz[idx3] = -gz;
}
// Kernel: Interpolate force to particles (CIC gather) and update
__global__ void kernel_apply_force_CIC(Body *p, int n,
                                       const float *fx, const float *fy, const float *fz,
                                       int Nx, int Ny, int Nz,
                                       float xmin, float ymin, float zmin,
                                       float Lx, float Ly, float Lz,
                                       float inv_dx, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = p[i].x;
    float y = p[i].y;
    float z = p[i].z;
    float gx = (x - xmin) * inv_dx;
    float gy = (y - ymin) * inv_dx;
    float gz = (z - zmin) * inv_dx;
    int ix = floorf(gx);
    int iy = floorf(gy);
    int iz = floorf(gz);
    float dx = gx - ix;
    float dy = gy - iy;
    float dz = gz - iz;
    int ix0 = (ix ) & (Nx - 1);
    int iy0 = (iy ) & (Ny - 1);
    int iz0 = (iz ) & (Nz - 1);
    int ix1 = (ix + 1) & (Nx - 1);
    int iy1 = (iy + 1) & (Ny - 1);
    int iz1 = (iz + 1) & (Nz - 1);
    #define IDX(ix,iy,iz) ((ix) + (iy)*(Nx) + (iz)*(Nx)*(Ny))
    float w000 = (1-dx)*(1-dy)*(1-dz);
    float w100 = dx*(1-dy)*(1-dz);
    float w010 = (1-dx)*dy*(1-dz);
    float w001 = (1-dx)*(1-dy)*dz;
    float w110 = dx*dy*(1-dz);
    float w101 = dx*(1-dy)*dz;
    float w011 = (1-dx)*dy*dz;
    float w111 = dx*dy*dz;
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
    Fx += w000 * fx[IDX(ix0,iy0,iz0)];
    Fx += w100 * fx[IDX(ix1,iy0,iz0)];
    Fx += w010 * fx[IDX(ix0,iy1,iz0)];
    Fx += w001 * fx[IDX(ix0,iy0,iz1)];
    Fx += w110 * fx[IDX(ix1,iy1,iz0)];
    Fx += w101 * fx[IDX(ix1,iy0,iz1)];
    Fx += w011 * fx[IDX(ix0,iy1,iz1)];
    Fx += w111 * fx[IDX(ix1,iy1,iz1)];
    Fy += w000 * fy[IDX(ix0,iy0,iz0)];
    Fy += w100 * fy[IDX(ix1,iy0,iz0)];
    Fy += w010 * fy[IDX(ix0,iy1,iz0)];
    Fy += w001 * fy[IDX(ix0,iy0,iz1)];
    Fy += w110 * fy[IDX(ix1,iy1,iz0)];
    Fy += w101 * fy[IDX(ix1,iy0,iz1)];
    Fy += w011 * fy[IDX(ix0,iy1,iz1)];
    Fy += w111 * fy[IDX(ix1,iy1,iz1)];
    Fz += w000 * fz[IDX(ix0,iy0,iz0)];
    Fz += w100 * fz[IDX(ix1,iy0,iz0)];
    Fz += w010 * fz[IDX(ix0,iy1,iz0)];
    Fz += w001 * fz[IDX(ix0,iy0,iz1)];
    Fz += w110 * fz[IDX(ix1,iy1,iz0)];
    Fz += w101 * fz[IDX(ix1,iy0,iz1)];
    Fz += w011 * fz[IDX(ix0,iy1,iz1)];
    Fz += w111 * fz[IDX(ix1,iy1,iz1)];
    // Update velocity (assuming unit mass)
    p[i].vx += dt * Fx;
    p[i].vy += dt * Fy;
    p[i].vz += dt * Fz;
    // Update position
    p[i].x += p[i].vx * dt;
    p[i].y += p[i].vy * dt;
    p[i].z += p[i].vz * dt;
    // CRITICAL FIX: Apply periodic boundary conditions
    // Wrap positions back into domain [xmin, xmax]
    if (p[i].x < xmin) p[i].x += Lx;
    if (p[i].x >= xmin + Lx) p[i].x -= Lx;
    if (p[i].y < ymin) p[i].y += Ly;
    if (p[i].y >= ymin + Ly) p[i].y -= Ly;
    if (p[i].z < zmin) p[i].z += Lz;
    if (p[i].z >= zmin + Lz) p[i].z -= Lz;
    #undef IDX
}
// Precompute Green's function for Poisson solver (FIX: Use discrete version for accuracy)
void precompute_green(cufftComplex *green_h, int Nx, int Ny, int Nz, float L) {
    float fac = L * L / (M_PI);  // Common factor (positive)
    for (int iz = 0; iz < Nz; ++iz) {
        int kz = (iz <= Nz/2) ? iz : (iz - Nz);
        float kz2 = (float)(kz * kz);
        for (int iy = 0; iy < Ny; ++iy) {
            int ky = (iy <= Ny/2) ? iy : (iy - Ny);
            float ky2 = (float)(ky * ky);
            for (int ix = 0; ix < Nx; ++ix) {
                int idx = ix + iy*Nx + iz*Nx*Ny;

                if (ix == 0 && iy == 0 && iz == 0) {
                    green_h[idx].x = 0.0f;
                    green_h[idx].y = 0.0f;
                    continue;
                }

                int kx = (ix <= Nx/2) ? ix : (ix - Nx);
                float kx2 = (float)(kx * kx);

                float k2 = kx2 + ky2 + kz2;
                if (k2 < EPSILON) {
                    green_h[idx].x = 0.0f;
                    green_h[idx].y = 0.0f;
                    continue;
                }

                // G(k) = - L² / (π k²)   → negative for attractive gravity with +4π ρ
                float val = - fac / k2;

                green_h[idx].x = val;
                green_h[idx].y = 0.0f;
            }
        }
    }
}
int main(const int argc, const char** argv) {
    int nBodies = 2<<11;
    if (argc > 1) nBodies = 2<<atoi(argv[1]);
    const char *initialized_values;
    const char *solution_values;
    if (nBodies == 2 << 11) {
        initialized_values = "09-nbody/files/initialized_4096";
        solution_values = "09-nbody/files/solution_4096";
    } else {
        initialized_values = "09-nbody/files/initialized_65536";
        solution_values = "09-nbody/files/solution_65536";
    }
    if (argc > 2) initialized_values = argv[2];
    if (argc > 3) solution_values = argv[3];
    const float dt = 0.01f; // Match original black-box test parameters
    const int nIters = 10;
    int bytes = nBodies * sizeof(Body);
    Body *pHost = NULL;
    Body *d_p = NULL;
    cufftComplex *d_grid = NULL, *d_green = NULL;
    float *d_fx = NULL, *d_fy = NULL, *d_fz = NULL;
    cufftHandle plan = 0;
    CHECK_CUDA(cudaMallocHost((void**)&pHost, bytes));
    read_values_from_file(initialized_values, (float*)pHost, bytes);
    // Compute bounding box and make it cubic for PM method
    float xmin = pHost[0].x, ymin = pHost[0].y, zmin = pHost[0].z;
    float xmax = xmin, ymax = ymin, zmax = zmin;
    for (int i = 1; i < nBodies; ++i) {
        if (pHost[i].x < xmin) xmin = pHost[i].x;
        if (pHost[i].y < ymin) ymin = pHost[i].y;
        if (pHost[i].z < zmin) zmin = pHost[i].z;
        if (pHost[i].x > xmax) xmax = pHost[i].x;
        if (pHost[i].y > ymax) ymax = pHost[i].y;
        if (pHost[i].z > zmax) zmax = pHost[i].z;
    }
   
    // Find maximum extent
    float Lx_raw = xmax - xmin;
    float Ly_raw = ymax - ymin;
    float Lz_raw = zmax - zmin;
    float L_max = fmaxf(fmaxf(Lx_raw, Ly_raw), Lz_raw);
   
    // Add padding
    float pad = 0.1f * L_max; // 10% padding
    if (pad < 0.1f) pad = 0.1f; // Minimum padding
   
    // Make cubic domain centered on particles
    float center_x = 0.5f * (xmin + xmax);
    float center_y = 0.5f * (ymin + ymax);
    float center_z = 0.5f * (zmin + zmax);
   
    float Lx = L_max + 2.0f * pad;
    float Ly = Lx; // Cubic!
    float Lz = Lx;
   
    xmin = center_x - 0.5f * Lx;
    ymin = center_y - 0.5f * Ly;
    zmin = center_z - 0.5f * Lz;
    // Grid size
    int approx = ceilf(powf((float)nBodies, 1.0f/3.0f) * GRID_FACTOR);
    int Nx = next_pow2(approx);
    if (Nx < MIN_GRID) Nx = MIN_GRID;
    if (Nx > MAX_GRID) Nx = MAX_GRID;
    int Ny = Nx, Nz = Nx;
    int Ngrid = Nx * Ny * Nz;
    assert(is_power_of_two(Nx) && is_power_of_two(Ny) && is_power_of_two(Nz));
    fprintf(stderr, "PM parameters: N=%d, grid=%d^3, L=(%.3f,%.3f,%.3f)\n",
            nBodies, Nx, Lx, Ly, Lz);
    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_p, bytes));
    CHECK_CUDA(cudaMemcpy(d_p, pHost, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc((void**)&d_grid, sizeof(cufftComplex) * Ngrid));
   
    cufftComplex *h_green = (cufftComplex*)malloc(sizeof(cufftComplex) * Ngrid);
    float dx = Lx / (float)Nx;  // FIX: Compute dx here for precompute_green
    precompute_green(h_green, Nx, Ny, Nz, Lx);  // FIX: Pass dx (cubic)
    CHECK_CUDA(cudaMalloc((void**)&d_green, sizeof(cufftComplex) * Ngrid));
    CHECK_CUDA(cudaMemcpy(d_green, h_green, sizeof(cufftComplex) * Ngrid, cudaMemcpyHostToDevice));
    free(h_green);
    CHECK_CUDA(cudaMalloc((void**)&d_fx, sizeof(float) * Ngrid));
    CHECK_CUDA(cudaMalloc((void**)&d_fy, sizeof(float) * Ngrid));
    CHECK_CUDA(cudaMalloc((void**)&d_fz, sizeof(float) * Ngrid));
    CHECK_CUFFT(cufftPlan3d(&plan, Nx, Ny, Nz, CUFFT_C2C));
    const int TPB = 256;
    int nBlocksParticles = (nBodies + TPB - 1) / TPB;
    int nBlocksGrid = (Ngrid + TPB - 1) / TPB;
    float inv_dx = 1.0f / dx;
    float cell_volume = dx * dx * dx;
    double totalTime = 0.0;
    // Save initial velocities for debug
    float init_vx0 = pHost[0].vx;
    float init_vy0 = pHost[0].vy;
    float init_vz0 = pHost[0].vz;
    float init_vx1 = (nBodies > 1) ? pHost[1].vx : 0.0f;
    float init_vy1 = (nBodies > 1) ? pHost[1].vy : 0.0f;
    float init_vz1 = (nBodies > 1) ? pHost[1].vz : 0.0f;
    for (int iter = 0; iter < nIters; ++iter) {
        StartTimer();
        // 1) Zero grid
        kernel_zero_grid<<<nBlocksGrid, TPB>>>(d_grid, Ngrid);
        CHECK_CUDA(cudaGetLastError());
        // 2) Deposit mass (deposits mass, sum = nBodies)
        kernel_deposit_CIC<<<nBlocksParticles, TPB>>>(d_p, nBodies, d_grid, Nx, Ny, Nz,
                                                     xmin, ymin, zmin, inv_dx);
        CHECK_CUDA(cudaGetLastError());
        // FIX: Scale to density ρ = mass / cell_volume
        kernel_scale_complex<<<nBlocksGrid, TPB>>>(d_grid, 1.0f / cell_volume, Ngrid);
        CHECK_CUDA(cudaGetLastError());
        // 3) Forward FFT
        CHECK_CUFFT(cufftExecC2C(plan, d_grid, d_grid, CUFFT_FORWARD));
        // 4) Multiply by Green's function
        kernel_multiply_green<<<nBlocksGrid, TPB>>>(d_grid, d_green, Ngrid);
        CHECK_CUDA(cudaGetLastError());
        // 5) Inverse FFT
        CHECK_CUFFT(cufftExecC2C(plan, d_grid, d_grid, CUFFT_INVERSE));
        // 6) Scale by 1/N (FIX: Positive now, since green handles negative sign)
        float scale = 1.0f / (float)Ngrid;
        kernel_scale_complex<<<nBlocksGrid, TPB>>>(d_grid, scale, Ngrid);
        CHECK_CUDA(cudaGetLastError());
        // 7) Compute forces
        kernel_compute_force_from_potential<<<nBlocksGrid, TPB>>>(d_grid, d_fx, d_fy, d_fz,
                                                                 Nx, Ny, Nz, inv_dx);
        CHECK_CUDA(cudaGetLastError());
        // 8) Apply forces and integrate
        kernel_apply_force_CIC<<<nBlocksParticles, TPB>>>(d_p, nBodies, d_fx, d_fy, d_fz,
                                                         Nx, Ny, Nz, xmin, ymin, zmin,
                                                         Lx, Ly, Lz, inv_dx, dt);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;

        // Debug: After first step, copy back and print velocities (to see acceleration applied)
        if (iter == 0) {
            CHECK_CUDA(cudaMemcpy(pHost, d_p, bytes, cudaMemcpyDeviceToHost));
            fprintf(stderr, "Debug after first step:\n");
            fprintf(stderr, "Particle 0: vx=%.6f (delta=%.6f, acc=%.6f), vy=%.6f (delta=%.6f, acc=%.6f), vz=%.6f (delta=%.6f, acc=%.6f)\n",
                    pHost[0].vx, pHost[0].vx - init_vx0, (pHost[0].vx - init_vx0)/dt,
                    pHost[0].vy, pHost[0].vy - init_vy0, (pHost[0].vy - init_vy0)/dt,
                    pHost[0].vz, pHost[0].vz - init_vz0, (pHost[0].vz - init_vz0)/dt);
            if (nBodies > 1) {
                fprintf(stderr, "Particle 1: vx=%.6f (delta=%.6f, acc=%.6f), vy=%.6f (delta=%.6f, acc=%.6f), vz=%.6f (delta=%.6f, acc=%.6f)\n",
                        pHost[1].vx, pHost[1].vx - init_vx1, (pHost[1].vx - init_vx1)/dt,
                        pHost[1].vy, pHost[1].vy - init_vy1, (pHost[1].vy - init_vy1)/dt,
                        pHost[1].vz, pHost[1].vz - init_vz1, (pHost[1].vz - init_vz1)/dt);
            }
            // Copy back to device to continue simulation (since we overwrote pHost)
            CHECK_CUDA(cudaMemcpy(d_p, pHost, bytes, cudaMemcpyHostToDevice));
        }
    }
    double avgTime = totalTime / (double)nIters;
    float billionsOfOpsPerSecond = 1e-9f * nBodies * (float)Ngrid / avgTime;
   
    CHECK_CUDA(cudaMemcpy(pHost, d_p, bytes, cudaMemcpyDeviceToHost));
    write_values_to_file(solution_values, (float*)pHost, bytes);
   
    printf("%0.3f Billion 'interaction-grid' ops / second (PM heuristic)\n",
           billionsOfOpsPerSecond);
    cleanup(d_p, d_grid, d_green, d_fx, d_fy, d_fz, pHost, plan);
    return 0;
}
