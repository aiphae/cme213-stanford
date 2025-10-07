#include <math_constants.h>

#include "BC.h"
#include "Grid.h"
#include "mp1-util.h"

/**
 * Calculates the next finite difference step given a
 * grid point and step lengths.
 *
 * @param curr Pointer to the grid point that should be updated.
 * @param width Number of grid points in the x dimension.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 * @returns Grid value of next timestep.
 */
template<int order>
__device__
float Stencil(const float* curr, int width, float xcfl, float ycfl) {
    switch(order) {
        case 2:
            return curr[0] + xcfl * (curr[-1] + curr[1] - 2.f * curr[0]) +
                   ycfl * (curr[width] + curr[-width] - 2.f * curr[0]);

        case 4:
            return curr[0] + xcfl * (-curr[2] + 16.f * curr[1] - 30.f * curr[0]
                                     + 16.f * curr[-1] - curr[-2])
                           + ycfl * (- curr[2 * width] + 16.f * curr[width]
                                     - 30.f * curr[0] + 16.f * curr[-width]
                                     - curr[-2 * width]);

        case 8:
            return curr[0] + xcfl * (-9.f * curr[4] + 128.f * curr[3]
                                     - 1008.f * curr[2] + 8064.f * curr[1]
                                     - 14350.f * curr[0] + 8064.f * curr[-1]
                                     - 1008.f * curr[-2] + 128.f * curr[-3]
                                     - 9.f * curr[-4])
                           + ycfl * (-9.f * curr[4 * width]
                                     + 128.f * curr[3 * width]
                                     - 1008.f * curr[2 * width]
                                     + 8064.f * curr[width]
                                     - 14350.f * curr[0]
                                     + 8064.f * curr[-width]
                                     - 1008.f * curr[-2 * width]
                                     + 128.f * curr[-3 * width]
                                     - 9.f * curr[-4 * width]);

        default:
            printf("ERROR: Order %d not supported", order);
            return CUDART_NAN_F;
    }
}

/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be very simple and only use global memory
 * and 1d threads and blocks.
 *
 * @param next [out] Next grid state.
 * @param curr Current grid state.
 * @param gx   Number of grid points in the x dimension.
 * @param nx   Number of grid points in the x dimension to which the full
 *             stencil can be applied (ie the number of points that are at least
 *             order/2 grid points away from the boundary).
 * @param ny   Number of grid points in the y dimension to which th full
 *             stencil can be applied.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int order>
__global__
void gpuStencilGlobal(
    float* next, const float* __restrict__ curr, int gx, int nx, int ny, float xcfl, float ycfl
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int halo = order / 2;

    for (int idx = tid; idx < nx * ny; idx += stride) {
        const int x = tid % nx + halo;
        const int y = tid / ny + halo;
        const int out = y * gx + x;
        next[out] = Stencil<order>(curr + out, gx, xcfl, ycfl);
    }
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilGlobal kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
double gpuComputationGlobal(Grid& curr_grid, const simParams& params) {
    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    const int order = params.order();
    const int gx = curr_grid.gx();
    const int nx = curr_grid.gx() - order;
    const int ny = curr_grid.gy() - order;
    constexpr int threads_per_block = 256;
    const int num_blocks = (nx * ny + threads_per_block - 1) / threads_per_block;
    const float xcfl = params.xcfl();
    const float ycfl = params.ycfl();

    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        switch (order) {
            case 2:
                gpuStencilGlobal<2><<<num_blocks, threads_per_block>>>(
                    next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl
                );
                break;
            case 4:
                gpuStencilGlobal<2><<<num_blocks, threads_per_block>>>(
                    next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl
                );
                break;
            case 8:
                gpuStencilGlobal<2><<<num_blocks, threads_per_block>>>(
                    next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl
                );
                break;
            default:
                printf("Unsupported order\n");
                return 0;
        }

        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilGlobal");
    return stop_timer(&timer);
}

/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be optimized to compute finite difference updates
 * in blocks of size (blockDim.y * numYPerStep) * blockDim.x. Each thread
 * should calculate at most numYPerStep updates. It should still only use
 * global memory.
 *
 * @param next [out] Next grid state.
 * @param curr Current grid state.
 * @param gx   Number of grid points in the x dimension.
 * @param nx   Number of grid points in the x dimension to which the full
 *             stencil can be applied (ie the number of points that are at least
 *             order/2 grid points away from the boundary).
 * @param ny   Number of grid points in the y dimension to which th full
 *             stencil can be applied.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int order, int numYPerStep>
__global__
void gpuStencilBlock(
    float* next, const float* __restrict__ curr, int gx, int nx, int ny, float xcfl, float ycfl
) {
    const int halo = order / 2;
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x + halo;
    const int start_global_y = (blockIdx.y * blockDim.y + threadIdx.y) * numYPerStep + halo;

    for (int i = 0; i < numYPerStep; ++i) {
        if (const int global_y = start_global_y + i; global_y < nx + halo && global_y < ny + halo) {
            const float *curr_ptr = curr + global_y * gx + global_x;
            next[global_y * gx + global_x] = Stencil<order>(curr_ptr, gx, xcfl, ycfl);
        }
    }
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilBlock kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
double gpuComputationBlock(Grid& curr_grid, const simParams& params) {
    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    const int order = params.order();
    const int gx = curr_grid.gx();
    const int gy = curr_grid.gy();
    const int nx = gx - order;
    const int ny = gy - order;
    constexpr int numYPerStep = 4;
    const float xcfl = params.xcfl();
    const float ycfl = params.ycfl();

    const dim3 threads(32, 4);
    const dim3 blocks(
    (nx + threads.x - 1) / threads.x,
    (ny + threads.y * numYPerStep - 1) / (threads.y * numYPerStep)
    );

    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);
        gpuStencilBlock<2, numYPerStep><<<blocks, threads>>>(
            next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl
        );
        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilBlock");
    return stop_timer(&timer);
}


/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be optimized to compute finite difference updates
 * in blocks of size side * side using shared memory.
 *
 * @param next [out] Next grid state.
 * @param curr Current grid state.
 * @param gx   Number of grid points in the x dimension.
 * @param gy   Number of grid points in the y dimension.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int side, int order>
__global__
void gpuStencilShared(
    float* next,
    const float* __restrict__ curr,
    int gx, int gy,
    float xcfl, float ycfl
) {
    const int halo = order / 2;

    __shared__ float tile[side + order][side + order];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * side;
    int by = blockIdx.y * side;

    int gx_idx = bx + tx - halo;
    int gy_idx = by + ty - halo;

    gx_idx = min(max(gx_idx, 0), gx - 1);
    gy_idx = min(max(gy_idx, 0), gy - 1);

    tile[ty][tx] = curr[gy_idx * gx + gx_idx];

    __syncthreads();

    if (tx >= halo && tx < side + halo && ty >= halo && ty < side + halo) {
        int out_x = bx + (tx - halo);
        int out_y = by + (ty - halo);
        if (out_x >= halo && out_x < gx - halo && out_y >= halo && out_y < gy - halo) {
            float* currPtr = &tile[ty][tx];
            int sharedWidth = side + 2 * halo;
            float val = Stencil<order>(currPtr, sharedWidth, xcfl, ycfl);
            next[out_y * gx + out_x] = val;
        }
    }
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilShared kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
template<int order>
double gpuComputationShared(Grid& curr_grid, const simParams& params) {
    boundary_conditions BC(params);
    Grid next_grid(curr_grid);

    constexpr int side = 16;
    dim3 threads(side + order, side + order);
    dim3 blocks(
        (params.gx() + side - 1) / side,
        (params.gy() + side - 1) / side
    );

    event_pair timer;
    start_timer(&timer);

    for (int i = 0; i < params.iters(); ++i) {
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        gpuStencilShared<side, order><<<blocks, threads>>>(
            next_grid.dGrid_, curr_grid.dGrid_, params.gx(), params.gy(), params.xcfl(), params.ycfl()
        );

        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilShared");
    return stop_timer(&timer);
}
