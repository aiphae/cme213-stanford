#ifndef PAGERANK_CUH
#define PAGERANK_CUH

#include "util.cuh"

/* 
 * Each kernel handles the update of one pagerank score. In other
 * words, each kernel handles one row of the update:
 *
 * pi(t + 1) = A pi(t) + (1 / (2N))
 *
 */
__global__
void device_graph_propagate(
    const uint *graph_indices,
    const uint *graph_edges,
    const float *graph_nodes_in,
    float *graph_nodes_out,
    const float *inv_edges_per_node,
    int num_nodes
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) {
        return;
    }

    float sum = 0.0f;
    for (uint i = graph_indices[idx]; i < graph_indices[idx + 1]; ++i) {
        sum += graph_nodes_in[graph_edges[i]] * inv_edges_per_node[graph_edges[i]];
    }

    graph_nodes_out[idx] = 0.5f / (float) num_nodes + 0.5f * sum;
}

/* 
 * This function executes a specified number of iterations of the
 * pagerank algorithm. The variables are:
 *
 * H_graph_indices, h_graph_edges:
 *     These arrays describe the indices of the neighbors of node i.
 *     Specifically, node i is adjacent to all nodes in the range
 *     h_graph_edges[h_graph_indices[i] ... h_graph_indices[i+1]].
 *
 * H_node_values_input:
 *     An initial guess of pi(0).
 *
 * h_gpu_node_values_output:
 *     Output array for the pagerank vector.
 *
 * H_inv_edges_per_node:
 *     The i'th element in this array is the reciprocal of the
 *     out degree of the i'th node.
 *
 * Nr_iterations:
 *     The number of iterations to run the pagerank algorithm for.
 *
 * Num_nodes:
 *     The number of nodes in the whole graph (ie N).
 *
 * Avg_edges:
 *     The average number of edges in the graph. You are guaranteed
 *     that the whole graph has num_nodes * avg_edges edges.
 */
double device_graph_iterate(
        const uint *h_graph_indices,
        const uint *h_graph_edges,
        const float *h_node_values_input,
        float *h_gpu_node_values_output,
        const float *h_inv_edges_per_node,
        int nr_iterations,
        int num_nodes,
        int avg_edges
) {
    // Allocate GPU memory
    uint *d_graph_indices, *d_graph_edges;
    float *buffer_1, *buffer_2, *d_inv_edges_per_node;

    cudaMalloc(&d_graph_indices, (num_nodes + 1) * sizeof(uint));
    cudaMalloc(&d_graph_edges, h_graph_indices[num_nodes] * sizeof(uint));
    cudaMalloc(&buffer_1, num_nodes * sizeof(float));
    cudaMalloc(&buffer_2, num_nodes * sizeof(float));
    cudaMalloc(&d_inv_edges_per_node, num_nodes * sizeof(float));

    cudaMemcpy(d_graph_indices, h_graph_indices, (num_nodes + 1) * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph_edges, h_graph_edges, h_graph_indices[num_nodes] * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(buffer_1, h_node_values_input, num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv_edges_per_node, h_inv_edges_per_node, num_nodes * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch
    event_pair timer;
    start_timer(&timer);

    int block_size = 192;
    int grid_size = (num_nodes + block_size - 1) / block_size;

    for (int iter = 0; iter < nr_iterations; ++iter) {
        device_graph_propagate<<<grid_size, block_size>>>(
                d_graph_indices,
                d_graph_edges,
                buffer_1,
                buffer_2,
                d_inv_edges_per_node,
                num_nodes
        );
        check_launch("gpu graph propagate");
        std::swap(buffer_1, buffer_2);
    }

    double gpu_time = stop_timer(&timer);

    // Copy result back
    cudaMemcpy(h_gpu_node_values_output, buffer_1, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_graph_indices);
    cudaFree(d_graph_edges);
    cudaFree(buffer_1);
    cudaFree(buffer_2);
    cudaFree(d_inv_edges_per_node);

    return gpu_time;
}

/**
 * This function computes the number of bytes read from and written to
 * global memory by the pagerank algorithm.
 * 
 * nodes:
 *      The number of nodes in the graph
 *
 * edges: 
 *      The average number of edges in the graph
 *
 * iterations:
 *      The number of iterations the pagerank algorithm was run
 */
uint get_total_bytes(uint nodes, uint edges, uint iterations) {
    return iterations * (edges * sizeof(float) + nodes * sizeof(float));
}

#endif
