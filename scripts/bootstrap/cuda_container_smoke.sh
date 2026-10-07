#!/usr/bin/env bash
set -euo pipefail

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

cat > "$tmpdir/cuda_smoke.cu" <<'CU'
#include <cstdio>
#include <cuda_runtime.h>

__global__ void add_one(int* x) {
    x[0] += 1;
}

int main() {
    int h = 41;
    int* d = nullptr;

    cudaError_t err = cudaMalloc(&d, sizeof(int));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed: %s\n",
                     cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(d, &h, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "H2D failed: %s\n",
                     cudaGetErrorString(err));
        return 2;
    }

    add_one<<<1, 1>>>(d);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "kernel failed: %s\n",
                     cudaGetErrorString(err));
        return 3;
    }

    err = cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "D2H failed: %s\n",
                     cudaGetErrorString(err));
        return 4;
    }

    cudaFree(d);

    std::printf("CUDA result: %d\n", h);
    return h == 42 ? 0 : 5;
}
CU

echo "=== NVCC ==="
nvcc --version

echo
echo "=== BUILD CUDA SMOKE ==="
nvcc "$tmpdir/cuda_smoke.cu" -o "$tmpdir/cuda_smoke"

echo
echo "=== RUN CUDA SMOKE ==="
"$tmpdir/cuda_smoke"

echo
echo "=== CUDA CONTAINER SMOKE: PASS ==="
