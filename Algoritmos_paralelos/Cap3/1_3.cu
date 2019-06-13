#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

__global__ void matrixAdd(float* A, float* B, float* C, int M) {
    int i = threadIdx.x;
    int offset;
    for (int j=0; j<M; j++) {
        offset = j*M + i;
        C[offset] = A[offset] + B[offset];
    }
}

int main(void) {
    int M = 10;
    int numElements = M * M;
    size_t size = numElements * sizeof(float);

    //matrices en host
    float* h_A = (float*) malloc(size);
    float* h_B = (float*) malloc(size);
    float* h_C = (float*) malloc(size);

    int i, j, offset;
    for (i = 0; i <  M; i++) {
        for (j = 0; j < M; j++) {
            offset = i*M + j;
            h_A[offset] = 1.;
            h_B[offset] = 1.;
        }
    }
    //matrices en device
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    //copiar matrices a device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock( 32, 32 );    
    dim3 dimGrid( ceil((double)x/32), ceil((double)y/32) );
    matrixAdd<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M);

    // copiar resultados al host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // imprimir
    for (i = 0; i <  M; i++)
        for (j = 0; j < M; j++)
            printf("%f ", h_C[i*M + j]);

    // liberar memoria
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}