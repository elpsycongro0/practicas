#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

__global__ void matrixVecMultiply(float* A, float* B, float* C, int M) {
    int i = threadIdx.x;
    int offset;
    float sum = 0;
    for (int j=0; j<M; j++) {
        offset = i*M + j;
        sum += A[offset] * B[j];
    }
    C[i] = sum;
}

int main(void) {
    int M = 3;
    size_t sizeMatrix = M * M * sizeof(float);
    size_t sizeVec = M * sizeof(float);

    //matrices en host
    float* h_A = (float*) malloc(sizeMatrix);
    float* h_B = (float*) malloc(sizeVec);
    float* h_C = (float*) malloc(sizeVec);

    int i, j, offset;
    float count1, count2 = 0.;
    for (i = 0; i <  M; i++) {
        for (j = 0; j < M; j++) {
            offset = i*M + j;
            h_A[offset] = ++count1;
        }
        h_B[i] = ++count2;
    }

    // imprimir matrices
    printf("A: ");
    for (i = 0; i <  M; i++)
        for (j = 0; j < M; j++)
            printf("%f ", h_A[i*M + j]);
    printf("\nB: ");
    for (i = 0; i <  M; i++)
            printf("%f ", h_B[i]);

    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc((void**)&d_A, sizeMatrix);
    cudaMalloc((void**)&d_B, sizeVec);
    cudaMalloc((void**)&d_C, sizeVec);

    // matrices a device
    cudaMemcpy(d_A, h_A, sizeMatrix, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeVec, cudaMemcpyHostToDevice);

    int numThreads = M;
    int numBlocks = 1;
    matrixVecMultiply<<<numBlocks, numThreads>>>(d_A, d_B, d_C, M);

    // matrices a host
    cudaMemcpy(h_C, d_C, sizeVec, cudaMemcpyDeviceToHost);

    // imprimir resultado
    printf("\nC: ");
    for (i = 0; i <  M; i++)
        printf("%f ", h_C[i]);
    printf("\n");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}