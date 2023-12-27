#include <stdio.h>
#include <assert.h>

void initWith(float num, float *a, int N)
{
//    int i = threadIdx.x;
    int i = 0;
    for(i = 0; i < N; i++)
        a[i] = num;
}

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{
    
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < N)
  {
    result[idx] = a[idx] + b[idx];
  }
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main()
{
  const int N = 2<<20;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;
  
//  a = (float *)malloc(size);
  cudaMallocManaged(&a, size);
  
//  b = (float *)malloc(size);
  cudaMallocManaged(&b, size);

//  c = (float *)malloc(size);
  cudaMallocManaged(&c, size);

 initWith(3, a, N);
//  initWith<<<1, N>>>(3, a);
  
  initWith(4, b, N);
//  initWith<<<1, N>>>(4, b);
  
  initWith(0, c, N);
//  initWith<<<1, N>>>(0, c);


//  addVectorsInto(c, a, b, N);

  size_t threads_per_block = 256;
  size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
    
  addVectorsInto<<<number_of_blocks,threads_per_block>>>(c, a, b, N);
  checkCuda( cudaDeviceSynchronize() );

  checkElementsAre(7, c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
