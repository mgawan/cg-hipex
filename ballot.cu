#include <cooperative_groups.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include "./include/hipex.hpp"

using namespace cooperative_groups;
const int nTPB = 32;
__global__ void my_reduce_kernel(int *data){

  //   __shared__ int sdata[nTPB];
    // task 1a: create a proper thread block group below
    auto g1 = this_thread_block();
    auto g2 = tiled_partition(g1,8);
    // auto g2 = tiled_partition<4>(g1);
  //   int test_b = 0;
  //   if(g1.thread_rank() == 31 || g1.thread_rank() == 30) {test_b = 12;}
  
  //   auto ballot = hipex::ballot(g1,test_b);
    // auto ballot = g2.ballot(test_b);
  
    // printf("thread id(g1): %d, thread id(g2): %d, meta_group_size(g1):%d, meta_grp_size(g2):%d, \
    // meta_group_id(g1):%d, meta_group_id(g2):%d my thread id:%d blocksize:%d, hipex:metasize(g2):%d\n",g1.thread_rank(), g2.thread_rank(), g1.size(), \
    // g2.size(), g1.thread_rank(), g2.thread_rank(), threadIdx.x, blockDim.x, hipex::meta_group_size(g2));
  
    printf("threadId:%d, meta_group_size:%d, group_rank:%d\n",threadIdx.x, hipex::meta_group_size(g2), hipex::meta_group_rank(g2));
    // printf("threadId:%d, meta_group_size:%d, group_rank:%d\n",threadIdx.x, g2.meta_group_size(), g2.meta_group_rank());
  }
  

int main(){

  int *data;
  cudaMallocManaged(&data, nTPB*sizeof(data[0]));
  for (int i = 0; i < nTPB; i++) data[i] = 1;
  my_reduce_kernel<<<1,nTPB>>>(data);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) printf("cuda error: %s\n", cudaGetErrorString(err));
}