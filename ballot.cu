#include <cooperative_groups.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>

#define GRPSIZE 32
using namespace cooperative_groups;
const int nTPB = 32;
// __device__ int reduce(thread_group g, int *x, int val) { 
//   int lane = g.thread_rank();
//   for (int i = g.size()/2; i > 0; i /= 2) {
//     x[lane] = val;       g.sync();
//     if (lane < i) val += x[lane + i];  g.sync();
//   }
//   if (g.thread_rank() == 0) printf("group partial sum: %d\n", val);
//   return val;
// }
namespace hipex{

    __device__ uint32_t reduce(cooperative_groups::thread_group &g, uint32_t* target){
        
        int lane = (int)g.thread_rank();
        uint32_t val = target[lane];
        for(int i = g.size()/2; i > 0; i /=2){
            target[lane] = val;
            g.sync();
            if(lane < i)
                val |= target[lane+i]; 
            g.sync();
        }
        if(lane == 0)
            target[lane] = val;
        g.sync();
    }

    __device__ uint32_t ohc_id(uint32_t idx){
        uint32_t ohc_ = 0;
        ohc_ |= 1U << idx;

        return ohc_;
    }

    // template<Typename T> make it templated later, may be only uptill 32 buts for now
    __device__ uint32_t ballot(cooperative_groups::thread_group &g, int predicate){
        uint32_t ballot_ = 0;
        __shared__ uint32_t pred_sh[GRPSIZE]; // thread group sizes are limited to 32 for now
        
        assert(g.size() <= GRPSIZE);

        if(g.thread_rank() < GRPSIZE){
            pred_sh[g.thread_rank()] = 0;
            if(predicate != 0)
                pred_sh[g.thread_rank()] = ohc_id(g.thread_rank());
        }
        g.sync();
        reduce(g, pred_sh);
        ballot_ = pred_sh[0];
        return ballot_;
    }
}
__global__ void my_reduce_kernel(int *data){

//   __shared__ int sdata[nTPB];
  // task 1a: create a proper thread block group below
  auto g1 = this_thread_block();
  // auto g2 = tiled_partition<32>(g1);

  int test_b = 0;
  if(g1.thread_rank() == 31 || g1.thread_rank() == 30) {test_b = 12;}

  auto ballot = hipex::ballot(g1,test_b);
  // auto ballot = g2.ballot(test_b);

  printf("ballot from g1: %x from thread:%d\n",ballot, g1.thread_rank());

}

int main(){

  int *data;
  cudaMallocManaged(&data, nTPB*sizeof(data[0]));
  for (int i = 0; i < nTPB; i++) data[i] = 1;
  my_reduce_kernel<<<1,nTPB>>>(data);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) printf("cuda error: %s\n", cudaGetErrorString(err));
}