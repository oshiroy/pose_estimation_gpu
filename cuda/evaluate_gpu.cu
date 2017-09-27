#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/replace.h>
#include <thrust/execution_policy.h>

#include "include/evaluate_gpu.h"


using namespace std;

texture<float4, cudaTextureType2D, cudaReadModeElementType> texRef;
cudaArray *cuarr;

struct is_more_than_thre
{
  float thre;
  __host__ __device__
  bool operator()(float x)
  {
    return x > thre;
  }
};


__device__ static float atomicMax(float* address, float val)
{
  int* address_as_i = (int*) address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed,
                      __float_as_int(::fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}



__device__ static float atomicMin(float* address, float val)
{
  int* address_as_i = (int*) address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed,
                      __float_as_int(::fminf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}


__global__
void calc_obj_score_map(const float*  __restrict__ depth, const float* __restrict__ mask,
                        int* mask_cnt_arr, int* pred_cnt_arr, int imw, int imh)
{
    int x = threadIdx.x;
    int y = blockIdx.x;
    int offset = x + y * imw;
    float4 sample =  tex2D(texRef, x, imh - y); // for convert coords gl -> cv

    __shared__ int nonzero_cnt, mask_cnt;
    if(threadIdx.x == 0){
      nonzero_cnt = 0;
      mask_cnt = 0;
    }
    __syncthreads();
    if (depth[offset] != 0 && mask[offset] != 0){
      atomicAdd(&mask_cnt, 1);

      if(sample.x != 0){
        atomicAdd(&nonzero_cnt, 1);
      }
    }
    __syncthreads();
    mask_cnt_arr[y] = mask_cnt;
    pred_cnt_arr[y] = nonzero_cnt;
}


__global__
void accumulate_obj_score(int* mask_cnt_arr, int* nonzero_cnt_arr, float* ret){
  __shared__ int sum_mask, sum_nonzero;
  int x = threadIdx.x;

  if(threadIdx.x == 0){
    sum_nonzero = 0;
    sum_mask = 0;
  }

  atomicAdd(&sum_mask, mask_cnt_arr[x]);
  atomicAdd(&sum_nonzero, nonzero_cnt_arr[x]);
  __syncthreads();

  if(threadIdx.x == 0){
    if(sum_mask != 0){
      *ret = 1.0 / (1.0 + sum_nonzero / (float)sum_mask);
    }
    else{
      *ret = 1e10;
    }
  }
  __syncthreads();
}


__global__
void calc_sorted_visib_map(const float*  __restrict__ depth, const float* __restrict__ mask,
                           float* ret, int* count_arr, int imw, int imh)
{
    int x = threadIdx.x;
    int y = blockIdx.x;
    int offset = x + y * imw;
    float4 sample =  tex2D(texRef, x, imh - y); // for convert coords gl -> cv

    __shared__ int nonzero_cnt, old, idx, i_iter;
    if(threadIdx.x == 0){
      nonzero_cnt = 0;
      i_iter = 0;
    }
    ret[offset] = 0; // initalization
    __syncthreads();
    if (sample.x != 0 && depth[offset] != 0 && mask[offset] != 0){
      old = atomicAdd(&nonzero_cnt, 1);
      ret[old + y * imw] = sample.x - depth[offset];
    }
    __syncthreads();

    // absolute selection sort per block
    __shared__ float min_val;
    if(threadIdx.x == 0){
      count_arr[y] = nonzero_cnt;
    }

    while(i_iter < nonzero_cnt){
      if(threadIdx.x == 0){
        idx = i_iter + y * imw;
        min_val = fabsf(ret[idx]);
      }
      __syncthreads();
      if(x > i_iter  && x < nonzero_cnt){
        atomicMin(&min_val, fabsf(ret[offset]));
      }
      __syncthreads();
      if(min_val == fabsf(ret[offset])){
        ret[offset] = ret[idx];
        ret[idx] = min_val;
      }
      if(threadIdx.x == 0) ++i_iter;
      __syncthreads();
    }
}


__global__
void calc_sorted_invisib_map(const float* __restrict__ depth, const float* __restrict__ mask,
                             float* ret, int* count_arr, int imw, int imh)
{
    int x = threadIdx.x;
    int y = blockIdx.x;
    int offset = x + y * imw;
    float4 sample =  tex2D(texRef, x, imh - y); // for convert coords gl -> cv
    __shared__ float min_val;
    __shared__ int nonzero_cnt, idx, old, i_iter;
    if(threadIdx.x == 0){
      nonzero_cnt = 0;
      i_iter = 0;
    }
    ret[offset] = 0; // initalization
    __syncthreads();
    if (sample.x != 0 && depth[offset] != 0 && mask[offset] == 0){
      old = atomicAdd(&nonzero_cnt, 1);
      ret[old + y * imw] = sample.x - depth[offset];
      if(ret[old + y * imw] > 0.020){
        ret[old + y * imw] = 0;
      }
    }
    __syncthreads();

    // absolute selection sort per block
    if(threadIdx.x == 0){
      count_arr[y] = nonzero_cnt;
    }

    while(i_iter < nonzero_cnt){
      if(threadIdx.x == 0){
        idx = i_iter + y * imw;
        min_val = fabsf(ret[idx]);
      }
      __syncthreads();
      if(x > i_iter  && x < nonzero_cnt){
        atomicMin(&min_val, fabsf(ret[offset]));
      }
      __syncthreads();
      if(min_val == fabsf(ret[offset])){
        ret[offset] = fabsf(ret[idx]);
        ret[idx] = min_val;
      }
      if(threadIdx.x == 0) ++i_iter;
      __syncthreads();
    }
}


__global__
void percentile_remove(float* ret, int* count_arr, int* out_pthre, int imw, int imh){
  int y = threadIdx.x;
  __shared__ int pthre, cnt, i_iter;
  __shared__ float max_val;
  extern __shared__ int shared_arr[];
  if(threadIdx.x == 0){
    cnt = 0;
  }
  shared_arr[y] = count_arr[y];
  __syncthreads();
  if(shared_arr[y] > 0){
    atomicAdd(&cnt, shared_arr[y]);
  }
  __syncthreads();
  if(threadIdx.x == 0){
    pthre = cnt * 0.95;
    i_iter = 0;
    *out_pthre = pthre;
  }
  __syncthreads();
  int idx;
  while(i_iter < cnt - pthre){
    idx = shared_arr[y] - 1 + y * imw;
    if(threadIdx.x == 0){
      ++i_iter;
      max_val = 0;
    }
    __syncthreads();
    if(shared_arr[y] > 0){
      atomicMax(&max_val, ret[idx]);
    }
    __syncthreads();
    if(max_val <= ret[idx]){
      ret[idx] = 0;
      shared_arr[y] -= 1;
    }
    __syncthreads();
  }
  count_arr[y] = shared_arr[y];
}


__global__
void thre_row_sum(float* ret, float* score_array, int* count_array, float max_thre, int imw){
  int x = threadIdx.x;
  int y = blockIdx.x;

  __shared__ float score;
  if(threadIdx.x == 0) score = 0;
  __syncthreads();
  // sum score per block
  if(x < count_array[y]){
    atomicAdd(&score, fminf(ret[x + y * imw], max_thre));
  }
  __syncthreads();
  if(threadIdx.x == 0){
    score_array[y] = score;
  }
}

__global__
void calc_final(float* score_array, float* out_score, int* perthre){
  int y = threadIdx.x;

  __shared__ float score;
  if(threadIdx.x == 0){
    score = 0;
  }
  __syncthreads();
  // sum score per block
  atomicAdd(&score, score_array[y]);
  __syncthreads();
  if(threadIdx.x == 0){
    score = score / *perthre;
    // *out_score = score; // error?
    score_array[0] = score; //tmp impl
  }
  __syncthreads();
}


__global__
void calc_final_with_storage(float* visib_score_arr, float* invisib_score_arr,
                             float* storage, int* v_perthre, int* iv_perthre, int target_id){
  int y = threadIdx.x;
  __shared__ float v_score, iv_score;

  if(threadIdx.x == 0){
    v_score = 0;
    iv_score = 0;
  }
  __syncthreads();

  // sum score per block
  atomicAdd(&v_score, visib_score_arr[y]);
  atomicAdd(&iv_score, invisib_score_arr[y]);
  __syncthreads();

  if(threadIdx.x == 0){
    v_score = v_score / *v_perthre;
    iv_score = iv_score / *iv_perthre;
    storage[target_id * 2] = v_score;
    storage[target_id * 2 + 1] = iv_score;
    // printf("visib : %f, invisib : %f\n", v_score, iv_score);
    // printf("visib : %f, invisib : %f\n", storage[target_id * 2], storage[target_id * 2 + 1]);
  }
  __syncthreads();
}

__global__
void calc_final_with_storage_single(float* score_arr, float* storage, int* perthre,
                                    int target_id, int target_pos){
  int y = threadIdx.x;
  __shared__ float score;

  if(threadIdx.x == 0){
    score = 0;
  }
  __syncthreads();

  // sum score per block
  atomicAdd(&score, score_arr[y]);
  __syncthreads();

  if(threadIdx.x == 0){
    score = score / *perthre;
    storage[target_id * 2 + target_pos] = score;
  }
  __syncthreads();
}


__global__
void tex_test(float* ret, int imw, int imh)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  // float u = x / (float) imw;
  // float v = y / (float) imh;

  float4 sample =  tex2D(texRef, x, imh - y);
  ret[y * imw + x] = sample.x;
}


void CUDAManager::gpuInit(GLuint bufferID, int ih, int iw){
  cudaDeviceProp  prop;
  int device;
  memset( &prop, 0, sizeof( cudaDeviceProp ) );
  prop.major = 1;
  prop.minor = 0;
  cudaChooseDevice( &device, &prop );
  printf("[CUDAManager] cuda set device id : % d\n", device);
  cudaGLSetGLDevice( device );
  if (cudaGraphicsGLRegisterImage(&resource, bufferID, GL_RENDERBUFFER,
                                  cudaGraphicsMapFlagsReadOnly) != cudaSuccess){
    fprintf(stderr, "Error in registering rbo color with cuda\n");
  }
  // check zero copy memory
  if(prop.canMapHostMemory == 1){
    printf("[CUDAManager] Zero copy memory is supported\n");
  }
  else{
    printf("[CUDAManager] Zero copy memory is not supported\n");
  }

  im_h = ih;
  im_w = iw;

  // allocate memory for depth and mask
  cudaMalloc((void**)&depth_d, im_h * im_w * sizeof(float));
  cudaMalloc((void**)&mask_d, im_h * im_w * sizeof(float));

  // allocate memory
  cudaMalloc((void**)&score_d, sizeof(float));
  cudaMalloc((void**)&score_arr, im_h * im_w * sizeof(float));
  cudaMalloc((void**)&invisib_progress, im_h * im_w * sizeof(float));

  cudaMalloc((void**)&sum_arr, im_h * sizeof(float));
  cudaMalloc((void**)&iv_sum_arr, im_h * sizeof(float));

  cudaMalloc((void**)&height_arr, im_h * sizeof(int));
  cudaMalloc((void**)&invisib_cnt, im_h * sizeof(int));

  cudaMalloc((void**)&mask_cnts, im_h * sizeof(int));
  cudaMalloc((void**)&nonzero_cnts, im_h * sizeof(int));

  cudaMalloc((void**)&obj_score, sizeof(float));
  cudaMalloc((void**)&v_pthre, sizeof(int));
  cudaMalloc((void**)&iv_pthre, sizeof(int));

  cudaMallocHost((void**)&host_arr, im_h * im_w * sizeof(float));

  // cudaStreamCreate(&s1);
  // cudaStreamCreate(&s2);
  cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&s3, cudaStreamNonBlocking);

  printf("[CUDAManager] finish gpuInit\n");
}


void CUDAManager::update_gl_pixels(){
  cudaGraphicsMapResources( 1, &resource, 0);
  cudaGraphicsSubResourceGetMappedArray(&cuarr, resource, 0, 0);
}


void CUDAManager::unmap_resources(){
  cudaGraphicsUnmapResources(1, &resource, 0);
}


void CUDAManager::host_to_dev_depth(float* depth){
	cudaMemcpy( depth_d, depth, im_h * im_w * sizeof(float), cudaMemcpyHostToDevice);
}


void CUDAManager::host_to_dev_mask(float* mask){
	cudaMemcpy( mask_d, mask, im_h * im_w * sizeof(float), cudaMemcpyHostToDevice);
}

void CUDAManager::alloc_ret_score(int length){
  len_score = length;
  cudaFree(ret_score_d);
  cudaFreeHost(ret_score_h);

  cudaMalloc((void**)&ret_score_d, sizeof(float) * 2 * length);
  cudaMemset((void**)&ret_score_d, 0, sizeof(float) * 2 * length);
  cudaMallocHost((void**)&ret_score_h, sizeof(float) * 2 * length);
  cudaMemcpy(ret_score_h, ret_score_d, sizeof(float) * 2 * len_score, cudaMemcpyDeviceToHost);
}

void CUDAManager::transfer_score(float* ret_host){
  cudaMemcpy(ret_score_h, ret_score_d, sizeof(float) * 2 * len_score, cudaMemcpyDeviceToHost);
  memcpy(ret_host, ret_score_h, sizeof(float) * 2 * len_score);
  // ret_host = ret_score_h;
}


void CUDAManager::evaluate_visibility(int target_id, int pthre, float max_lim){
  cudaGraphicsMapResources( 1, &resource, 0);
  cudaGraphicsSubResourceGetMappedArray(&cuarr, resource, 0, 0);
  cudaBindTextureToArray(texRef, cuarr);

  cudaStreamSynchronize(0);

  calc_sorted_visib_map<<<im_h, im_w, 0, s1>>>(depth_d, mask_d, score_arr,
                                               height_arr, im_w, im_h);
  percentile_remove<<<1, im_h, sizeof(int)*im_h, s1>>>(score_arr, height_arr, v_pthre, im_w, im_h);
  thre_row_sum<<<im_h, im_w, 0, s1>>>(score_arr, sum_arr, height_arr, 0.10, im_w);
  calc_final_with_storage_single<<<1, im_h, 0, s1>>>(sum_arr, ret_score_d,
                                                     v_pthre, target_id, 0);

  calc_sorted_invisib_map<<<im_h, im_w, 0, s2>>>(depth_d, mask_d, invisib_progress,
                                                invisib_cnt, im_w, im_h);
  percentile_remove<<<1, im_h, sizeof(int)*im_h, s2>>>(invisib_progress,
                                                       invisib_cnt, iv_pthre, im_w, im_h);
  thre_row_sum<<<im_h, im_w, 0, s2>>>(invisib_progress, iv_sum_arr, height_arr, 0.10, im_w);
  calc_final_with_storage_single<<<1, im_h, 0, s2>>>(iv_sum_arr, ret_score_d,
                                                     iv_pthre, target_id, 1);

  cudaStreamSynchronize(s1);
  cudaStreamSynchronize(s2);

  cudaUnbindTexture(texRef);
  cudaGraphicsUnmapResources(1, &resource, 0);
}


void CUDAManager::evaluate_visibility(float* score, int pthre, float max_lim){
  cudaGraphicsMapResources( 1, &resource, 0);
  cudaGraphicsSubResourceGetMappedArray(&cuarr, resource, 0, 0);
  cudaBindTextureToArray(texRef, cuarr);
  // score_d = 0;
  // dim3 grid(im_w/16, im_h/16);
  // dim3 block(16, 16);
  // tex_test<<<grid, block>>>(score_arr, im_w, im_h);

  calc_sorted_visib_map<<<im_h, im_w, 0, s1>>>(depth_d, mask_d, score_arr,
                                               height_arr, im_w, im_h);
  percentile_remove<<<1, im_h, sizeof(int)*im_h, s1>>>(score_arr, height_arr,
                                                       v_pthre, im_w, im_h);
  thre_row_sum<<<im_h, im_w, 0, s1>>>(score_arr, sum_arr, height_arr, 0.10, im_w);
  calc_final<<<1, im_h, 0, s1>>>(sum_arr, score_d, v_pthre);

  cudaMemcpyAsync(&host_arr[0], &sum_arr[0], sizeof(float), cudaMemcpyDeviceToHost, s1);

  calc_sorted_invisib_map<<<im_h, im_w, 0, s2>>>(depth_d, mask_d, invisib_progress,
                                                 invisib_cnt, im_w, im_h);
  percentile_remove<<<1, im_h, sizeof(int)*im_h, s2>>>(invisib_progress,
                                                       invisib_cnt, iv_pthre, im_w, im_h);
  thre_row_sum<<<im_h, im_w, 0, s2>>>(invisib_progress, iv_sum_arr, height_arr, 0.10, im_w);
  calc_final<<<1, im_h, 0, s2>>>(iv_sum_arr, score_d, iv_pthre);
  cudaMemcpyAsync(&host_arr[1], &iv_sum_arr[0], sizeof(float), cudaMemcpyDeviceToHost, s2);

  // calc obj score
  float obj_score_h;
  calc_obj_score_map<<<im_h, im_w, 0, s3>>>(depth_d, mask_d, mask_cnts, nonzero_cnts, im_w, im_h);
  accumulate_obj_score<<<1, im_h, 0, s3>>>(mask_cnts, nonzero_cnts, obj_score);
  cudaMemcpyAsync(&obj_score_h, obj_score, sizeof(float), cudaMemcpyDeviceToHost, s3);

  // cudaMemcpy(host_arr, score_arr, sizeof(float) * im_h * im_w, cudaMemcpyDeviceToHost);
  // for(int i = 0; i < im_h; i++){
  // for(int i = 0; i < 3; i++){
  //   for(int j = 0; j < im_w; j++){
  //     printf("%f, ", host_arr[i * im_w + j]);
  //   }
  //   printf("\n");
  // }
  // for(int i = 0; i < im_h; i++){
  //   for(int j = 0; j < im_w; j++){
  //     if(host_arr[i * im_w + j] != 0){
  //       printf("%f, ", host_arr[i * im_w + j]);
  //       // printf("(%d, %d)", i ,j);
  //     }

  //   }
  //   printf("\n");
  // }

  // int* tmp;
  // tmp = new int[im_h];
  // cudaMemcpy(tmp, height_arr, sizeof(int) * im_h, cudaMemcpyDeviceToHost);
  // for(int i = 0; i < im_h; i++){
  //   printf("%d\n", tmp[i]);
  // }
  // delete[] tmp;

  // float* tmp2;
  // tmp2 = new float[im_h];
  // cudaMemcpy(tmp2, sum_arr, sizeof(float) * im_h, cudaMemcpyDeviceToHost);
  // for(int i = 0; i < im_h; i++){
  //   printf("%f\n", tmp2[i]);
  // }
  // delete[] tmp2;

  cudaUnbindTexture(texRef);
  cudaGraphicsUnmapResources(1, &resource, 0);

  cudaStreamSynchronize(s1);
  cudaStreamSynchronize(s2);
  cudaStreamSynchronize(s3);

  *score = host_arr[0] * obj_score_h;
  // *score = host_arr[1];
  // printf("obj_score : %f\n", obj_score_h);
}


CUDAManager::CUDAManager(){}


CUDAManager::CUDAManager(GLuint bufferID, int ih, int iw){
  gpuInit(bufferID, ih, iw);
}


CUDAManager::~CUDAManager(){
  cudaGraphicsUnregisterResource(resource);

  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);

  cudaFree(depth_d);
  cudaFree(mask_d);
  cudaFree(score_arr);
  cudaFree(score_d);
  cudaFree(obj_score);
  cudaFree(height_arr);

  cudaFree(mask_cnts);
  cudaFree(nonzero_cnts);

  cudaFree(v_pthre);
  cudaFree(iv_pthre);

  cudaFree(ret_score_d);
  cudaFreeHost(ret_score_h);
  cudaFreeHost(host_arr);
}
