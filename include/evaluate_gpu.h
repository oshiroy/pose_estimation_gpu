// -*- mode: c++ -*-

#pragma once

#include <thrust/device_vector.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#include <GL/gl.h>
#include <thread>

class CUDAManager{
private:

  void gpuInit(GLuint bufferID, int ih, int iw);
  cudaGraphicsResource *resource;

  float* score_arr; // device
  float* invisib_progress; // device
  float* depth_d; // device
  float* mask_d; // device

  int *height_arr, *invisib_cnt, *mask_cnts, *nonzero_cnts;
  float* sum_arr; // device
  float* iv_sum_arr; // device

  float* obj_score;
  float* ret_score_d;
  float* ret_score_h;

  int im_h;
  int im_w;
  int len_score;

  int* v_pthre;
  int* iv_pthre;

  float* host_arr; // host
  cudaStream_t s1, s2, s3;

  std::thread th;

public:
  void host_to_dev_depth(float* depth);
  void host_to_dev_mask(float* mask);
  void alloc_ret_score(int length);
  void transfer_score(float* ret_host);

  void update_gl_pixels(void);
  void unmap_resources(void);

  void evaluate_visibility(float* score, int pthre, float max_lim);
  void evaluate_visibility(int target_id, int pthre, float max_lim);

  CUDAManager();
  CUDAManager(GLuint bufferID, int ih, int iw);
  ~CUDAManager();
};
