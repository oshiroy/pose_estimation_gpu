// -*- mode: c++ -*-

#include <stdio.h>
#include <vector>

#include <iostream>
#include <string>
#include <algorithm>
#include <random>
#include <thread>

#include <GL/glew.h>

#if defined(__APPLE__) || defined(MACOSX)
#include <GLFW/glfw3.h>
# include <OpenGL/gl.h>

#else
# define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#endif


#include <Eigen/SVD>
#include <Eigen/LU>

#include "include/Mesh.h"
#include "include/shader.h"
#include "include/pose_estimator.h"


using namespace Eigen;



glm::mat4 projection_matrix_from_parmas(double* K, int x0, int y0, int w, int h,
                                        double nc, double fc, int window_coords)
{
  // if window_coords = 0, y_up
  // else y_down
  double depth = float(fc - nc);
  double q = -(fc + nc) / depth;
  double qn = -2 * (fc * nc) / depth;
  glm::mat4 ret_mat;
  if(window_coords == 0)
    {
      ret_mat = glm::mat4(
                          2 * K[0] / w, -2 * K[1] / w, (-2 * K[2] + w + 2 * x0) / w, 0,
                          0, -2 * K[4] / h, (-2 * K[5] + h + 2 * y0) / h, 0,
                          0, 0, q, qn,
                          0, 0, -1, 0);
    }
  else
    {
      ret_mat = glm::mat4(
                          2 * K[0] / w, -2 * K[1] / w, (-2 * K[2] + w + 2 * x0) / w, 0,
                          0, 2 * K[4] / h, (2 * K[5] - h + 2 * y0) / h, 0,
                          0, 0, q, qn,
                          0, 0, -1, 0);
    }
  return  glm::transpose(ret_mat);
}


bool PoseEstimator::initialize_gl(std::vector<std::string> path_list)
{
  printf("[pose_estimator] initialize OpenGL\n");
  // Initialise GLFW
  if( !glfwInit() )
    {
      fprintf( stderr, "Failed to initialize GLFW\n" );
      return -1;
    }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_VISIBLE, GL_FALSE); // if GL_FALSE, offscreen render

  // Create a windowed mode window and its OpenGL context
  window = glfwCreateWindow( _im_w, _im_h, "Offscreen Depth Rendering", NULL, NULL );
  if (!window) {
    glfwTerminate();
    printf("cannot create window");
    return -1;
  }
  // Make the window's context current
  glfwMakeContextCurrent(window);

  // if glew functions failed, please comment in below option,
  glewExperimental = GL_TRUE;

  // Initialize GLEW
  if (glewInit() != GLEW_OK) {
    fprintf(stderr, "Failed to initialize GLEW\n");
    return -1;
  }

  // Enable depth test
  glEnable(GL_DEPTH_TEST);

  // Accept fragment if it closer to the camera than the former one
  glDepthFunc(GL_LESS);

  printf("Root Path : %s\n", rootpath.c_str());

  std::string vert_path("/shader/simpleShader.vert");
  std::string frag_path("/shader/simpleShader.frag");
  // Create and compile our GLSL program from the shaders
  programID = LoadShaders((rootpath + vert_path).c_str(),
                          (rootpath + frag_path).c_str());

  // Get a handle for our "MVP" uniform
  MatrixMVP_ID = glGetUniformLocation(programID, "MVP");
  // Get a handle for our "MVP" uniform
  MatrixMV_ID = glGetUniformLocation(programID, "MV");

  for(size_t i = 0; i < path_list.size(); i++){
    mesh_vector.push_back(new Mesh(path_list[i].c_str()));
  }

  glGenFramebuffers(1, &framebuffer);
  glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

  glGenRenderbuffers(1, &colorrenderbuffer);
  glBindRenderbuffer(GL_RENDERBUFFER, colorrenderbuffer);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, _im_w, _im_h);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER,
                            colorrenderbuffer);
  glBindRenderbuffer(GL_RENDERBUFFER, 0);

  // Always check that our framebuffer is ok
  if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    return false;

  glEnable(GL_CULL_FACE); // Cull triangles which normal is not towards the camera
  glCullFace(GL_BACK); // Cull back-facing triangles -> draw only front-facing triangles

  glViewport(0, 0, _im_w, _im_h);
  glUseProgram(programID);

  printf("[pose_estimator] initialize OpenGL is finished\n");

  return true;
}

bool PoseEstimator::initialize_cuda(GLuint bufferID, int im_h, int im_w)
{
  printf("[pose_estimator] initialize CUDA\n");
  cuda_manager = new CUDAManager(bufferID, im_h, im_w);
  return true;
}

void PoseEstimator::render(VectorXd t, MatrixXd R)
{
  // Render to framebuffer
  glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
  // Clear the screen
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glm::mat4 ProjectionMat = projection_matrix_from_parmas(_k, 0, 0, _im_w, _im_h,
                                                          0.01, 10.0, 1);
  // note : view mat is identity matrix
  glm::mat4 MV = glm::mat4(
                           R(0,0), -R(1,0), -R(2,0),   0,
                           R(0,1), -R(1,1), -R(2,1),   0,
                           R(0,2), -R(1,2), -R(2,2),   0,
                           t(0),     -t(1),   -t(2),   1);
  glm::mat4 MVP = ProjectionMat * MV;

  // Send transformation to the currently bound shader in the "MVP" uniform
  glUniformMatrix4fv(MatrixMVP_ID, 1, GL_FALSE, &MVP[0][0]);
  // Send transformation to the currently bound shader in the "MV" uniform
  glUniformMatrix4fv(MatrixMV_ID, 1, GL_FALSE, &MV[0][0]);

  // render depth
  mesh_vector[obj_id]->render();


  /* for debug */
  // glBindFramebuffer(GL_FRAMEBUFFER, 0);
  // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  // mesh_vector[obj_id]->render();
  // glfwSwapBuffers(window);
  // glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
}

void PoseEstimator::set_depth_image(double* depth_img)
{
  depth = depth_img;
  float* tmp = new float[_im_h * _im_w];
  for(int i = 0; i < _im_h; i++){
    for(int j = 0; j < _im_w; j++){
      tmp[i * _im_w + j] = (float)depth[i * _im_w + j];
    }
  }
  cuda_manager->host_to_dev_depth(tmp);
  delete[] tmp;
}

void PoseEstimator::set_mask(double* mask_img)
{
  mask = mask_img;
  float* tmp = new float[_im_h * _im_w];
  for(int i = 0; i < _im_h; i++){
    for(int j = 0; j < _im_w; j++){
      tmp[i * _im_w + j] = (float) mask[i * _im_w + j];
    }
  }
  cuda_manager->host_to_dev_mask(tmp);
  delete[] tmp;
}

void PoseEstimator:: set_object_id(int id)
{
  obj_id = id;
}

void PoseEstimator:: set_ransac_count(int c)
{
  n_ransac = c;
  // R_arr.resize(c);
  // t_arr.resize(c);
  cuda_manager->alloc_ret_score(c);
}

void PoseEstimator::set_k(double* k)
{
  _k = k;
}


void PoseEstimator::evaluate_score(double* t, double* R, double max_dist_lim, int pthre, double *ret_score)
{
  float visib_score;
  MatrixXd eigen_R(3,3);
  VectorXd eigen_t(3);
  // double* to eigen
  for(int i = 0; i < 3; i++){
    eigen_t(i) = t[i];
    for(int j = 0;  j < 3; j++){
      eigen_R(i, j) = R[i * 3 +j];
    }
  }
  render(eigen_t, eigen_R);
  cuda_manager->evaluate_visibility(&visib_score, pthre, (float) max_dist_lim);
  *ret_score = visib_score;
}

void PoseEstimator::ransac_estimation(double* x_arr, double* y_arr, int len_arr,
                                      double max_dist_lim, int pthre,
                                      double* ret_t, double* ret_r)
{
  std::random_device rnd;
  std::mt19937 mt(rnd());
  std::uniform_int_distribution<> rand_sample(0, len_arr);

  int j, k, r_sample;

  MatrixXd x_arr_mat = Map<Matrix<double, Dynamic, Dynamic, RowMajor> >(x_arr, 3, len_arr);
  MatrixXd y_arr_mat = Map<Matrix<double, Dynamic, Dynamic, RowMajor> >(y_arr, 3, len_arr);

  MatrixXd x_mat(3,3);
  MatrixXd y_mat(3,3);
  VectorXd x_mean(3);
  VectorXd y_mean(3);
  MatrixXd x_demean, y_demean;

  MatrixXd R(3,3);
  MatrixXd best_R(3,3);
  VectorXd t(3);
  VectorXd best_t(3);

  // for svd
  MatrixXd v, u;

  // for scoring
  float visib_score;
  double score, best_score = 1e15;

  for(int i = 0; i < n_ransac; i++){
    // random sampling
    for (j = 0; j < 3; j++){
      r_sample = rand_sample(mt);
      x_mat.col(j) = x_arr_mat.col(r_sample);
      y_mat.col(j) = y_arr_mat.col(r_sample);
    }
    x_mean = x_mat.rowwise().mean();
    y_mean = y_mat.rowwise().mean();
    x_demean = x_mat.colwise() - x_mean;
    y_demean = y_mat.colwise() - y_mean;

    // compute SVD
    JacobiSVD<MatrixXd> svd(x_demean * y_demean.transpose(), ComputeFullU | ComputeFullV);
    v = svd.matrixV();
    u = svd.matrixU();
    // Compute R = V * U'
    if (u.determinant() * v.determinant() < 0){
      for (size_t x = 0; x < 3; ++x)
        v(x, 2) *= -1;
    }
    R = v * u.transpose();
    t = y_mean - R * x_mean;

    // rendering model depth image
    // render(t, R);
    // cuda_manager->evaluate_visibility(&visib_score, pthre, (float) max_dist_lim);
    // score = visib_score;
    /* no use render impl */
    score = (((R * x_arr_mat).colwise() + t) -
             y_arr_mat).cwiseAbs().rowwise().mean().sum();

    // printf("score : %f\n", score);

    if(score < best_score){
      best_score = score;
      best_t = t;
      best_R = R;
    }
  }

  // copy results device -> host
  // float *results_host;
  // results_host = (float*)malloc(sizeof(float) * n_ransac * 2);
  // cuda_manager->transfer_score(results_host);
  // // search best score
  // for(int i = 0; i < n_ransac; i++){
  //   score = results_host[2 * i] + results_host[2 * i + 1];
  //   printf("visib score : %f\n", score);
  //   if(score < best_score){
  //     score = best_score;
  //   }
  // }
  // free(results_host);

  // Eigen -> double*
  for(j = 0 ; j < 3 ; j++){
    ret_t[j] = best_t(j);
    for(k = 0 ; k < 3 ; k++){
      ret_r[j * 3 + k] = best_R(j, k);
    }
  }
}


PoseEstimator::PoseEstimator(std::vector<std::string> path_list, int im_h, int im_w,
                             std::string root_path)
{
  _im_h = im_h;
  _im_w = im_w;

  rootpath = root_path;

  if (!initialize_gl(path_list)){
    printf("Fail Initialize OpenGL\n");
  }
  if (!initialize_cuda(colorrenderbuffer, im_h, im_w)){
    printf("Fail Initialize CUDA\n");
  }

  depth = new double[_im_h * _im_w];
  mask = new double[_im_h * _im_w];

  // initialize params
  set_ransac_count(100);
  printf("[pose_estimator] success Intialization\n");
}


PoseEstimator::~PoseEstimator(void)
{
  // Cleanup shader
	glDeleteProgram(programID);
  // Close OpenGL window and terminate GLFW
  glfwTerminate();
  delete[] depth;
  delete[] mask;
}
