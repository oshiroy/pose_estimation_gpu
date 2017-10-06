// -*- mode: c++ -*-

#pragma once

// include glew before include gl and glfw !!
# include <GL/glew.h>

#if defined(__APPLE__) || defined(MACOSX)
# include <GLFW/glfw3.h>
#else
# define GL_GLEXT_PROTOTYPES
# include <GLFW/glfw3.h>
#endif

# include <vector>
# include <string>

#include <Eigen/Core>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

# include "Mesh.h"
# include "evaluate_gpu.h"


class PoseEstimator
{
private:
  bool initialize_gl(std::vector<std::string> path_list);
  bool initialize_cuda(GLuint bufferID, int im_h, int im_w);
  void render(Eigen::VectorXd t, Eigen::MatrixXd R);

  // input depth image
  double* depth;

  // input object mask
  double* mask;

  // ransac iter
  int n_ransac;

  // for OpenGL
  GLFWwindow* window;
  GLuint programID;
  GLuint MatrixMVP_ID;
  GLuint MatrixMV_ID;

  GLuint colorrenderbuffer;
  GLuint depthrenderbuffer;
  GLuint framebuffer;

  GLenum DrawBuffers[1];

  GLuint depthTexture;

  // open gl mesh
  std::vector<Mesh *> mesh_vector;

  // for CUDA
  CUDAManager* cuda_manager;

  //root path
  std::string rootpath;

public :

  // object id for using mesh vector
  int obj_id;

  // camera parameter
  double* _k;

  // image size
  int _im_h;
  int _im_w;

  // for scoring
  double score;
  double best_score;
  std::vector<double> trans;
  std::vector<double> best_trans;
  std::vector<std::vector <double> > rot;
  std::vector<std::vector <double> > best_rot;

  // Member Functions
  void set_depth_image(double* depth_img);
  void set_mask(double* mask_img);
  void set_object_id(int id);
  void set_ransac_count(int c);
  void set_k(double* k);
  void ransac_estimation(double* x_arr, double* y_arr, int len_arr,
                         double max_dist_lim, int pthre, double* ret_t, double* ret_r);

  PoseEstimator(std::vector<std::string> path_list, int im_h, int im_w,
                std::string root_path);
  ~PoseEstimator();
};
