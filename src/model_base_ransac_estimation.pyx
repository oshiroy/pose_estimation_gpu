# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: profile=False

import os
import pyflann
import pyassimp

import numpy as np
cimport numpy as np
cimport cython


from libcpp.vector cimport vector
from libc.math cimport round

ctypedef np.float64_t DOUBLE_t
ctypedef np.int32_t INT_t
ctypedef np.float32_t FLOAT_t

cimport pose_estimation_cpu
cimport pose_estimator_wrapper as pew


cdef class CyPoseEstimator:
    cdef pew.PoseEstimator* _pose_est
    cdef int object_id
    models_pointcloud = []
    flann_idx_array = []
    def __init__(self, path_list, im_h, im_w, model_partial=4):
        pyflann.set_distance_type('euclidean')
        for path in path_list:
            pc = None
            model = pyassimp.core.load(path)
            for mesh in model.meshes:
                vert = mesh.vertices
                if pc is None:
                    pc = vert[0::model_partial, :]
                else:
                    pc = np.vstack((pc, vert[0::model_partial, :]))
            search_idx = pyflann.FLANN()
            search_idx.build_index(pc.astype(np.float32), algorithm='kmeans',
                                   centers_init='kmeanspp', random_seed=1234)
            self.flann_idx_array.append(search_idx)
            self.models_pointcloud.append(pc.transpose(1, 0))


    def __cinit__(self, path_list, im_h, im_w, model_partial=4):
        root_path = os.path.split(__file__)[0]
        self._pose_est = new pew.PoseEstimator(path_list, im_h, im_w, root_path)

    def __dealloc(self):
        del self._pose_est

    def set_depth(self, np.ndarray[DOUBLE_t, ndim=2] depth, use_refinement=True):
        cdef np.ndarray[DOUBLE_t, ndim=1] depth_in = np.asanyarray(depth.ravel())
        self._pose_est.set_depth_image(<double*> depth_in.data)

    def set_mask(self, np.ndarray[DOUBLE_t, ndim=2] mask):
        cdef np.ndarray[DOUBLE_t, ndim=1] mask_in = np.asanyarray(mask.ravel())
        self._pose_est.set_mask(<double*> mask_in.data)

    def set_object_id(self, int id):
        self._pose_est.set_object_id(id)
        self.object_id = id

    def set_ransac_count(self, int c):
        self._pose_est.set_ransac_count(c)

    def set_k(self, np.ndarray[DOUBLE_t, ndim=2] k):
        cdef np.ndarray[DOUBLE_t, ndim=1] k_in = np.asanyarray(k.ravel())
        self._pose_est.set_k(<double*> k_in.data)

    def icp_refinement(self, np.ndarray[DOUBLE_t, ndim=2] src, np.ndarray[DOUBLE_t, ndim=2] dst,
                       dst_search_idx=None, n_iter=50, thre_percent=95):

        ## pyflann search index build
        pyflann.set_distance_type('euclidean')
        if dst_search_idx is None:
            search_idx = pyflann.FLANN()
            search_idx.build_index(dst.transpose(1, 0).astype(np.float32), algorithm='kmeans',
                                   centers_init='kmeanspp', random_seed=1234)
        else:
            search_idx = dst_search_idx
        ## intialize for iteration
        cdef np.ndarray[DOUBLE_t, ndim=1] _t = np.zeros(3)
        cdef np.ndarray[DOUBLE_t, ndim=2] _R = np.diag((1, 1, 1)).astype(np.float64)
        cdef np.ndarray[DOUBLE_t, ndim=2] old_src = src
        cdef np.ndarray[DOUBLE_t, ndim=2] new_src
        cdef np.ndarray[DOUBLE_t, ndim=1] src_mean = np.mean(src, axis=1)
        cdef np.ndarray[DOUBLE_t, ndim=2] src_demean = src - src_mean[:, np.newaxis]
        cdef np.ndarray[INT_t, ndim=1] indices
        cdef np.ndarray[FLOAT_t, ndim=1] distances
        cdef np.ndarray[DOUBLE_t, ndim=1] dst_mean
        cdef np.ndarray[DOUBLE_t, ndim=2] dst_demean
        cdef int i
        for i in xrange(n_iter):
            indices, distances = search_idx.nn_index(old_src.transpose(1, 0).astype(np.float32),  1)
            # percentile outlier removal
            percentile_thre = np.percentile(distances, thre_percent)
            inlier_mask = (distances <= percentile_thre)

            dst_mean = np.mean(dst[:, indices[inlier_mask]], axis=1)
            dst_demean = dst[:, indices[inlier_mask]] - dst_mean[:, np.newaxis]

            _R = calc_rot_by_svd_cy(dst_demean, src_demean[:, inlier_mask])
            _t = dst_mean - np.dot(_R, src_mean)
            new_src = np.dot(_R, src) + _t[:, np.newaxis]

            if np.mean(np.abs(new_src - old_src)) < 1e-12:
                break
            old_src = new_src
        return _t, _R

    def evaluate_score(self,
                       np.ndarray[DOUBLE_t, ndim=1] t,
                       np.ndarray[DOUBLE_t, ndim=2] R,
                       double max_thre=0.1,
                       int percentile_thre = 90):
        ## intialize
        cdef np.ndarray[DOUBLE_t, ndim=1] ret = np.empty(1)
        cdef np.ndarray[DOUBLE_t, ndim=1] t_tmp = np.asanyarray(t.ravel())
        cdef np.ndarray[DOUBLE_t, ndim=1] R_tmp = np.asanyarray(R.ravel())
        self._pose_est.evaluate_score(<double*> t.data, <double*> R.data,
                                      max_thre, percentile_thre, <double*> ret.data)
        return ret.data

    def ransac_estimation(self,
                          np.ndarray[DOUBLE_t, ndim=2] y_arr,
                          np.ndarray[DOUBLE_t, ndim=2] x_arr,
                          int n_ransac=100, double max_thre=0.1,
                          int percentile_thre = 90):
        ## intialize
        cdef np.ndarray[DOUBLE_t, ndim=1] ret_t = np.zeros(3)
        cdef np.ndarray[DOUBLE_t, ndim=2] ret_R = np.diag((1.0, 1.0, 1.0))
        cdef np.ndarray[DOUBLE_t, ndim=1] x_arr_tmp = np.asanyarray(x_arr.ravel())
        cdef np.ndarray[DOUBLE_t, ndim=1] y_arr_tmp = np.asanyarray(y_arr.ravel())
        self._pose_est.ransac_estimation(<double*> x_arr_tmp.data, <double*> y_arr_tmp.data,
                                         len(x_arr[0]), max_thre, percentile_thre,
                                         <double*>ret_t.data, <double*>ret_R.data)
        return ret_t, ret_R

    def ransac_estimation_with_refinement(self,
                                          np.ndarray[DOUBLE_t, ndim=2] y_arr,
                                          np.ndarray[DOUBLE_t, ndim=2] x_arr,
                                          int n_ransac=100, double max_thre=0.1,
                                          int percentile_thre = 90):
        cdef np.ndarray[np.int32_t, ndim=2] rand_sample = np.array(
            np.random.randint(0, y_arr.shape[1], (n_ransac, 3)), dtype=np.int32)
        cdef np.ndarray[DOUBLE_t, ndim=3] rand_x = x_arr[:,rand_sample]
        cdef np.ndarray[DOUBLE_t, ndim=2] rand_x_mean = np.mean(rand_x, axis=2)
        cdef np.ndarray[DOUBLE_t, ndim=3] rand_y = y_arr[:, rand_sample]
        cdef np.ndarray[DOUBLE_t, ndim=2] rand_y_mean = np.mean(rand_y, axis=2)
        cdef np.ndarray[DOUBLE_t, ndim=2] rand_x_demean, rand_y_demean
        ## intialize
        cdef np.ndarray[DOUBLE_t, ndim=1] _t = np.zeros(3)
        cdef np.ndarray[DOUBLE_t, ndim=2] _R = np.diag((1.0, 1.0, 1.0))
        cdef np.ndarray[DOUBLE_t, ndim=1] ret_t = np.zeros(3)
        cdef np.ndarray[DOUBLE_t, ndim=2] ret_R = np.diag((1.0, 1.0, 1.0))
        cdef np.ndarray[DOUBLE_t, ndim=1] t_tmp
        cdef np.ndarray[DOUBLE_t, ndim=1] R_tmp
        cdef np.ndarray[DOUBLE_t, ndim=1] score = np.empty(1)

        cdef double best_score = 1e15
        cdef size_t i_ransac = 0
        for i_ransac in xrange(n_ransac):
            rand_x_demean = rand_x[:, i_ransac, :] - rand_x_mean[:, i_ransac, np.newaxis]
            rand_y_demean = rand_y[:, i_ransac, :] - rand_y_mean[:, i_ransac, np.newaxis]
            _R = calc_rot_c(rand_y_demean, rand_x_demean)
            _t = rand_y_mean[:, i_ransac] - np.dot(_R, rand_x_mean[:, i_ransac])
            t_tmp = np.asanyarray(_t.ravel())
            R_tmp = np.asanyarray(_R.ravel())
            self._pose_est.evaluate_score(<double*> t_tmp.data, <double*> R_tmp.data,
                                          max_thre, percentile_thre, <double*> score.data)
            if score[0] < best_score:
                best_score = score[0]
                ret_t = _t
                ret_R = _R

        # icp_t, icp_R = self.icp_refinement(np.dot(_R.T, y_arr - _t[:, np.newaxis]).astype(np.float64),
        #                                    self.models_pointcloud[self.object_id].astype(np.float64),
        #                                    dst_search_idx=self.flann_idx_array[self.object_id], n_iter=50)
        # ret_R = np.dot(ret_R, icp_R.T)
        # ret_t -= np.dot(ret_R, icp_t)
        return ret_t, ret_R

## caution !! this function is not correct!!
cdef calc_rot_c(np.ndarray[DOUBLE_t, ndim=2] Y,
                np.ndarray[DOUBLE_t, ndim=2] X):
    cdef np.ndarray[DOUBLE_t, ndim=2] R = np.empty((3,3),dtype=np.float64)
    pose_estimation_cpu.calc_rot_eigen_svd3x3(<double *> Y.data, <double *>  X.data, <double *> R.data)
    return R


cdef inline pointcloud_to_depth_c(np.ndarray[DOUBLE_t, ndim=2] pc,
                                  np.ndarray[DOUBLE_t, ndim=2] K, int im_h, int im_w):
    cdef np.ndarray[DOUBLE_t, ndim=1] depth = np.zeros(im_h * im_w)
    pose_estimation_cpu.pointcloud_to_depth_impl(<double *> pc.data, <double *> K.data, <double *> depth.data,
                             im_h, im_w, len(pc[0]))

    return np.asanyarray(depth).reshape(im_h, im_w)


cdef calc_rot_by_svd_cy(np.ndarray[DOUBLE_t, ndim=2] Y,
                        np.ndarray[DOUBLE_t, ndim=2] X):
    cdef np.ndarray[DOUBLE_t, ndim=2] R, U, V, H
    cdef np.ndarray[DOUBLE_t, ndim=1] S
    cdef double VU_det
    U, S, V = np.linalg.svd(np.dot(Y, X.T))
    VU_det = np.linalg.det(np.dot(V, U))
    H = np.array([[1.0,  0,      0],
                  [0,  1.0,      0],
                  [0,    0, VU_det]])
    R = np.dot(np.dot(U, H), V)
    return R


def model_base_ransac_estimation_cpp(np.ndarray[DOUBLE_t, ndim=2] y_arr,
                                     np.ndarray[DOUBLE_t, ndim=2] x_arr,
                                     np.ndarray[DOUBLE_t, ndim=2] model,
                                     np.ndarray[DOUBLE_t, ndim=2] depth,
                                     np.ndarray[DOUBLE_t, ndim=2] K,
                                     np.ndarray[DOUBLE_t, ndim=2] obj_mask,
                                     im_size,
                                     int n_ransac=100, double max_thre=0.1,
                                     int percentile_thre = 90):
    ## intialize
    cdef np.ndarray[DOUBLE_t, ndim=1] ret_t = np.zeros(3)
    cdef np.ndarray[DOUBLE_t, ndim=2] ret_R = np.diag((1.0, 1.0, 1.0))
    cdef int imsize_h = im_size[0]
    cdef int imsize_w = im_size[1]
    cdef np.ndarray[DOUBLE_t, ndim=1] x_arr_tmp = np.asanyarray(x_arr.ravel())
    cdef np.ndarray[DOUBLE_t, ndim=1] y_arr_tmp = np.asanyarray(y_arr.ravel())
    cdef np.ndarray[DOUBLE_t, ndim=1] model_tmp = np.asanyarray(model.ravel())
    cdef np.ndarray[DOUBLE_t, ndim=1] depth_tmp = np.asanyarray(depth.ravel())
    cdef np.ndarray[DOUBLE_t, ndim=1] K_tmp = np.asanyarray(K.ravel())
    cdef np.ndarray[DOUBLE_t, ndim=1] obj_mask_tmp = np.asanyarray(obj_mask.ravel())

    # print depth_tmp[0, 0], depth_tmp[0, 1], depth_tmp[1, 0]
    pose_estimation_cpu.ransac_estimation_loop(<double*> x_arr_tmp.data, <double*> y_arr_tmp.data,
                                <double*> depth_tmp.data, <double*> model_tmp.data,
                                <double*> K_tmp.data, <double*> obj_mask_tmp.data,
                                len(x_arr[0]), len(model[0]), imsize_h, imsize_w, n_ransac,
                                max_thre, 90, <double*>ret_t.data, <double*>ret_R.data)
    return ret_t, ret_R



## cython and c++ loop impl
def model_base_ransac_estimation_cy(np.ndarray[DOUBLE_t, ndim=2] y_arr,
                                    np.ndarray[DOUBLE_t, ndim=2] x_arr,
                                    np.ndarray[DOUBLE_t, ndim=2] model,
                                    np.ndarray[DOUBLE_t, ndim=2] depth,
                                    np.ndarray[DOUBLE_t, ndim=2] K,
                                    np.ndarray[DOUBLE_t, ndim=2] obj_mask,
                                    im_size,
                                    int n_ransac=100, double max_thre=0.1,
                                    int percentile_thre = 90):

    cdef np.ndarray[np.int32_t, ndim=2] rand_sample = np.array(
        np.random.randint(0, y_arr.shape[1], (n_ransac, 3)), dtype=np.int32)
    cdef np.ndarray[DOUBLE_t, ndim=3] rand_x = x_arr[:,rand_sample]
    cdef np.ndarray[DOUBLE_t, ndim=2] rand_x_mean = np.mean(rand_x, axis=2)
    cdef np.ndarray[DOUBLE_t, ndim=3] rand_y = y_arr[:, rand_sample]
    cdef np.ndarray[DOUBLE_t, ndim=2] rand_y_mean = np.mean(rand_y, axis=2)

    ## intialize
    cdef np.ndarray[DOUBLE_t, ndim=1] _t = np.zeros(3)
    cdef np.ndarray[DOUBLE_t, ndim=2] _R = np.diag((1.0, 1.0, 1.0))
    cdef np.ndarray[DOUBLE_t, ndim=1] ret_t = np.zeros(3)
    cdef np.ndarray[DOUBLE_t, ndim=2] ret_R = np.diag((1.0, 1.0, 1.0))
    cdef np.ndarray[DOUBLE_t, ndim=1] ret_t_tri = np.zeros(3)
    cdef np.ndarray[DOUBLE_t, ndim=2] ret_R_tri = np.diag((1.0, 1.0, 1.0))

    cdef double best_score = 1e15
    cdef double best_score_tri = 1e15
    cdef int obj_visib_thre = np.sum(obj_mask) * 0.5

    cdef int imsize_h = im_size[0]
    cdef int imsize_w = im_size[1]

    cdef np.ndarray[DOUBLE_t, ndim=2] depth_mask = (depth != 0).astype(np.float64)
    cdef np.ndarray[DOUBLE_t, ndim=2] depth_obj_mask = depth_mask * obj_mask
    cdef np.ndarray[DOUBLE_t, ndim=2] depth_nonobj_mask = depth_mask * (1 - obj_mask)

    cdef np.ndarray[DOUBLE_t, ndim=2] depth_model, depth_diff, invisib_mask
    cdef np.ndarray[DOUBLE_t, ndim=1] dist, score_visib_arr

    cdef double score, score_tri, score_visib, score_invisib

    cdef np.ndarray[DOUBLE_t, ndim=2] rand_x_demean,  rand_y_demean

    cdef size_t i_ransac = 0

    for i_ransac in xrange(n_ransac):
        rand_x_demean = rand_x[:, i_ransac, :] - rand_x_mean[:, i_ransac, np.newaxis]
        rand_y_demean = rand_y[:, i_ransac, :] - rand_y_mean[:, i_ransac, np.newaxis]
        # _R = calc_rot_by_svd_cy(rand_y_demean, rand_x_demean)
        _R = calc_rot_c(rand_y_demean, rand_x_demean)
        _t = rand_y_mean[:, i_ransac] - np.dot(_R, rand_x_mean[:, i_ransac])

        dist = np.sum(np.abs(np.dot(_R, x_arr) + _t[:, np.newaxis] - y_arr), axis=0)
        score = pose_estimation_cpu.mean1d_up_limit(<double*> dist.data, <int> len(dist), max_thre)

        if score < best_score:
            best_score = score
            ret_t = _t
            ret_R = _R

        depth_model = pointcloud_to_depth_c(np.dot(_R, model) + _t[:, np.newaxis],
                                            K, imsize_h, imsize_w)
        depth_diff = depth_model - depth

        score_visib =  pose_estimation_cpu.calc_visib_socre_from_map(<double *> depth_diff.data,
                                                 <double *> depth_obj_mask.data,
                                                 imsize_h, imsize_w,
                                                 obj_visib_thre, percentile_thre, max_thre)

        invisib_mask = (depth_model != 0) * depth_nonobj_mask
        score_invisib =  pose_estimation_cpu.calc_invisib_socre_from_map(<double *> depth_diff.data,
                                                     <double *> invisib_mask.data,
                                                     imsize_h, imsize_w,
                                                     0.015, percentile_thre, max_thre)

        score_tri = score + score_visib + score_invisib

        if score_tri < best_score_tri:
            best_score_tri = score_tri
            ret_t_tri = _t
            ret_R_tri = _R
    # print score_tri

    return ret_t_tri, ret_R_tri
