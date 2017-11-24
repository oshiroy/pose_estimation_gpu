


cdef extern from "pose_estimation_cpu.cpp":
    cdef void calc_rot_eigen_svd3x3(double* y_arr, double* x_arr, double* out_arr)
    cdef double mean1d_up_limit(double* x, int len_x, double uplim)
    cdef double visibility_scoring(double* x, int len_x, int percentile_thre, double max_dist)


    cdef double calc_visib_socre_from_map(double* depth, double* mask, int im_h, int im_w,
                                             int visib_thre, double percentile_thre,
                                             double max_dist_lim)

    cdef double calc_invisib_socre_from_map(double* depth_diff, double* mask,
                                               int im_h, int im_w, double fore_thre,
                                               double percentile_thre, double max_dist_lim)

    cdef void pointcloud_to_depth_impl(double* pc, double* K, double* depth,
                                          int im_h, int im_w, int len_pc)

    cdef void ransac_estimation_loop(double* x_arr, double* y_arr,
                                     double* depth, double* model, double* K, double* obj_mask,
                                     int len_arr, int len_model, int im_h, int im_w, int n_ransac,
                                     double max_thre, int pthre, double* ret_t, double* ret_r)

    cdef void simple_ransac_estimation_loop(double* x_arr, double* y_arr,
                                            int len_arr, int n_ransac,
                                            double* ret_t, double* ret_r)
