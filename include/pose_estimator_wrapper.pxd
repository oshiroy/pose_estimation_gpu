from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "pose_estimator.h":
    cdef cppclass PoseEstimator:
        PoseEstimator(vector[string], int, int, string) except +
        void set_depth_image(double*)
        void set_mask(double*)
        void set_object_id(int)
        void set_ransac_count(int)
        void set_k(double*)
        void evaluate_score(double*, double*, double, int, double*)
        void ransac_estimation(double*, double*, int, double, int, double*, double*)
