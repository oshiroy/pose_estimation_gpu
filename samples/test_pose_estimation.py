#!/usr/bin/env python
import cv2
import numpy as np
import quaternion

import pose_estimation

# inference outputs
y = np.load("sample_data/mbr_y.npy")
obj_mask = np.load("sample_data/mbr_obj_mask.npy")

# rgb-d
rgb = cv2.imread("sample_data/mbr_rgb.jpg")
rgb = rgb.astype(np.float32)
depth = np.load("sample_data/mbr_depth.npy")
pc = np.load("sample_data/mbr_pc.npy")

# object model
model = np.load("sample_data/mbr_model.npy")
# true data
t_cp = np.load("sample_data/mbr_cp.npy")
t_rot = np.load("sample_data/mbr_rot.npy")

# camera parameter
k = np.load("sample_data/mbr_k.npy")

im_size = rgb.shape[:2]

# pose_estimator gpu instance
pose_estimator = pose_estimation.CyPoseEstimator(["sample_data/Driller.ply"],
                                                 im_size[0], im_size[1])
pose_estimator.set_ransac_count(100)
pose_estimator.set_depth(depth)
pose_estimator.set_k(k)
pose_estimator.set_mask(obj_mask)
pose_estimator.set_object_id(0)

for i in range(100):
    ret_t, ret_R = pose_estimator.ransac_estimation_with_refinement(pc, y)
    quat = quaternion.from_rotation_matrix(np.dot(ret_R.T, t_rot))
    quat_w = min(1, abs(quat.w))
    diff_t = np.linalg.norm(ret_t - t_cp)
    diff_angle = np.rad2deg(np.arccos(quat_w)) * 2
    print "epoch : ", i + 1, ", diff_t : ", diff_t, ", diff_angle : ", diff_angle
    if diff_t > 0.05 or diff_angle > 5.0:
        print "epoch : ", i + 1, "difference error!!"
