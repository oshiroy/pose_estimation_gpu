#!/usr/bin/env python

from matplotlib import pylab as plt
import cv2
import numpy as np
import time

import pose_estimation


def pointcloud_to_depth(pc, K, img_size):
    xs = np.round(pc[:, 0] * K[0, 0] / pc[:, 2] + K[0, 2])
    ys = np.round(pc[:, 1] * K[1, 1] / pc[:, 2] + K[1, 2])

    inimage_mask = (xs > 0) * (xs < img_size[0]) * \
                   (ys > 0) * (ys < img_size[1])

    xs = xs[inimage_mask].astype(np.int32)
    ys = ys[inimage_mask].astype(np.int32)
    zs = pc[:, 2][inimage_mask]

    idx = np.argsort(zs)[::-1]

    # render depth
    img_depth = np.zeros(img_size[::-1])
    img_depth[ys[idx], xs[idx]] = zs[idx]

    return img_depth




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

# output fig
fig = plt.figure(figsize=(3, 2))

print "image_size (height, width) = ", im_size

## pose estimation only CPU
print "pose estimation CPU"
t1 = time.time()
ret_t1, ret_R1 = pose_estimation.model_base_ransac_estimation_cy(pc, y, model,
                                                                 depth, k, obj_mask,
                                                                 im_size)
t1 = time.time() - t1

## pose estimation GPU
print "pose estimation GPU"
pose_estimator.set_ransac_count(100);
t2 = time.time()
pose_estimator.set_depth(depth)
pose_estimator.set_k(k)
pose_estimator.set_mask(obj_mask)
pose_estimator.set_object_id(0)
ret_t2, ret_R2 = pose_estimator.ransac_estimation(pc, y)
t2 = time.time() - t2


# import pstats, cProfile
# cProfile.runctx("model_base_ransac_estimation.model_base_ransac_estimation_cpp(pc, y, model, depth, k, obj_mask, im_size, n_ransac=1000)", globals(), locals(), "Profile.prof")
# s = pstats.Stats("Profile.prof")
# s.strip_dirs().sort_stats("time").print_stats()


## clac true pose
t_ren = pointcloud_to_depth((np.dot(t_rot, model) + t_cp[:, np.newaxis]).transpose(1,0),
                            k, im_size[::-1])

p_ren = pointcloud_to_depth((np.dot(ret_R1, model) + ret_t1[:, np.newaxis]).transpose(1,0),
                            k, im_size[::-1])

c_ren = pointcloud_to_depth((np.dot(ret_R2, model) + ret_t2[:, np.newaxis]).transpose(1,0),
                            k, im_size[::-1])

print "obj_mask count : ", len(obj_mask[obj_mask != 0])
print "depth nonzero count : ", len(depth[depth != 0])
print "pose_estimation cython cpu : ", t1
print "pose_estimation gpu: ", t2
print t_rot
print ret_R2
print t_cp
print ret_t2

ax = fig.add_subplot(2, 3, 1)
plt.title("RGB", fontsize=12)
ax.imshow(rgb[:,:,::-1] / 255.0)
ax = fig.add_subplot(2, 3, 2)
plt.title("Depth", fontsize=12)
ax.imshow(depth)
ax = fig.add_subplot(2, 3, 3)
plt.title("Mask", fontsize=12)
ax.imshow(obj_mask)
ax = fig.add_subplot(2, 3, 4)
plt.title("Grand Truth", fontsize=12)
ax.imshow(rgb[:,:,::-1] /255.0 * 0.5 + t_ren[:,:,np.newaxis] * 0.4)
ax = fig.add_subplot(2, 3, 5)
plt.title("CPU version", fontsize=12)
ax.imshow(rgb[:,:,::-1] /255.0 * 0.5 + p_ren[:,:,np.newaxis] * 0.4)
ax = fig.add_subplot(2, 3, 6)
plt.title("GPU version", fontsize=12)
ax.imshow(rgb[:,:,::-1] /255.0 * 0.5 + c_ren[:,:,np.newaxis] * 0.4)

plt.show()
