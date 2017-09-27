# pose_estimation_gpu


## What is this?
To Be Edited...  


regacy version(https://github.com/oshiroy/test/tree/master/pose_estimation_gpu)

## Input Data
RGB image  
Depth image  
Mask image  
Object coordinates candidates  
Object model (.ply)  

## OutPut Data
Position  
Rotation matrix



## Requirements
Eigen, cython, cuda, opengl, glfw, glew, etc...

## Sample
```
pip install -e .
python sample_pose_estimaiton.py
```

<img src="sample_data/sample_output.png" alt="output" title="output">
