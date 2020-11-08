# MobileNeuralNetworks

Requirements:
All code is run on a Jetson Nano. Basic instructions for setting this up can be found here:
https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit

The following packages are required for this project to work
1. Jetpack 4.2+
2. Python 3.6+
3. CMake
4. Cython3
5. SciPy
6. NumPy
7. OpenCV 4.1.2+
8. Tensorflow 2.x+
9. TensorRT 6.x+
10. Keras
11. CUDA
12. FaceNet (https://github.com/davidsandberg/facenet)

The following sites have good instructions for installing all these on the Jetson Nano:
https://www.pyimagesearch.com/2019/05/06/getting-started-with-the-nvidia-jetson-nano/
https://medium.com/@ageitgey/build-a-face-recognition-system-for-60-with-the-new-nvidia-jetson-nano-2gb-and-python-46edbddd7264

<a name="NN"></a>
Building SSD and MTCNN
--------------

1. Build TensorRT engines from the pre-trained MTCNN model. 

   ```shell
   $ cd ${HOME}/project/MobileNeuralNetworks/mtcnn
   $ make
   $ ./create_engines
   ```
   
2. Build the Cython code

   ```shell
   $ sudo pip3 install Cython
   $ cd ${HOME}/project/tensorrt_demos
   $ make
   ```

3. Build TensorRT engines from the pre-trained SSD models.

   ```shell
   $ cd ${HOME}/project/MobileNeuralNetworks/ssd
   $ ./install.sh
   $ ./build_engines.sh
   ```

<a name="Executing"></a>
Runing the Code
--------

Simply execute the trt_autonomous.py file, with a selected video input. For this project --usb 0 --height 720 --width 1280 was used

References
--------
TensorFlow FaceNet model from: https://github.com/davidsandberg/facenet
MTCNN PreTrained model from: https://github.com/AlphaQi/MTCNN-light
SSD PreTrained model from: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
Basis of TensorRT Implementation: https://github.com/jkjung-avt/tensorrt_demos
GraphSurgeon SSD Patch: https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/graphsurgeon/graphsurgeon.html
MTCNN PReLU to ReLU conversion: https://github.com/PKUZHOU/MTCNN_FaceDetection_TensorRT

There were many different suggestions for optimisation for the Jetson Nano, from blog posts to forum posts, which were all used for these models