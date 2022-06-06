# yolov4

The Pytorch implementation is from [ultralytics/yolov3 archive branch](https://github.com/ultralytics/yolov3/tree/archive). It can load yolov4.cfg and yolov4.weights(from AlexeyAB/darknet).

## Config

- Input shape `INPUT_H`, `INPUT_W` defined in yololayer.h
- Number of classes `CLASS_NUM` defined in yololayer.h
- FP16/FP32 can be selected by the macro `USE_FP16` in yolov4.cpp
- GPU id can be selected by the macro `DEVICE` in yolov4.cpp
- NMS thresh `NMS_THRESH` in yolov4.cpp
- bbox confidence threshold `BBOX_CONF_THRESH` in yolov4.cpp
- `BATCH_SIZE` in yolov4.cpp

## How to generate yolo4.wts trained model from pytorch implementation with yolov4.cfg and yolov4.weights

```
git clone https://github.com/wang-xinyu/tensorrtx.git
git clone -b archive https://github.com/ultralytics/yolov3.git
// download yolov4.weights from https://github.com/AlexeyAB/darknet#pre-trained-models
cp {tensorrtx}/yolov4/gen_wts.py {ultralytics/yolov3/}
cd {ultralytics/yolov3/}
python gen_wts.py yolov4.weights
// a file 'yolov4.wts' will be generated.
// the master branch of yolov3 should work, if not, you can checkout be87b41aa2fe59be8e62f4b488052b24ad0bd450
Copy yolov4.wts into the main folder
```
## Dependencies installation && building environment:
```
Ubuntu 18.04,  CuDNN 7.6.5.32-1 + cuda10.2 + TensorTx 7.0.0-1
Gstreamer + OpenCV 4.4
+ sudo apt-get install gstreamer1.0*
+ sudo apt install ubuntu-restricted-extras
+ sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
+ Download OpenCV 4.4 source code under https://opencv.org/opencv-4-4-0/
+ Build OpenCV 4.4 with Gstreamer 
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D PYTHON_EXECUTABLE=$(which python3) \
-D BUILD_opencv_python2=OFF \
-D CMAKE_INSTALL_PREFIX=$(python3 -c “import sys; print(sys.prefix)”) \
-D PYTHON3_EXECUTABLE=$(which python3) \
-D PYTHON3_INCLUDE_DIR=$(python3 -c “from distutils.sysconfig import get_python_inc; print(get_python_inc())”) \
-D PYTHON3_PACKAGES_PATH=$(python3 -c “from distutils.sysconfig import get_python_lib; print(get_python_lib())”) \
-D WITH_GSTREAMER=ON \
-D BUILD_EXAMPLES=ON ..

See https://galaktyk.medium.com/how-to-build-opencv-with-gstreamer-b11668fa09c for further instructions
```


## Build app 
```
sudo mkdir build 
cd build
cmake ..
make 
```
## Run app
```
Choose video to be run for testing
Go to the application folder
Run 
sudo ./yolo4 -s (convert wts model to model stream)
sudo ./yolo4 -d <video_path>
```
## More Information
```
The code used:
Pedestrian Detection: https://github.com/wang-xinyu/tensorrtx
Centroid tracking:  https://github.com/prat96/Centroid-Object-Tracking
```