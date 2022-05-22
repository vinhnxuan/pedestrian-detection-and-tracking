#ifndef __YOLOMODEL_H_
#define __YOLOMODEL_H_
#include <opencv2/opencv.hpp>
#include "yololayer.h"
#include "NvInfer.h"

#define NMS_THRESH 0.4
#define BBOX_CONF_THRESH 0.5

using namespace nvinfer1;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int DETECTION_SIZE = sizeof(Yolo::Detection) / sizeof(float);
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * DETECTION_SIZE + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1

cv::Mat preprocess_img(const cv::Mat &img) ;
cv::Rect get_rect(const cv::Mat &img, float bbox[4]);
float iou(float lbox[4], float rbox[4]);
bool cmp(const Yolo::Detection& a, const Yolo::Detection& b);
void nms(std::vector<Yolo::Detection>& res, float *output, float nms_thresh = NMS_THRESH);
std::map<std::string, Weights> loadWeights(const std::string file);
IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps);
ILayer* convBnMish(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int p, int linx);
ILayer* convBnLeaky(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int p, int linx);
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt);
void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream);
void doInference(IExecutionContext& context, float* input, float* output, int batchSize);

#endif