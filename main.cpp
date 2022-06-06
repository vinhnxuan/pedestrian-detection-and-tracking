
#include <fstream>
#include <iostream>

#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "subprocess.hpp"
#include "NvInfer.h"
#include "utils.h"
#include "cuda_runtime_api.h"
#include "logging.h"

#include <stdio.h>

#include "mish.h"
#include "Tracker.h"
#include "YoloModel.h"

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1

#define OUT_WIDTH 768
#define OUT_HEIGHT 576

using namespace nvinfer1;
static Logger gLogger;



// create a model using the API directly and serialize it to a stream
int convert_wtsModel2ModelStream(){
    IHostMemory* modelStream{nullptr};
    APIToModel(BATCH_SIZE, &modelStream);
    assert(modelStream != nullptr);
    std::ofstream p("yolov4.engine", std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    return 0;
}

char* readModelStream(size_t &size){
    std::ifstream file("yolov4.engine", std::ios::binary);
    char *trtModelStream{nullptr};
    
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    return trtModelStream;
}

bool is_file_exist(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}


void doDetectionAndTrackingOnFrame(
    const cv::Mat &frame, 
    IExecutionContext& context, 
    const std::vector<std::vector<cv::Point>> &restricted_area_contours,
    CentroidTracker *centroidTracker
    ){
    static float data[3 * INPUT_H * INPUT_W];
    static float prob[OUTPUT_SIZE];

    //std::cout << INPUT_H << INPUT_W << OUTPUT_SIZE << std::endl;
    
    // do detection here for each frame
    cv::Mat pr_img = preprocess_img(frame);
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        data[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
        data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
        data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
    }
    std::vector<std::vector<int>> pedestrian_boxes;
    
    // Run inference
    auto start = std::chrono::system_clock::now();
    doInference(context, data, prob, 1);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    std::vector<std::vector<Yolo::Detection>> batch_res(1);
    auto& res = batch_res[0];
    nms(res, &prob[0]);

    for (size_t j = 0; j < res.size(); j++) {
        cv::Rect r = get_rect(frame, res[j].bbox);
        if ((int)res[j].class_id == 0)
        {
            
            //cv::putText(frame, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            pedestrian_boxes.insert(pedestrian_boxes.end(), {r.x, r.y, r.width + r.x, r.y + r.height});
            if (inside_restricted_area(restricted_area_contours, r )){
                cv::rectangle(frame, r, cv::Scalar(0, 0, 255), 2);
                cv::putText(frame, "Not permitted", cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0, 0, 255), 2);
            }
            else{
                cv::rectangle(frame, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            }
        }
    }

    auto objects = centroidTracker->update(pedestrian_boxes);

    if (!objects.empty()) {
        for (auto obj: objects) {
            std::string ID = std::to_string(obj.first);
            cv::putText(frame, ID, cv::Point(obj.second.first, obj.second.second),
                        cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0x27, 0xC1, 0x36), 2);
        }
    }
}

int doDotectionAndTracking(const char* video_path, const size_t size, char *trtModelStream)
{
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    cv::VideoCapture cap(video_path);
    // Check if camera opened successfully
    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    CentroidTracker* centroidTracker = new CentroidTracker(20);
    bool is_first_frame = true; 
    cv::Mat restricted_area_img;
    std::vector<std::vector<cv::Point> > restricted_area_contours;
    //unsigned char buffer[OUT_HEIGHT][OUT_WIDTH][3] = {0};

    cv::VideoWriter writer;
    writer.open("appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=1000 key-int-max=30 ! video/x-h264, profile=constrained-baseline ! rtph264pay pt=96 mtu=1200 ! udpsink host=127.0.0.1 port=10000"
        , 0, (double)30, cv::Size(768,576), true);
    //writer.open("appsrc ! videoconvert ! x264enc noise-reduction=10000 tune=zerolatency byte-stream=true threads=4 bitrate=1000 key-int-max=30 ! video/x-h264, profile=constrained-baseline ! rtph264pay pt=96 mtu=1200 ! udpsink host=127.0.0.1 port=10000"
    //    , 0, (double)30, cv::Size(768,576), true);
    /*writer.open("appsrc ! videoconvert ! x264enc noise-reduction=10000 tune=zerolatency byte-stream=true threads=4 ! mpegtsmux ! udpsink host=127.0.0.1 port=6000"
                , 0, (double)30, cv::Size(768,576), true);*/
    if (!writer.isOpened()) {
        printf("=ERR= can't create video writer\n");
        return -1;
    }


    while(true){
        cv::Mat frame;
        // Capture frame-by-frame
        cap >> frame;
        // If the frame is empty, break immediately
        if (frame.empty()) 
        {
            printf("!!! cvQueryFrame failed: no frame\n");
            cap = cv::VideoCapture(video_path);
            continue;
        } 
        assert (frame.cols * frame.rows == OUT_HEIGHT* OUT_WIDTH);
        
        if (is_first_frame){
            restricted_area_img = create_restricted_area_on_src_img(frame);
            cv::findContours( restricted_area_img, restricted_area_contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
            cv::imshow( "Mask", restricted_area_img );
            is_first_frame = false;
        }

         

        drawContours( frame, restricted_area_contours, -1, cv::Scalar(0, 0, 255), 2 );
        doDetectionAndTrackingOnFrame (frame, *context, restricted_area_contours, centroidTracker);
        // Display the resulting frame
        cv::imshow( "Frame", frame );


        /*for (int j = 0; j < OUT_WIDTH; j++) {
            for (int i = 0; i < OUT_HEIGHT; i++) {
                    cv::Vec3b intensity = frame.at<cv::Vec3b>(i, j);
                    buffer[i][j][0] = intensity.val[0];
                    buffer[i][j][1] = intensity.val[1];
                    buffer[i][j][2] = intensity.val[2];
                }
        }*/
        //fwrite(buffer, 1, frame.rows*frame.cols*3, pipeout);
        writer << frame;
        // Press  ESC on keyboard to exit
        char c=(char)cv::waitKey(25);
        if(c==27)
        break;
    }

    // When everything done, release the video capture object
    cap.release();
    delete centroidTracker;
    // Closes all the frames
    cv::destroyAllWindows();

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}


int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // convert wts model to model stream
    if (argc == 2 && std::string(argv[1]) == "-s") {
        return convert_wtsModel2ModelStream();    
    // use model stream to run pedestrian detection and so on ..
    } else if (argc == 3 && std::string(argv[1]) == "-d") {
        char *trtModelStream{nullptr};
        size_t size{0};
        trtModelStream = readModelStream(size);
        if (trtModelStream == nullptr) {
            std::cout << "Reading model failed." << std::endl;
            return -1;
        }
        // prepare input data ---------------------------
        if (!is_file_exist(argv[2])) {
            std::cout << "File video is not found." << std::endl;
            return -1;
        }

        
        //FILE *pipeout = popen("ffmpeg -re -stream_loop -1 -f rawvideo -vcodec rawvideo -pix_fmt bgr24 -s 768x576 -i pipe:0 -f rtsp rtsp://127.0.0.1:8554/mystream", "w");
        //FILE *pipeout = popen("ffmpeg -re -stream_loop -1 -f rawvideo -vcodec rawvideo -pix_fmt bgr24 -r 30 -s 768x576 -i pipe:0 -c:v libx264 -tune zerolatency -preset fast -payload_type 96 -ssrc 42 -an -bufsize 300k -f rtp rtp://127.0.0.1:10000?pkt_size=1200", "w");
        
        
        //FILE *pipeout = popen("gst-launch-1.0 -v videotestsrc ! video/x-raw, width = 768, width =576 ! x264enc tune=zerolatency bitrate=1000 key-int-max=30 ! video/x-h264, profile=constrained-baseline ! rtph264pay pt=96 mtu=1200 ! udpsink host=127.0.0.1 port=6000", "w");
        
        //FILE *pipeout = popen("gst-launch-1.0 -v appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=1000 key-int-max=30 ! video/x-h264, profile=constrained-baseline ! rtph264pay pt=96 mtu=1200 ! udpsink host=127.0.0.1 port=10000", "w");
        
        return doDotectionAndTracking (argv[2], size, trtModelStream); 
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov4 -s  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov4 -d ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }
    
}
