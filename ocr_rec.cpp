#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <opencv2/opencv.hpp>
#include <string>
#include "logging.h"
#include "postprocess_op.h"


#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id


using namespace nvinfer1;

static Logger gLogger;

void CrnnResizeImg(const cv::Mat &img, cv::Mat &resize_img, float wh_ratio, const std::vector<int> &rec_image_shape) {
    int imgH, imgW;
    imgH = rec_image_shape[1];
    imgW = rec_image_shape[2];

    imgW = int(imgH * wh_ratio);

    float ratio = float(img.cols) / float(img.rows);
    int resize_w = 0;

    if (ceilf(imgH * ratio) > imgW){
        resize_w = imgW;
    }
    else{
        resize_w = int(ceilf(imgH * ratio));
    }
    
    cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
                cv::INTER_LINEAR);
    cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0,
                        int(imgW - resize_img.cols), cv::BORDER_CONSTANT,
                        {127, 127, 127});
}


void CTCPostProcess(std::vector<float> predict_batch, std::vector<float> predict_shape, 
                std::string label_path,  std::vector<std::string> &rec_texts,
                std::vector<float> &rec_text_scores)
{
    // read dictionary for decode result text
    std::vector<std::string> label_list_ = Utility::ReadDict(label_path);
    label_list_.insert(label_list_.begin(), "#"); // blank char for ctc
    label_list_.push_back(" ");
    
    for (int m = 0; m < predict_shape[0]; m++) {
        std::string str_res;
        int argmax_idx;
        int last_index = 0;
        float score = 0.f;
        int count = 0;
        float max_value = 0.0f;

        for (int n = 0; n < predict_shape[1]; n++) {
            argmax_idx = int(Utility::argmax(
                &predict_batch[(m * predict_shape[1] + n) * predict_shape[2]],
                &predict_batch[(m * predict_shape[1] + n + 1) * predict_shape[2]]));
                    
            max_value = float(*std::max_element(
                &predict_batch[(m * predict_shape[1] + n) * predict_shape[2]],
                &predict_batch[(m * predict_shape[1] + n + 1) * predict_shape[2]]));

            if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
            score += max_value;
            count += 1;
            str_res += label_list_[argmax_idx];
            }
            last_index = argmax_idx;
        }
        
        score /= count;
        if (isnan(score)) {
            continue;
        }

        //   rec_texts[indices[beg_img_no + m]] = str_res;
        //   rec_text_scores[indices[beg_img_no + m]] = score;
        rec_texts.push_back(str_res);
        rec_text_scores.push_back(score);
    }
}

const float mean_vals[3] = {0.5f, 0.5f, 0.5f};
const float norm_vals[3] = {0.5f, 0.5f, 0.5f};
float* blobFromImage(cv::Mat& img)
{
    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;

    for (int c = 0; c < channels; c++) 
    {
        for (int  h = 0; h < img_h; h++) 
        {
            for (int w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (((float)img.at<cv::Vec3b>(h, w)[c]) / 255.0f - mean_vals[c]) / norm_vals[c];
            }
        }
    }
    return blob;
}

static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}


int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (argc == 4 && std::string(argv[2]) == "-i") {
        const std::string engine_file_path {argv[1]};
        std::ifstream file(engine_file_path, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "Then use the following command:" << std::endl;
        std::cerr << "./ocr_rec ../ch-pp-ocrv3-rec-fp16.engine -i ../dog.jpg  // run inference" << std::endl;
        return -1;
    }

    const std::string input_image_path {argv[3]};

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // std::vector<std::string> file_names;
    // if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
    //     std::cerr << "read_files_in_dir failed." << std::endl;
    //     return -1;
    // }

    // GPU pointer
    int nbBindings = 2;
    float* buffers[nbBindings];
    
    const int batch_size = 1;
    const int i_idx = engine->getBindingIndex("x"); //1x3x48x480
    const int loc_idx= engine->getBindingIndex("softmax_5.tmp_0"); // 1x100x6625
    std::chrono::steady_clock::time_point Tbegin, Iend, Tend;


    cv::Mat bgr = cv::imread(input_image_path, 1); // bgr

    // cv::Mat pr_img = static_resize(bgr, false);
    cv::Mat pr_img;
    std::vector<int> rec_image_shape_ = {3, 48, 480};
    CrnnResizeImg(bgr, pr_img, 10.0, rec_image_shape_);

    int net_in_w = pr_img.cols;
    int net_in_h = pr_img.rows;
    assert(net_in_h == rec_image_shape_[1]);

    // method 1
    cv::Mat pr_img_rgb, pr_img_chw;
    cv::cvtColor(pr_img, pr_img_rgb, cv::COLOR_BGR2RGB);
    float* input_raw_data = blobFromImage(pr_img_rgb);

    // method 2
    // cv::Mat chw_image = cv::dnn::blobFromImage
    //                     (
    //                         pr_img, 1.0, // scale factor of each channel value of image
    //                         cv::Size(pr_img.colsm pr_img.rows), // spatial size for output image
    //                         cv::Scalar(127.5f), // mean
    //                         true, // swapRB: BGR to RGB
    //                         false, // crop
    //                         CV_32F // Depth of output blob. Choose CV_32F or CV_8U.
    //                     );

    
    const int i_size = 1 * 3 * net_in_h * net_in_w;
    const int loc_size = 1 * 100 * 6625;
    std::cout << ">>>>>>> net_in_w:" << net_in_w << " | net_in_h:" << net_in_h << std::endl;

    // only for dynamic inference
    context->setBindingDimensions(i_idx, Dims4(batch_size, 3, net_in_h, net_in_w));
    // nvinfer1::Dims dim = context->getBindingDimensions(0);

    // Create CPU buffers
    static float* h_input = nullptr, *loca = nullptr;

    Tbegin = std::chrono::steady_clock::now();

    CHECK(cudaMallocHost((void**)&h_input, i_size*sizeof(float)));
    memcpy(h_input, input_raw_data, i_size*sizeof(float));
    CHECK(cudaMallocHost((void**)&loca, loc_size*sizeof(float)));

    CHECK(cudaMalloc((void**)&buffers[i_idx], i_size*sizeof(float)));
    CHECK(cudaMemcpyAsync(buffers[i_idx], h_input, i_size*sizeof(float), cudaMemcpyHostToDevice, stream));

    // cuda result
    CHECK(cudaMalloc((void**)&buffers[loc_idx], loc_size*sizeof(float)));

    context->enqueueV2((void**)buffers, stream, nullptr);

    CHECK(cudaMemcpyAsync(loca, buffers[loc_idx], loc_size*sizeof(float), cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);

    Iend = std::chrono::steady_clock::now();
    float infer_time = std::chrono::duration_cast <std::chrono::milliseconds> (Iend - Tbegin).count();
    std::cout << "only inference time : " << infer_time/1000.0 << " Sec" << std::endl;

    std::vector<float> predict_shape{1, 100, 6625};
    std::string label_path = "../ppocr_keys_v1.txt";
    std::vector<std::string> rec_texts{};
    std::vector<float> rec_text_scores{};
    std::vector<float> predict_batch(loca, loca+loc_size);
    
    CTCPostProcess(predict_batch, predict_shape, label_path, rec_texts, rec_text_scores);


    Tend = std::chrono::steady_clock::now();
    float f = std::chrono::duration_cast <std::chrono::milliseconds> (Tend - Tbegin).count();

    std::cout << "end to end total time : " << f/1000.0 << " Sec" << std::endl;


    for (unsigned int i = 0; i < rec_texts.size(); i++)
    {
        std::cout << "Result: " << rec_texts[i] << " score: " << rec_text_scores[i] << std::endl;
    }
    
    cudaStreamDestroy(stream);
    cudaFreeHost(context);
    cudaFreeHost(engine);
    cudaFreeHost(runtime);

    CHECK(cudaFree(buffers[i_idx]));
    CHECK(cudaFree(buffers[loc_idx]));
    return 0;
}
