# YOLACT-TensorRT in C++

our repo base on pytorch version of yolact [https://github.com/dbolya/yolact.git](https://github.com/dbolya/yolact.git)

## Step 0: Convert baidu paddlepaddle model to onnx model and generate dynamic TensoRT engine

```shell
paddle2onnx --model_dir ./ch_PP-OCRv3_rec_infer/ --model_filename ./ch_PP-OCRv3_rec_infer/inference.pdmodel --params_filename ./ch_PP-OCRv3_rec_infer/inference.pdiparams --save_file ch-pp-ocrv3-rec.onnx --opset_version 13

./trtexec --onnx=./ch_PP-OCRv3_rec_infer/ch-pp-ocrv3-rec.onnx --explicitBatch --minShapes=x:1x3x48x32 --optShapes=x:1x3x48x64 --maxShapes=x:32x3x48x960 --fp16 --saveEngine=ch-pp-ocrv3-rec-fp16.engine --workspace=10240
```


## Step 1: Build project to ocr_det and ocr_rec

```shell
mkdir -p build && cd build
cmake ..
make
```


## Step 2: Test inference yolact engine

```shell
./ocr_det ../ch-pp-ocrv3-det-fp16.engine -i ../0.jpg
./ocr_rec ../ch-pp-ocrv3-rec-fp16.engine -i ../1.jpg
```

ocr_det检测文字的框，斜的框，结果是多个框的四个点的坐标

ocr_rec根据字典和ocr_det的框裁剪出正的bbox框图像，得到最终的文字内容和文字置信度

