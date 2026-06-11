# -*- coding:utf-8 -*-

kGpuId = 0
kNumClass = 1
kInputH = 640
kInputW = 640
kNmsThresh = 0.45
kConfThresh = 0.5
kMaxNumOutputBbox = 1000
kNumBoxElement = 7

onnx_file = "/workspace/counting/models/onnx_models/yoloe.onnx"
trt_file = "/workspace/counting/models/trt_models/yoloe.plan"

# for FP16 mode
use_fp16_mode = False
# for INT8 mode
use_int8_mode = False
n_calibration = 20
cache_file = "/workspace/counting/models/trt_models/yoloe_int8.cache"
calibration_data_dir = "/workspace/counting/models/calibration"

class_name_list = [
    "person",
]
