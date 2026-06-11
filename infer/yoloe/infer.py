# -*- coding:utf-8 -*-

"""
    YOLOE 模型 TensorRT 推理模块

    输入：
        - images: float32 [N, 3, 640, 640]

    输出：
        - output0: float32 [N, 300, 38]  (box4 + conf1 + class1 + mask_coeff32)
"""

import os

import cv2
import numpy as np
import tensorrt as trt

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    try:
        from cuda import cudart
    except ImportError:
        raise ImportError(
            "无法导入 cudart。请安装 cuda-python: pip install cuda-python"
        )

from infer.yoloe.config import *
from infer.yoloe.preprocess import preprocess
from infer.yoloe.postprocess import postprocess
import infer.yoloe.calibrator as calibrator


class YoloeDetector:
    def __init__(
        self,
        trt_plan=trt_file,
        gpu_id=kGpuId,
        num_classes=kNumClass,
        nms_thresh=kNmsThresh,
        conf_thresh=kConfThresh
    ):
        self.trt_file = trt_plan
        self.logger = trt.Logger(trt.Logger.ERROR)
        cudart.cudaSetDevice(gpu_id)

        self.nums_classes = num_classes
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.class_name_list = class_name_list

        self.engine = self.get_engine()
        self.context = self.engine.create_execution_context()

        n_io = self.engine.num_io_tensors
        self.io_names = [self.engine.get_tensor_name(i) for i in range(n_io)]

        # 区分输入和输出 tensor 名称
        self.input_names = [
            n for n in self.io_names
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            n for n in self.io_names
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
        ]

        # 设置输入 shape（TRT 10.x API）
        for name in self.input_names:
            if "images" in name.lower():
                self.context.set_input_shape(name, [1, 3, kInputH, kInputW])

        # 分配 host / device buffers
        self.buffer_h = []
        self.buffer_d = []
        for name in self.io_names:
            shape = self.context.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            print(f"Tensor '{name}': shape={shape}, dtype={dtype}, nptype={trt.nptype(dtype)}")
            self.buffer_h.append(np.empty(shape, dtype=trt.nptype(dtype)))
            self.buffer_d.append(cudart.cudaMalloc(self.buffer_h[-1].nbytes)[1])

        # 预设置 tensor device 地址
        for i, name in enumerate(self.io_names):
            self.context.set_tensor_address(name, self.buffer_d[i])

        # 创建 CUDA stream
        self.stream = cudart.cudaStreamCreate()[1]

        # 记录输出 tensor 在 buffer_h 中的索引
        self.output_indices = {
            name: self.io_names.index(name) for name in self.output_names
        }

        self.input_dtypes = {}
        for name in self.input_names:
            trt_dtype = self.engine.get_tensor_dtype(name)
            self.input_dtypes[name] = trt.nptype(trt_dtype)

    def _copy_to_buffer(self, index, data):
        """将 data 安全地拷贝到 self.buffer_h[index]（要求形状和类型完全匹配）"""
        buf = self.buffer_h[index]
        if data.shape != buf.shape or data.dtype != buf.dtype:
            raise ValueError(f"Buffer mismatch: expected {buf.shape} {buf.dtype}, got {data.shape} {data.dtype}")
        np.copyto(buf, np.ascontiguousarray(data))

    def release(self):
        for b in self.buffer_d:
            cudart.cudaFree(b)
        cudart.cudaStreamDestroy(self.stream)

    def get_engine(self):
        if os.path.exists(self.trt_file):
            with open(self.trt_file, "rb") as f:
                engine_string = f.read()
            if engine_string is None:
                print("Failed getting serialized engine!")
                return
            print("Succeeded getting serialized engine!")
        else:
            builder = trt.Builder(self.logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            profile = builder.create_optimization_profile()
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
            if use_fp16_mode:
                config.set_flag(trt.BuilderFlag.FP16)
            if use_int8_mode:
                config.set_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = calibrator.MyCalibrator(
                    calibration_data_dir, n_calibration,
                    (8, 3, kInputH, kInputW), cache_file
                )

            parser = trt.OnnxParser(network, self.logger)
            if not os.path.exists(onnx_file):
                print("Failed finding ONNX file!")
                return
            print("Succeeded finding ONNX file!")
            with open(onnx_file, "rb") as model:
                if not parser.parse(model.read()):
                    print("Failed parsing .onnx file!")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return
                print("Succeeded parsing .onnx file!")

            n_inputs = network.num_inputs
            input_names = [network.get_input(i).name for i in range(n_inputs)]
            for name in input_names:
                if "images" in name.lower():
                    profile.set_shape(name,
                                    [1, 3, kInputH, kInputW],
                                    [1, 3, kInputH, kInputW],
                                    [1, 3, kInputH, kInputW])
            config.add_optimization_profile(profile)

            engine_string = builder.build_serialized_network(network, config)
            if engine_string is None:
                print("Failed building engine!")
                return
            print("Succeeded building engine!")
            with open(self.trt_file, "wb") as f:
                f.write(engine_string)

        engine = trt.Runtime(self.logger).deserialize_cuda_engine(engine_string)
        return engine

    def inference_one(self, image_input):
        """
        使用 tensorrt runtime 做一次推理 (TRT 10.x API)

        :param image_input: [1, 3, H, W] float32
        :return: dict, 包含所有输出 tensor
        """
        for i, name in enumerate(self.io_names):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if "images" in name.lower():
                    self._copy_to_buffer(i, image_input)
                    cudart.cudaMemcpy(self.buffer_d[i],
                                    self.buffer_h[i].ctypes.data,
                                    self.buffer_h[i].nbytes,
                                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        self.context.execute_async_v3(self.stream)
        cudart.cudaStreamSynchronize(self.stream)

        # 拷贝所有输出 tensor
        outputs = {}
        for name in self.output_names:
            idx = self.output_indices[name]
            cudart.cudaMemcpy(
                self.buffer_h[idx].ctypes.data, self.buffer_d[idx],
                self.buffer_h[idx].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
            )
            outputs[name] = self.buffer_h[idx]
        return outputs

    def inference(self, image, classes=None):
        """
        对单张图片进行推理

        :param image: numpy array, BGR 格式
        :param classes: list[int] 或 None，只保留指定类别
        :return: [N, 6] 检测框，每行为 [x1, y1, x2, y2, conf, class_id]
        """
        # 预处理：直接 resize
        input_data = preprocess(image, kInputH, kInputW)
        input_data = np.expand_dims(input_data, axis=0)

        # 推理
        outputs = self.inference_one(input_data)

        # 解析输出：YOLOE 只有一个 output0
        output0 = None
        for name, tensor in outputs.items():
            if "output0" in name.lower():
                output0 = tensor
                break

        if output0 is None:
            raise RuntimeError(f"未找到 output0 输出，可用输出: {list(outputs.keys())}")
        # 后处理
        detect_res = postprocess(
            image, output0,
            self.conf_thresh,
            classes=classes
        )

        return detect_res

    @staticmethod
    def draw_image(detected_res, image, line_color=(255, 0, 255),
                   label_color=(255, 255, 255), line_thickness=2):
        """在图像上绘制检测框和标签"""
        for x1, y1, x2, y2, conf, class_id in detected_res:
            c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(image, c1, c2, line_color,
                         thickness=line_thickness, lineType=cv2.LINE_AA)

            label = f"{class_name_list[int(class_id)]} {conf:.2f}"
            t_size = cv2.getTextSize(
                label, 0, fontScale=line_thickness / 3,
                thickness=line_thickness
            )[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, line_color, -1, cv2.LINE_AA)
            cv2.putText(image, label, (c1[0], c1[1] - 2), 0,
                       line_thickness / 3, label_color,
                       thickness=line_thickness, lineType=cv2.LINE_AA)
