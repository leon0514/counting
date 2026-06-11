# ================================================================
# 人流量统计 Webhook 通知器 Dockerfile
# 基础镜像: nvcr.io/nvidia/tensorrt:25.10-py3 (CUDA 12.6 + TensorRT 10.x + Python 3.12)
# ================================================================

FROM nvcr.io/nvidia/tensorrt:25.10-py3

WORKDIR /workspace/counting

# ------------------------------------------------------------------
# 1. 安装系统编译工具
#    cython_bbox、lap 等包包含 C/C++ 扩展，需要在容器内编译
# ------------------------------------------------------------------
# 安装系统工具：
#   - build-essential: 编译 cython_bbox / lap 等 C 扩展
#   - tzdata: 时区数据，解决容器内 UTC 与 CST(UTC+8) 的 8 小时时差问题
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# 设置容器时区为东八区（中国标准时间）
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ------------------------------------------------------------------
# 2. 解决 numpy 版本冲突（关键步骤）
#
#   - 基础镜像预装 numpy 2.x 和 opencv-python。
#   - cython_bbox 在编译时绑定 numpy C API，目前只兼容 numpy 1.x。
# ------------------------------------------------------------------
RUN pip uninstall -y opencv-python opencv-python-headless numpy && \
    pip install --no-cache-dir "numpy<2" cython opencv-python-headless

# ------------------------------------------------------------------
# 3. 安装项目 Python 依赖
# ------------------------------------------------------------------
RUN pip install --no-cache-dir \
    cuda-python \
    scipy \
    lap \
    cython_bbox \
    grpcio \
    requests \
    pyyaml \
    loguru \
    protobuf

# ------------------------------------------------------------------
# 4. 复制项目代码
# ------------------------------------------------------------------
# COPY . /workspace/counting/

# ------------------------------------------------------------------
# 5. 默认启动命令（从配置文件加载摄像头和 webhook）
# ------------------------------------------------------------------
CMD ["python", "main.py", "--config", "config.yaml"]
