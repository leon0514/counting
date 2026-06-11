# 人流事件 Webhook 通知器

基于 YOLO + ByteTrack + SHM 的多摄像头实时人流检测系统。检测到人员进入/离开时，自动向 Webhook 发送 HTTP 通知。

---

## 快速开始

### Docker 运行（推荐）

```bash
docker build -t counting-notifier:latest .
docker run -itd --name counting-notifier \
    --gpus all --shm-size=16g --ipc=host \
    -v $(pwd):/workspace/counting \
    -e TZ=Asia/Shanghai \
    counting-notifier:latest \
    python main.py --config config.yaml
```

查看日志：
```bash
docker logs -f counting-notifier
tail -f logs/app_$(date +%Y-%m-%d).log
```

### 本地运行

```bash
pip install -r requirements.txt
python main.py --config config.yaml
```

---

## 配置示例

```yaml
cameras:
  - name: "entrance"
    rtsp_url: "rtsp://admin:pass@192.168.1.10:554/stream"
    server: "172.16.20.193:50052"      # SHM gRPC 服务地址
    line: "696,549,2138,1185"           # 检测线 x1,y1,x2,y2
    gate_width: 60                      # 双线间距（像素）
    reverse: false                      # 是否反转 IN/OUT 方向
    model_type: "yolo"
    detect_model: "models/trt_models/yolo11s.plan"
    gpu_id: 0
    decoder: 1
    interval_ms: 0
    event_dir: "./events"               # 可选：保存事件截图

webhooks:
  - url: "http://your-server/api/notify"
    method: "POST"
    headers:
      Content-Type: "application/json"
    body_template:
      camera: "{camera}"
      event: "{event}"
      timestamp: "{timestamp}"
    events: ["in", "out"]
    cooldown_seconds: 0
```

**body_template 占位符：** `{camera}` `{event}` `{timestamp}`

---

## 项目结构

```
.
├── config.yaml        # 配置文件
├── main.py            # 入口脚本
├── notifier/          # 核心逻辑（配置加载 / 检测循环 / Webhook 发送）
├── counting/          # 检测线计数逻辑
├── infer/             # YOLO 检测 + ByteTrack 跟踪
├── stream/            # SHM 视频流客户端
├── models/            # TensorRT 模型文件
├── logs/              # 运行日志
├── events/            # 事件截图
├── Dockerfile
└── requirements.txt
```

---

## 注意事项

1. **SHM 流服务**：需先启动 gRPC 流服务，程序才能拉取视频流
2. **模型文件**：确保 TensorRT 模型路径正确
3. **GPU**：运行时需要 `--gpus all`，确保容器能访问 NVIDIA GPU
