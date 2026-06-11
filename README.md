# 人流事件 Webhook 通知器

基于 YOLO11 + ByteTrack + SHM 的多摄像头实时人流检测系统。检测到人员进入/离开时，自动向指定 Webhook 接口发送 HTTP 通知。

---

## 功能特点

- **多摄像头并发**：每个摄像头独立线程，支持同时监控多个出入口
- **Webhook 通知**：检测到进出事件时自动发送 HTTP 请求
- **配置文件驱动**：所有参数通过 `config.yaml` 配置，无需修改代码
- **事件图片**：可选保存进出事件截图
- **日志归档**：使用 loguru 记录运行日志，每天自动轮转，保留 7 天

---

## 项目结构

```
.
├── config.yaml           # 摄像头和 webhook 配置
├── main.py               # 入口脚本
├── notifier/             # 核心模块
│   ├── config.py         # YAML 配置加载
│   ├── main.py           # 后台检测主循环
│   └── webhook.py        # HTTP 通知发送
├── counting/             # 检测线计数逻辑
├── infer/                # YOLO 检测 + ByteTrack 跟踪
├── stream/               # SHM 视频流客户端
├── models/               # TensorRT 模型文件
├── logs/                 # 运行日志（运行时自动生成）
├── events/               # 事件图片（如开启保存）
├── Dockerfile            # Docker 构建文件
└── requirements.txt      # Python 依赖
```

---

## 快速开始（Docker）

### 1. 准备配置文件

编辑 `config.yaml`，配置摄像头 RTSP 地址和 webhook 接收地址：

```yaml
cameras:
  - name: "entrance"
    rtsp_url: "rtsp://admin:password@192.168.1.10:554/stream"
    server: "172.16.20.193:50052"      # SHM gRPC 流服务地址
    line: "696,549,2138,1185"           # 检测线坐标 x1,y1,x2,y2
    reverse: false
    detect_model: "./models/trt_models/yolo11s.plan"
    gpu_id: 0
    event_dir: "./events"               # 可选：保存事件图片

webhooks:
  - url: "http://your-server/api/notify"
    method: "POST"
    headers:
      Content-Type: "application/json"
    body_template:
      camera: "{camera_name}"
      event: "{direction}"
      current_count: "{current_count}"
    events: ["in", "out"]
    cooldown_seconds: 0
```

### 2. 构建镜像

```bash
docker build -t counting-notifier:latest .
```

### 3. 启动容器

**方式一：挂载代码目录（开发/调试，推荐）**

代码修改后无需重新构建镜像，重启容器即可生效：

```bash
docker run -itd --name counting-notifier \
    --gpus all \
    --shm-size=16g \
    --ipc=host \
    -v $(pwd):/workspace/counting \
    -e TZ=Asia/Shanghai \
    trt-counting :latest \
    python main.py --config config.yaml
```

**方式二：直接运行（生产环境）**

代码已打包进镜像，适合部署：

```bash
docker run -itd --name counting-notifier \
    --gpus all \
    --shm-size=16g \
    --ipc=host \
    -e TZ=Asia/Shanghai \
    counting-notifier:latest
```

### 4. 查看日志

```bash
# 实时查看控制台输出
docker logs -f counting-notifier

# 查看日志文件（挂载模式下直接在宿主机查看）
tail -f logs/app_$(date +%Y-%m-%d).log
```

### 5. 停止/重启

```bash
# 停止
docker stop counting-notifier

# 重启
docker restart counting-notifier

# 删除容器
docker rm -f counting-notifier
```

---

## 本地运行（不依赖 Docker）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行
python main.py --config config.yaml
```

---

## 配置说明

### cameras（摄像头）

| 字段 | 必填 | 说明 |
|------|------|------|
| `name` | ✅ | 摄像头标识，如 `entrance` |
| `rtsp_url` | ✅ | RTSP 流地址 |
| `line` | ✅ | 检测线坐标 `x1,y1,x2,y2` |
| `server` | ❌ | SHM gRPC 服务地址，默认 `172.16.20.193:50052` |
| `detect_model` | ❌ | TensorRT 模型路径 |
| `gpu_id` | ❌ | 推理 GPU ID，默认 `0` |
| `reverse` | ❌ | 是否反转 IN/OUT 方向，默认 `false` |
| `event_dir` | ❌ | 事件图片保存目录，不填则不保存 |

### webhooks（通知接口）

| 字段 | 必填 | 说明 |
|------|------|------|
| `url` | ✅ | 接收通知的 HTTP 地址 |
| `method` | ❌ | HTTP 方法，默认 `POST` |
| `headers` | ❌ | 请求头 |
| `body_template` | ❌ | 消息体模板，支持占位符 |
| `events` | ❌ | 监听事件，`["in"]` 只发进入，`["in","out"]` 都发 |
| `cooldown_seconds` | ❌ | 冷却时间，默认 `0` |

**body_template 支持的占位符：**

| 占位符 | 说明 |
|--------|------|
| `{camera_name}` | 摄像头名称 |
| `{direction}` | 事件方向：`in` 或 `out` |
| `{track_id}` | 跟踪 ID |
| `{timestamp}` | 事件时间 |
| `{current_count}` | 当前在馆人数 |
| `{image_path}` | 事件图片路径（如开启保存） |

---

## 日志说明

- **控制台输出**：实时显示运行状态
- **文件归档**：`logs/app_YYYY-MM-DD.log`
- **自动轮转**：每天零点生成新日志文件
- **保留策略**：自动清理 7 天前的旧日志

日志格式：
```
2026-04-27 11:09:55.123 | INFO     | notifier:main:52 - 消息内容
```

---

## 注意事项

1. **SHM 流服务**：需要先启动 gRPC 流服务（`172.16.20.193:50052`），程序才能拉取视频流
2. **时区**：容器内已设置为 `Asia/Shanghai`，日志和事件时间均为北京时间
3. **模型文件**：确保 `./models/trt_models/yolo11s.plan` 存在，或使用自己的 TensorRT 模型
4. **GPU**：运行时需要 `--gpus all` 参数，确保容器能访问 NVIDIA GPU
