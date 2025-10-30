cat > README.md << 'EOF'
# SGLang 性能测试工具

## 🚀 快速开始

### 安装
```bash
# 安装 SGLang
pip install "sglang[all]"

# 验证安装
python -c "import sglang; print(sglang.__version__)"
```

### 启动服务端
```bash
# 方法 1: 使用 Python 模块（推荐）
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-hf \
    --host 0.0.0.0 \
    --port 8000 \
    --tp-size 1

# 方法 2: 使用启动脚本
./start_server.sh meta-llama/Llama-2-7b-hf 8000 1

# 方法 3: 一键测试
./test_sglang.sh
```

### 常用参数
```bash
python -m sglang.launch_server \
    --model-path MODEL_PATH \       # 模型路径（必需）
    --host 0.0.0.0 \                # 服务器地址
    --port 8000 \                   # 端口号
    --tp-size 1 \                   # Tensor Parallel 大小
    --mem-fraction-static 0.9 \     # GPU 显存使用比例
    --context-length 4096           # 最大上下文长度
```

### 运行测试
```bash
# ShareGPT 数据集
python client.py \
    --dataset sharegpt \
    --dataset-path ../data/sharegpt_data.json \
    --duration 60 \
    --rps-schedule "0:10" \
    --save-results

# LongBench 数据集
python client.py \
    --dataset longbench \
    --dataset-path ../data/longbench/narrativeqa.jsonl \
    --max-input-length 4096 \
    --duration 60 \
    --rps-schedule "0:5" \
    --save-results
```

## 📊 与 vLLM 对比

| 特性 | SGLang | vLLM |
|------|--------|------|
| **核心技术** | RadixAttention | PagedAttention |
| **前缀缓存** | ✅ 自动 | ⚠️ 需配置 |
| **多轮对话** | ✅✅ 优化更好 | ✅ 支持 |
| **API 兼容** | ✅ OpenAI | ✅ OpenAI |
| **客户端** | 完全相同 | 完全相同 |

## 🔍 验证服务端
```bash
# 健康检查
curl http://localhost:8000/health

# 简单测试
curl -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Hello!",
        "max_tokens": 50
    }'
```

## 📁 文件说明
```
sglang_benchmark/
├── start_server.sh     # 服务端启动脚本
├── test_sglang.sh      # 一键测试脚本
├── client.py           # 测试客户端（与 vLLM 相同）
├── data_loader.py      # 数据加载器（与 vLLM 相同）
└── README.md           # 本文件
```

## 🐛 常见问题

### 问题 1: ModuleNotFoundError
```bash
# 重新安装 SGLang
pip uninstall sglang -y
pip install "sglang[all]"

# 或安装最新版
pip install "git+https://github.com/sgl-project/sglang.git"
```

### 问题 2: 端口被占用
```bash
# 检查端口
netstat -tlnp | grep 8000

# 使用其他端口
python -m sglang.launch_server \
    --model-path YOUR_MODEL \
    --port 8001
```

### 问题 3: CUDA 错误
```bash
# 检查 CUDA
nvidia-smi

# 设置单 GPU
export CUDA_VISIBLE_DEVICES=0

# 重新启动
./start_server.sh
```

## 📝 开源协议

MIT License

---