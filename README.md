# HBserve

基于vLLM实现的推理引擎

## 快速开始

### 1. 安装依赖

```bash
# 基础依赖
cd HBserve
pip install -e . 

```

### 2. 下载模型

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

### 3. 运行示例


```bash
python example.py

```


## 贡献

欢迎提交Issue和Pull Request！

## 参考

- [nanovllm](https://github.com/GeeeekExplorer/nanovllm)
- [vLLM](https://github.com/vllm-project/vllm)
- [Mooncake](https://github.com/kvcache-ai/Mooncake)
