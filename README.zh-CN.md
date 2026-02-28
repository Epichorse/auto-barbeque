# auto-barbeque

[English](README.md) | [简体中文](README.zh-CN.md)

面向汽车视频场景的字幕处理流水线：将英文 SRT 结合画面上下文，修订为中文字幕。

核心流程：

1. 解析 SRT 字幕条目。
2. 为每个条目提取 `start/mid/end` 三帧。
3. 使用严格 JSON Schema 调用 `codex exec`。
4. 将 `revised_zh` 合并回新 SRT，并保持原时间轴不变。

## 运行要求

- Python 3.9+
- `PATH` 中可用的 `codex` CLI
- 帧提取运行环境：
  - 推荐：系统安装 `ffmpeg`
  - 备用：Python 包 `imageio-ffmpeg`

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

`requirements.txt` 只包含帧提取备用依赖；如果你已安装系统 `ffmpeg`，依然可以正常使用。

## 流水线命令

### 1) 准备任务

```bash
python3 barbeque_pipeline.py prepare \
  --video "/path/to/video.mp4" \
  --srt "/path/to/input.en.srt" \
  --tasks-dir work/tasks \
  --schema schema.json
```

如需跳过抽帧：

```bash
python3 barbeque_pipeline.py prepare \
  --video "/path/to/video.mp4" \
  --srt "/path/to/input.en.srt" \
  --tasks-dir work/tasks \
  --schema schema.json \
  --skip-frames
```

### 2) 运行 Codex

```bash
python3 barbeque_pipeline.py run \
  --tasks-dir work/tasks \
  --schema schema.json \
  --model gpt-5.2 \
  --search
```

离线 mock 模式：

```bash
python3 barbeque_pipeline.py run \
  --tasks-dir work/tasks \
  --schema schema.json \
  --mock
```

### 3) 合并修订字幕

```bash
python3 barbeque_pipeline.py merge \
  --srt "/path/to/input.en.srt" \
  --tasks-dir work/tasks \
  --out out_final/output.zh.srt \
  --min-confidence 0.6
```

### 4) 一条命令跑完整流程

```bash
python3 barbeque_pipeline.py full \
  --video "/path/to/video.mp4" \
  --srt "/path/to/input.en.srt" \
  --tasks-dir work/tasks \
  --schema schema.json \
  --out out_final/output.zh.srt \
  --model gpt-5.2 \
  --search
```

## 批处理脚本

`run_work_batch_parallel.py` 用于运行 `work/` 中预定义数据项。

```bash
python3 run_work_batch_parallel.py \
  --root . \
  --model gpt-5.2 \
  --cue-workers 8
```

如需代理：

```bash
python3 run_work_batch_parallel.py --proxy "http://127.0.0.1:7897"
```

## 输出目录结构

```text
work/
  tasks/
    manifest.json
    000001/
      start.png
      mid.png
      end.png
      meta.json
      prompt.txt
      result.json
      codex.stderr.log
```

`merge` 只会在 `result.json.revised_zh` 可用时替换字幕文本，不会改动时间轴。

## 开源仓库规范

仓库默认忽略大体积本地产物（`work/`、`out_final/`、模型、视频、压缩包、虚拟环境）。
推送前请参考 `.gitignore` 与 `OPEN_SOURCE_CHECKLIST.md`。

## 相关工具

`whisper_mp4_srt/` 是独立的 MP4 -> 英文 SRT 工具（Whisper）。

## 许可证

MIT（见 `LICENSE`）。
