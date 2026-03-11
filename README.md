# auto-barbeque

面向汽车视频场景的字幕处理流水线。当前仓库用于把英文 SRT 结合视频帧证据修订为更可靠的中文字幕，同时提供一个 WhisperX 断句工具和一个独立的 YOLO 帧过滤脚本。

## 仓库内容

- `barbeque_pipeline.py`: 主流程 CLI，包含 `prepare`、`run`、`merge`、`segment`、`refine`、`full`
- `run_whisperx_guide_dp.py`: WhisperX + guide-rule + DP 的英文单行断句工具
- `filter_frames_yolo.py`: 对已抽帧目录执行 YOLO-World 人像占比过滤
- `schema.json`: LLM 输出 JSON schema
- `barbeque_ollama.yaml`: `run` 阶段使用 Ollama 的 YAML 示例
- `config_jzx_full.yaml`: `full` 流程 YAML 示例
- `segmentguide.md`: WhisperX 断句约束说明

## 环境要求

主流水线：

- Windows / Linux 均可，本文示例以 Windows PowerShell 为主
- Python 3.9+
- `ffmpeg` 在 PATH 中
- 以下翻译后端至少有一个可用：`codex` CLI、`opencode` CLI、`ollama`

可选能力：

- YAML 配置：`PyYAML`
- YOLO 过滤：`ultralytics` + YOLO-World 模型文件（默认 `yolov8x-worldv2.pt`）
- WhisperX 断句：建议单独使用 `conda whisper` 环境，并安装 `whisperx`、`deepmultilingualpunctuation`

## 安装

基础环境：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
```

如果要启用 YOLO 过滤，再额外安装：

```powershell
pip install ultralytics
```

说明：

- `barbeque_pipeline.py` 本身主要依赖标准库，`--config` 需要 `PyYAML`
- `requirements.txt` 还包含 Whisper 辅助脚本用到的依赖
- `run_whisperx_guide_dp.py` 仍需在自己的 WhisperX 环境里安装 `whisperx`

## 主流程

### 1. prepare

解析英文 SRT，生成任务目录，并按字幕时间抽取帧。

```powershell
python .\barbeque_pipeline.py prepare `
  --video "D:\video\demo.mp4" `
  --srt "D:\video\demo.en.srt" `
  --tasks-dir ".\output\demo\tasks" `
  --schema ".\schema.json" `
  --frame-mode adaptive `
  --yolo-filter
```

常用参数：

- `--frame-mode three|mid|none|adaptive`
- `--skip-frames`
- `--overwrite-frames`
- `--scale-width 854`
- `--yolo-filter --yolo-model yolov8x-worldv2.pt`

### 2. run

读取任务目录，调用 `codex` / `opencode` / `ollama` 生成每条字幕的 `result.json`。

```powershell
python .\barbeque_pipeline.py run `
  --tasks-dir ".\output\demo\tasks" `
  --schema ".\schema.json" `
  --translator codex `
  --model gpt-5.2 `
  --image-policy auto `
  --workers 4
```

常用参数：

- `--translator codex|opencode|ollama`
- `--image-policy three|mid|auto|none|rand1`
- `--serial-batch-size 30 --serial-overlap 8 --serial-history-size 6`
- `--proxy ""`：禁用默认代理 `http://127.0.0.1:7890`
- `--search`
- `--mock`

### 3. merge

把 `result.json` 合并回中文字幕 SRT，并可按模型返回的 `join_with_prev` 和锚点比例微调时间轴。

```powershell
python .\barbeque_pipeline.py merge `
  --srt "D:\video\demo.en.srt" `
  --tasks-dir ".\output\demo\tasks" `
  --out ".\output\demo\zh.srt" `
  --min-confidence 0.7 `
  --use-anchor-timing
```

### 4. full

一条命令完成 `prepare + run + merge`，并可选执行 `refine`。

```powershell
python .\barbeque_pipeline.py full `
  --video "D:\video\demo.mp4" `
  --srt "D:\video\demo.en.srt" `
  --slug demo `
  --schema ".\schema.json" `
  --translator opencode `
  --model "minimax/MiniMax-M2.5" `
  --frame-mode adaptive `
  --image-policy auto `
  --workers 4 `
  --yolo-filter `
  --refine-context-pass
```

使用 `--slug demo` 时，默认输出到 `output/demo/`。

## 后处理命令

### segment

对已有中文字幕 SRT 做自然分段，适合在 `merge` 之后微调节奏。

```powershell
python .\barbeque_pipeline.py segment `
  --in-srt ".\output\demo\zh.srt" `
  --out ".\output\demo\zh.segmented.srt" `
  --passes 2 `
  --target-chars 18
```

### refine

按顺序读取英文原文和中文字幕，对上下文做二次润色；必要时允许联动修改上一句。

```powershell
python .\barbeque_pipeline.py refine `
  --srt "D:\video\demo.en.srt" `
  --in-srt ".\output\demo\zh.srt" `
  --out ".\output\demo\zh.refined.srt" `
  --translator codex `
  --model gpt-5.2
```

## YAML 配置

CLI 支持通过 `--config` 预加载 YAML 默认值，命令行参数优先级更高。

`full` 流程示例：

```powershell
python .\barbeque_pipeline.py --config .\config_jzx_full.yaml full `
  --video "D:\video\demo.mp4" `
  --srt "D:\video\demo.en.srt" `
  --slug jzx100
```

`run` 阶段使用 Ollama：

```powershell
python .\barbeque_pipeline.py --config .\barbeque_ollama.yaml run `
  --tasks-dir ".\output\demo\tasks" `
  --schema ".\schema.json"
```

配置说明：

- YAML key 支持 `snake_case`
- 未知 key 会直接报错，避免静默拼写错误
- `barbeque_ollama.yaml` 适合本地串行翻译
- `config_jzx_full.yaml` 适合一次性跑完整流程

## WhisperX Guide DP 断句

`run_whisperx_guide_dp.py` 会先拿到词级时间戳，再根据 `segmentguide.md` 的约束做全局最优断句。

默认硬约束：

- 单条时长：`[5/6, 7]` 秒
- 单条字幕：`1` 行
- 每条最大字符：`42`
- 最大读速：`20 CPS`

推荐在单独的 `conda whisper` 环境中运行：

```powershell
conda run -n whisper python .\run_whisperx_guide_dp.py "D:\video\demo.mp4" -o .\output_whisperx
```

如果已经有对齐后的 WhisperX JSON：

```powershell
conda run -n whisper python .\run_whisperx_guide_dp.py `
  --from-json `
  "D:\video\demo.aligned.json" `
  --no-restore-punctuation `
  -o .\output_whisperx
```

也可以使用批处理入口：

```bat
run_guide_segment.bat "D:\video\demo.mp4" -o .\output_whisperx
```

输出内容：

- `*.guide.srt`
- `*.guide.txt`
- `*.guide.json`
- `*.aligned.json`（启用保存时）

## YOLO 帧过滤

`filter_frames_yolo.py` 可以独立用于清理任务目录中的人像主导帧。

```powershell
python .\filter_frames_yolo.py ".\output\demo\tasks" `
  --ext jpg `
  --model yolov8x-worldv2.pt `
  --person-threshold 0.70 `
  --preview
```

如果你在 `prepare` / `full` 里已经开启 `--yolo-filter`，通常不需要再单独运行它。

## 输出目录

常见输出结构如下：

```text
output/
  demo/
    tasks/
      manifest.json
      000001/
        meta.json
        prompt.txt
        result.json
        start.jpg / mid.jpg / end.jpg
    zh.srt
    zh.refined.srt
```

运行产物默认写入 `output/`，模型文件和本地编辑器配置已被 `.gitignore` 排除。
