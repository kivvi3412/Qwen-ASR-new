# Qwen3-ASR 极速视频字幕生成系统 (Qwen-ASR-new)

> 高吞吐视频转字幕 + 大模型字幕纠错一体化 Web 服务。
> ASR 与纠错大模型跑在**两套完全独立的 uv 环境**里，互不干扰，顶层 `start.sh` 一键拉起。

---

## 📌 1. 项目结构与两套环境

```
Qwen-ASR-new/
├── start.sh          # 一键启动：先拉起 9B 纠错服务(8002)，再前台运行 ASR 服务(8001)
├── stop.sh           # 停止两个服务并释放显存
├── README.md
├── json_to_srt.py    # 独立高精度通用声学停顿切分 CLI 脚本 (JSON -> SRT)
├── ai_sub_corrector.py # 独立通用大模型逐行纠错 CLI 脚本 (默认 20 并发)
├── data/tasks/       # 运行时任务数据（历史记录、输出副本、纠错缓存）
│
├── asr/              # ① ASR 字幕服务 + Web 控制台（独立 uv 环境）
│   ├── pyproject.toml / uv.lock
│   ├── .venv/        #   vLLM 0.14.0 + transformers 4.57.6（qwen-asr 官方适配组合）
│   ├── service/
│   │   ├── server.py       # FastAPI 路由：扫描/上传/任务/SSE/下载/纠错
│   │   ├── engine.py       # ASRInferenceEngine：vLLM 连续批处理 + ForcedAligner 单例
│   │   ├── task_queue.py   # 任务队列：转录流水线 + JSON 纠错流水线（共用看板/SSE）
│   │   ├── json_resegment.py # 升级版历史 JSON 清洗重排引擎（声学停顿锚定）
│   │   ├── correction_guard.py # 守门员 2.0（声母混淆组、严格数值锁、口吃平滑）
│   │   ├── subtitle.py     # 工业级字幕分句引擎（字符对齐、语义断句、连词前移）
│   │   └── corrector.py    # 大模型纠错客户端：领域画像 + 逐行锚定纠错 + 20并发支持
│   └── web/index.html      # 单页云控制台前端
│
└── llm4b/            # ② Qwen3.5-9B-AWQ-4bit 纠错大模型服务（独立 uv 环境，端口 8002）
    ├── pyproject.toml / uv.lock
    ├── run_server.sh # 独立启动 9B 纠错模型脚本 (max-num-seqs 20, enable_thinking=false)
    └── .venv/        #   vLLM 0.29.0 + transformers 5.x（Qwen3.5 架构所需）
```

两个环境只通过 `http://127.0.0.1:8002/v1`（OpenAI 兼容 API）通信，**没有任何 Python 依赖交叉**。

### 1.1 为什么必须两套环境？（关键决策）

| 组件 | 需要的栈 | 说明 |
| --- | --- | --- |
| ASR 主模型 `Qwen3-ASR-0.6B` + `Qwen3-ForcedAligner-0.6B` | `qwen-asr[vllm]==0.0.6` → **vLLM 0.14.0 + transformers 4.57.6** | qwen-asr 的 ForcedAligner 使用自带的 transformers 后端代码，**只在 4.57.x 上可用**（PyPI 无更新版本） |
| 纠错大模型 `cyankiwi/Qwen3.5-9B-AWQ-4bit` | **vLLM ≥ 0.27.1 + transformers ≥ 5.5.3** | `Qwen3_5ForCausalLM` 架构只在 vLLM 0.27.1+ 注册；0.14.0 完全没有该架构 |

两者对 `transformers` 的要求直接冲突（4.57 vs 5.x），因此拆成两个 venv 是最干净的解法：
ASR 环境跑官方适配的 4.57 组合，纠错环境跑最新稳定 vLLM，互不牵制。

> vLLM 版本与 Qwen3.5 支持对照（实测 registry）：
> `0.14.0 / 0.17 / 0.18 / 0.19 / 0.24 / 0.26` ❌ 不支持 `Qwen3_5*`；
> `0.27.1`（首个支持）、`0.28.0`、`0.29.0` ✅ 支持；要升级 LLM 环境时改 `llm4b/pyproject.toml` 里的 `vllm==` 版本即可。

---

## 📌 2. 服务器与代码同步

- **服务器**：`192.168.2.16`（用户 `kivvi`），GPU：NVIDIA A100 32GB
- **远程目录**：`/home/kivvi/Qwen-ASR-new/`
- **本地目录**：`/Users/kivvi/Downloads/Qwen-ASR-new/`
- **Web 控制台**：`http://192.168.2.16:8001`

```bash
# 本地 -> 服务器 推送
rsync -avz --exclude '.DS_Store' --exclude '__pycache__' --exclude 'data' \
      --exclude '.venv' /Users/kivvi/Downloads/Qwen-ASR-new/ kivvi@192.168.2.16:/home/kivvi/Qwen-ASR-new/

# 服务器 -> 本地 拉取
rsync -avz --exclude '.DS_Store' --exclude '__pycache__' --exclude 'data' \
      --exclude '.venv' kivvi@192.168.2.16:/home/kivvi/Qwen-ASR-new/ /Users/kivvi/Downloads/Qwen-ASR-new/
```

> 若提示 `./start.sh: Permission denied`，说明脚本丢失了可执行位：`chmod +x start.sh stop.sh`

---

## 📌 3. 部署与启动

```bash
# 首次部署：分别同步两个环境（各自独立，可并行执行）
cd asr   && uv sync      # 约 8~11GB：torch + vLLM 0.14 + transformers 4.57
cd ../llm4b && uv sync   # 约 5~8GB：torch + vLLM 0.29

# 启动（先拉起 4B 纠错服务并等待就绪，再前台运行 ASR 服务）
./start.sh
# 后台常驻：
nohup ./start.sh > server.log 2>&1 &

# 停止并释放显存
./stop.sh
```

`start.sh` 会自动：
1. 检查两个 `.venv`，缺失时自动 `uv sync`；
2. 关闭 FlashInfer 采样（`VLLM_USE_FLASHINFER_SAMPLER=0`）并把 `CUDA_HOME` 指向 venv 内置 nvcc（本机无系统 CUDA Toolkit，详见 §6.1）；
3. 在 8002 启动 `Qwen3.5-4B`（日志 `llm4b.log`）并等待 `/v1/models` 就绪；
4. 在 8001 前台启动 ASR 服务 + Web 控制台。

---

## 📌 4. 功能总览

### 4.1 视频转字幕（转录任务）
- **服务器本地扫描**：输入 NAS 目录 → 扫描视频 → 勾选批量创建任务（服务器端零拷贝读取，不占用上传带宽）；
- **网页上传**：直接拖拽上传视频文件转录；
- **看板实时监控**：SSE 推送每条视频的切片进度、字幕条数、RTF 实时倍速，可中途强制终止、打包下载；
- 输出：与视频同目录（或 `srt/` 子目录）生成 `.srt` / `.txt` / `.json`。

### 4.2 服务器已有 JSON 批量纠错（本次新增）
- 在「JSON 纠错」页输入服务器目录 → **扫描目录**（递归可选）→ 列表展示每个 JSON 的字幕条数 / 是否已有 SRT / 领域标签；
- 支持整目录全选、按文件夹全选反选、任意多选；
- 点击「创建批量纠错任务」后**同样进入任务看板**：每个 JSON 独立进度条、Stage 1 领域画像 / Stage 2 逐行纠错状态、实时耗时，支持终止与打包下载；
- 纠错完成后**原地覆盖**同目录下的 `.srt` / `.txt` / `.json`，同时在任务目录保留一份副本供下载。

### 4.3 上传 JSON 纠错
单文件上传 → 纠错 → 直接下载 SRT/TXT/JSON（不需要服务器路径）。

### 4.4 大模型纠错引擎（`asr/service/corrector.py`）
- **Stage 1 领域画像**：对全文多点采样，提炼学科领域、核心议题、专有名词库与常见同音字错误；
- **Stage 2 逐行锚定纠错**：严格按 `[ID]` 编号逐行纠错，100% 保持原时间戳不变；
- **断点缓存**：`data/tasks/<task_id>/llm_cache/*.corrections.json`，重跑时自动复用已纠错批次；
- **盘古排版**：自动规范中英文/数字间距。

### 4.5 纠错守门员（`asr/service/correction_guard.py`）——只放行「发音相近的局部修改」
大模型偶尔会把相邻行的文字搬来搬去、整句换成别的内容，或做发音跨度很大的“术语替换”。
守门员对每行「原文 → 模型输出」做三重校验，不通过就**丢弃模型输出、保留原文**：
1. **规范化白名单**：差异能由 `text_normalize` 的通用规则解释（缩写合并、中文数字、空格、标点）→ 放行；
2. **长度守卫**：忽略标点后的正文长度变化不得超过 ±35%（拦截跨行搬迁/整句替换）；
3. **发音守卫**：逐字符对齐，替换的字符必须拼音相同/编辑距离≤1/同声母/同韵母（数字互转、字母重排如 QMA→QAM 视为写法差异放行）。

### 4.6 通用文本规范化（`asr/service/text_normalize.py`）——与领域无关
- 英文缩写被逐字母读开 → 合并并大写（`t c p` → `TCP`、`a p` → `AP`），**仅在中文语境生效**，不会破坏英文原文；
- 中文数字读法 → 阿拉伯数字（`一百二十八` → `128`、`三点一四` → `3.14`、`八零二点幺幺` → `802.11`）；
- 数字/单位、中英文空格规范（盘古排版）。

**领域词不进代码**：需要固定某个领域的写法时，编辑 `asr/service/glossary.json`（改完保存即热加载，无需重启）：
```json
{
  "acronyms": { "CSMACA": "CSMA/CA" },
  "terms":    { "社恐": "时隙" }
}
```
不填这个文件时，系统只按通用规则处理，不做任何领域特调。

---

## 📌 5. 核心调优决策备忘

### 决策 1：ASR 为什么用 0.6B 而不是 1.7B？
考研专业课实测中两者字错率差异极小（语病/口误都无法凭空修正），但 0.6B 权重轻、KV 缓存小，配合 vLLM 连续批处理吞吐是 1.7B 的 3~4 倍，32GB 卡上更稳。

### 决策 2：为什么 `ASR gpu_util=0.25`、`LLM gpu_util=0.45`、`max_len=4096`？
Qwen 官方默认 `max_model_len=65536`，会让 vLLM 预分配海量 KV-Cache 直接吃满显存；长视频已在前端按 300s 静音切片，单切片不可能超过 4096 token，因此锁死 4096。

显存是三方共用的硬约束：ASR 引擎（0.25 → 约 7.9GB）、纠错大模型（0.45 → 约 14.2GB）、以及**跑在 ASR Python 进程里、不占 vLLM 额度的 ForcedAligner**（权重约 2GB + 4×180s 对齐批次的峰值激活约 2GB）。三方相加约 26GB，留出约 5GB 余量。若把两者提到 0.35/0.50，整卡只剩不到 1GB，长批次对齐时的激活峰值就会 OOM。

### 决策 3：如何压满 GPU 功率？
删除了切片内部的 `torch.cuda.empty_cache()`（会全局同步阻塞 CUDA 流，导致显卡瞬时掉电），依靠 PyTorch 显存池复用；CPU 线程池异步抽音频，默认 5 路并发让 GPU 流水线不断流。

### 决策 4：字幕分句引擎（`asr/service/subtitle.py`）
四阶语义切分：① 字符级单调平滑插值 → ② 标点语法主断句（10~22 字、1.5~5.0s 黄金区间）→ ③ 无标点长句基于连接词/声学停顿的递归语义断句 → ④ 连词前移吸附（`然后/但是/所以` 移到下一句句首）+ 防闪烁保底 0.8s。

### 决策 5：界面风格
全面云控制台化（深蓝黑 `#0b0f19`、大厂蓝 `#2563eb`、Monospace 指标网格），右上角硬件胶囊实时展示 GPU 利用率与显存。

---

## 📌 6. 故障记录（踩坑与修复）

### 6.1 ASR 引擎启动即崩溃：`Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`
- **现象**：EngineCore 抛 `RuntimeError: Could not find nvcc ...` → `Engine core initialization failed`，网页状态一直 `is_loaded: false`。
- **根因**：本机没有系统级 CUDA Toolkit。vLLM 默认启用 **FlashInfer top-k/top-p 采样器**，该算子不随 wheel 预编译，首次采样时触发 JIT 编译并强依赖 `nvcc`。
- **修复**：`start.sh` 全局 `export VLLM_USE_FLASHINFER_SAMPLER=0`（回退 PyTorch 原生采样，字幕/短文本场景无性能损失），并把 `CUDA_HOME` / `PATH` 指向 venv 内 pip 安装的 `nvidia/cu*/bin/nvcc` 作为兜底。
- **验证**：启动日志出现 `FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0.` 即生效。

### 6.2 ForcedAligner 加载失败：`KeyError: 'default'` / `pad_token_id` / `create_causal_mask`
- **现象**：ASR 引擎在加载对齐模型时崩溃（早期版本把 ASR 与 Qwen3.5-4B 装在同一环境，transformers 被 vLLM nightly 升到 5.x）。
- **根因**：qwen-asr 0.0.6 的 ForcedAligner 是按 `transformers==4.57.6` 写的，5.x 移除了 `ROPE_INIT_FUNCTIONS["default"]`、`PretrainedConfig.pad_token_id` 默认值、改了 `create_causal_mask` 签名。
- **修复**：拆成 `asr/`（transformers 4.57.6）与 `llm4b/`（transformers 5.x）两套 uv 环境，各用官方适配版本，**不再需要任何运行时补丁/兼容层**。

---

## 📌 7. 常用维护

```bash
# 实时日志
tail -f /home/kivvi/Qwen-ASR-new/llm4b.log      # 4B 纠错服务
tail -f /home/kivvi/Qwen-ASR-new/server.log     # ASR 服务（后台启动时）

# 健康检查
curl -s http://127.0.0.1:8001/api/status | jq .
curl -s http://127.0.0.1:8002/v1/models | jq '.data[].id'

# 显存/进程
nvidia-smi
pgrep -af "service.server:app|vllm serve"
```

---

## 🎯 8. 后续可拓展方向
- **双阶段并发纠错**：同一 JSON 的分批结果并行提交，进一步压榨 4B 模型吞吐；
- **标点/大小写还原**：引入轻量标点恢复模型或专业术语热词表；
- **Docker 交付**：ASR 与 LLM 各出一个镜像，用 compose 编排两个服务。
