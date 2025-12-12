# MiniMind 项目分析 (GEMINI.md)

## 1. 项目概述

`MiniMind` 是一个开源项目，旨在于以极低的成本（3元人民币，2小时）从零开始训练一个25.8M的超小型语言模型。该项目不仅仅是一个模型，更是一套完整的、从零开始使用 PyTorch 实现的 LLM 训练和学习教程。

### 核心目标与理念
*   **低门槛:** 让个人开发者也能体验和学习从数据处理到模型训练的全过程。
*   **白盒实现:** 核心算法（如 Attention、RoPE、LoRA、DPO、PPO 等）均由原生 PyTorch 从零构建，不依赖高度封装的第三方库，便于学习底层原理。
*   **全流程覆盖:** 提供了数据清洗、Tokenizer 训练、预训练 (Pretrain)、监督微调 (SFT)、LoRA、强化学习 (RLHF/RLAIF)、模型蒸馏等大模型训练的所有关键阶段的代码。

### 技术栈
*   **语言:** Python
*   **核心框架:** PyTorch
*   **主要依赖:** `transformers`, `datasets`, `streamlit`, `wandb`/`swanlab`
*   **特色:** 支持 MoE（Mixture-of-Experts）架构，实现了多种前沿的强化学习算法。

### 项目结构
*   `model/`: 存放模型定义，包括核心的 `model_minimind.py` 和 LoRA 的 `model_lora.py`。
*   `trainer/`: 包含所有训练阶段的脚本，如 `train_pretrain.py`, `train_full_sft.py` 等。
*   `dataset/`: 用于存放训练所需的数据集。
*   `scripts/`: 提供 WebUI (`web_demo.py`)、模型格式转换 (`convert_model.py`) 和 API 服务 (`serve_openai_api.py`) 等实用工具。
*   `eval_llm.py`: 用于在命令行与训练好的模型进行交互和评估。

## 2. 构建与运行

### 2.1 环境准备

首先，克隆项目并安装所需的依赖包：
```bash
git clone https://github.com/jingyaogong/minimind.git
cd minimind
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

### 2.2 数据准备

根据 `README.md` 中的指引，从 [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files) 或 [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main) 下载所需的数据集，并放置于 `./dataset/` 目录下。

为了最快速度复现一个聊天模型，推荐至少下载以下文件：
*   `pretrain_hq.jsonl`
*   `sft_mini_512.jsonl`

### 2.3 核心训练命令

所有训练命令均在 `trainer/` 目录下执行。

**a) 预训练 (Pretrain)**
此阶段让模型学习基础知识。
```bash
# cd trainer/
python train_pretrain.py
```
*   **输入:** `dataset/pretrain_hq.jsonl`
*   **输出:** 默认保存在 `out/pretrain_512.pth`

**b) 监督微调 (SFT)**
此阶段让模型学习对话格式。
```bash
# cd trainer/
# 注意：SFT 脚本默认会加载预训练好的权重
python train_full_sft.py
```
*   **输入:** `dataset/sft_mini_512.jsonl`
*   **输出:** 默认保存在 `out/full_sft_512.pth`

**c) (可选) 多卡训练**
项目支持使用 `torchrun` 进行 DDP 多卡训练。
```bash
# 例如，使用 N 张 GPU 进行预训练
# cd trainer/
torchrun --nproc_per_node N train_pretrain.py
```

### 2.4 模型评估与演示

**a) 命令行评估**
使用 `eval_llm.py` 与你训练好的模型进行对话。
```bash
# 在项目根目录执行
python eval_llm.py --weight full_sft 
```

**b) WebUI 演示**
项目提供了一个基于 Streamlit 的简单网页聊天界面。
```bash
# cd scripts/
streamlit run web_demo.py
```
然后根据提示在浏览器中打开相应地址即可。

## 3. 开发规范与约定

*   **代码风格:** 代码注释清晰，尤其是在模型定义和核心算法部分，有大量的说明性注释。
*   **参数化配置:** 所有训练脚本都通过 `argparse` 提供了丰富的命令行参数，方便调整模型结构、超参数和训练流程。
*   **模块化设计:** 项目结构清晰，模型 (`model`)、训练逻辑 (`trainer`)、数据处理 (`dataset`) 和工具 (`scripts`) 各司其职。
*   **检查点与续训:** 训练脚本内置了完善的检查点机制，支持从中断处无缝恢复训练 (`--from_resume 1`)。
*   **日志与可视化:** 支持使用 `wandb` 或 `swanlab` 对训练过程进行可视化 (`--use_wandb`)。

## 4. 关键文件解读

*   **`README.md`**: 项目的入口，提供了极其详尽的介绍、使用指南、设计理念和实验结果。
*   **`requirements.txt`**: 定义了项目的所有 Python 依赖。
*   **`model/model_minimind.py`**: 项目的灵魂，从零实现了 Transformer 模型架构，包括 MoE 模块。
*   **`trainer/train_pretrain.py`**: 预训练脚本的代表，展示了完整的训练循环。
*   **`trainer/train_dpo.py`, `train_ppo.py`**: 强化学习系列脚本，从零实现了 DPO、PPO 等先进的对齐算法。
*   **`eval_llm.py`**: 实用的模型交互脚本，演示了如何加载权重并进行文本生成。