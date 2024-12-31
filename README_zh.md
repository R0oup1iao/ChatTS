# ChatTS: 理解、聊天、推理时间序列与TS-MLLM
`ChatTS` 专注于**理解和推理**时间序列，类似于视觉/视频/音频-MLLM 所做的工作。
此仓库提供了 `ChatTS` 的代码、数据集和模型：[ChatTS: 通过合成数据对齐时间序列以增强理解和推理](https://arxiv.org/pdf/2412.03104)。

以下是一个 `ChatTS` 应用程序的示例，允许用户与 LLM 交互以理解和推理时间序列数据：
![Chat](figures/chat_example.png)

我们还提供了收集到的评估数据集。您可以从 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14349206.svg)](https://doi.org/10.5281/zenodo.14349206) 下载评估数据集。训练脚本可以在 [ChatTS-Training](https://github.com/xiezhe-24/ChatTS-Training) 中找到。
一个微调后的 `ChatTS` 模型已在 [HuggingFace](https://huggingface.co/bytedance-research/ChatTS-14B) 开源。您可以下载并尝试使用！

## 新闻
- **2024/12/30**: `vLLM` 对 `ChatTS` 的支持版本已发布！请查看 [demo_vllm.py](demo_vllm.py) 获取更多信息。（**注意**：此版本仍在开发中，可能不稳定。）我们还更新了 `ChatTS` 模型实现，现在支持 `kv_cache` 和 `AutoProcessor`。您可以在 [HuggingFace](https://huggingface.co/bytedance-research/ChatTS-14B) 找到它们。

## 介绍
此仓库提供了几种工具包，用于生成合成数据，并提供了评估代码和评估数据集以供复现：
- 生成合成时间序列数据及其对应属性的工具包：`chatts/ts_generator.py`。
- 使用预定义模板生成训练数据集的示例代码：`chatts/generate_template_qa.py`，可以进一步用作 TSEvol 的种子 QA。
- 使用 LLMs 生成训练数据集的示例代码：`chatts/generate_llm_qa`，可以进一步用作 TSEvol 的种子 QA。
- 使用生成的种子 QA 实现 `TSEvol` 的代码：`chatts/evol/evol_instruct.py`。
- 评估代码实现：`evaluation/`。
- 推理演示：`demo_hf.ipynb` 和 `demo_vllm.py`。
- 训练好的 `ChatTS` 模型：[HuggingFace](https://huggingface.co/bytedance-research/ChatTS-14B)。
- 评估数据集：[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14349206.svg)](https://doi.org/10.5281/zenodo.14349206)。
- 训练自己的模型的训练脚本：[ChatTS-Training](https://github.com/xiezhe-24/ChatTS-Training)。

## 如何使用
### 安装
- 模型推理的基本要求：`python>=3.11`，`deepspeed`，`vllm==0.6.6.post1`，`torch==2.5.1`，`flash-attn`（参考 `requirements.txt`）。
- 从 [Zenodo](https://doi.org/10.5281/zenodo.14349206) 下载评估数据集，并将其放在 `evaluation/dataset` 目录下（`evaluation/dataset/dataset_a.json` 和 `evaluation/dataset/dataset_b.json`）。
- 从 [HuggingFace](https://huggingface.co/bytedance-research/ChatTS-14B) 下载训练好的模型权重，解压后将所有文件放在 `ckpt/` 目录下（`ckpt/config.json` 等）。
- **注意**：`ChatTS` 是基于一个 14B 大小的基础模型训练的，因此需要确保您的 GPU 具有足够的内存进行推理。此外，由于模型的要求，`Flash-Attention`（https://github.com/Dao-AILab/flash-attention）是必需的，因此需要确保您的 GPU 符合 Flash-Attention 的安装要求。推荐 GPU：A100/A800。

### 尝试使用 ChatTS 模型
- 按照 `Installation` 中的步骤下载训练好的 `ChatTS` 模型并将其放在 `ckpt` 目录下。
- `ChatTS` 模型可以直接使用 `transformers` 库加载。**请参考 `demo_hf.ipynb` 获取更多信息。**
- **关于 `sp` 编码**。为了便于输入可变长度批次的时间序列，我们在编码时间序列时采用了名为 `sp` 编码的方法。对于每个时间序列数据点，添加一个额外的数值 1.0 作为掩码。为了方便，我们提供了一个 Processor，可以通过 `transformers` 中的 `AutoProcessor` 加载，用于归一化和转换时间序列和文本（值保留时间序列编码）。请参考 `demo_hf.ipynb` 了解其用法。
- 使用 `ChatTS` 的简单示例（使用 `HuggingFace`）：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch
import numpy as np

# 加载模型、分词器和处理器
model = AutoModelForCausalLM.from_pretrained("./ckpt", trust_remote_code=True, device_map=0, torch_dtype='float16')
tokenizer = AutoTokenizer.from_pretrained("./ckpt", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("./ckpt", trust_remote_code=True, tokenizer=tokenizer)
# 创建时间序列和提示
timeseries = np.sin(np.arange(256) / 10) * 5.0
timeseries[100:] -= 10.0
prompt = f"我有一个长度为 256 的时间序列：<ts><ts/>。请分析这个时间序列中的局部变化。"
# 应用 Chat 模板
prompt = f"{prompt}"
# 转换为张量
inputs = processor(text=[prompt], timeseries=[timeseries], padding=True, return_tensors="pt")
# 模型生成
outputs = model.generate(**inputs, max_new_tokens=300)
print(tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True))
```

### vLLM 推理（Beta）
由于 [vLLM](https://github.com/vllm-project/vllm) 缺乏对 `ChatTS` 模型的原生支持，我们提供了一个 [补丁](chatts/vllm/chatts_vllm.py)，使 vLLM 支持推理。因此，在使用 vLLM 加载模型之前，请确保代码包含：`import chatts.vllm.chatts_vllm` 以在 vLLM 中注册 `ChatTS` 模型。请参考以下步骤使用 vLLM 加载 `ChatTS`：

1. 安装 `vllm==0.6.6.post1`（请确保安装了确切版本，因为 vLLM 的多模态 API 变化频繁）。
2. 请参考 `demo_vllm.py` 获取详细用法。

使用 vLLM 加载 `ChatTS` 的简单示例：
```python
import chatts.vllm.chatts_vllm
from vllm import LLM, SamplingParams
# 加载模型
language_model = LLM(model="./ckpt", trust_remote_code=True, max_model_len=ctx_length, tensor_parallel_size=1, gpu_memory_utilization=0.95, limit_mm_per_prompt={"timeseries": 50})
# 创建时间序列（np.ndarray）和提示（应用 chat 模板）
ts1, ts2 = ...
prompt = ...
# 模型推理
outputs = language_model.generate([{
      "prompt": prompt,
      "multi_modal_data": {"timeseries": [ts1, ts2]}
  }], sampling_params=SamplingParams(max_tokens=300))
```

### 训练数据生成
1. **使用模板生成 QA**。使用 `python3 -m chatts.generate_template_qa` 生成带有预定义模板的训练数据集。
2. **使用 LLMs 生成 QA**。您需要一个可以使用 `vLLM` 加载的本地 LLM。在 `chatts/generate_llm_qa.py` 中设置 `[LOCAL_LLM_PATH]` 为本地 LLM 模型（例如 QWen2.5-32B-Instruct，**不是 ChatTS 模型**) 并根据需要设置 `num_gpus` 和 `gpu_per_model`。使用 `python3 -m chatts.generate_llm_qa` 生成带有 LLMs 的训练数据集。
3. **TSEvol**。您需要一个可以使用 `vLLM` 加载的本地 LLM。步骤 1 和步骤 2 生成的数据集将作为 TSEvol 的种子 QA，因此请确保在运行 TSEvol 之前成功生成了这些数据集。然后，按照 `chatts/evol/evol_instruct.py` 中的步骤：
    1. 在 `evol_instruct.py` 中设置 `[LOCAL_LLM_PATH]` 为本地 LLM 模型路径（例如 QWen2.5-32B-Instruct，**不是 ChatTS 模型**) 并根据需要在 `chatts/evol/evol_instruct.py` 中设置 `num_gpus` 和 `gpu_per_model`。
    2. 运行 `python3 -m chatts.evol.evol_instruct`。
    3. 输出将保存到 `OUTPUT_FILE` 中指定的文件。

### 使用 Deepspeed 进行评估推理
- 我们提供了一个简单的脚本 `chatts/inference_tsmllm_deepspeed.py`，用于使用 `deepspeed` 进行 `ChatTS` 推理。安装 `deepspeed` 后，请在脚本中设置 `WORKDIR`（当前目录的绝对路径）和评估数据集。然后，运行以下命令进行模型推理：
```sh
deepspeed --num_gpus [YOUR_NUM_GPUS] --master_port 12345 chatts/inference_tsmllm_deepspeed.py
```
您应该在 `exp/` 文件夹中找到推理结果，这些结果将进一步用于评估。

### 评估
- 安装 `ragas==0.1.9`（https://github.com/explodinggradients/ragas），用于评估归纳推理结果。
- 在 `evaluation/ragas/config/config.toml` 中设置 `API_KEY` 和 `OPENAI_URL`（参考 https://platform.openai.com/docs/api-reference）。
- 运行 `python3 -m evaluation.evaluate_tsmllm_models` 以评估 `ChatTS`（确保在此之前已完成模型推理）。
- 我们还提供了一个简单的演示来评估基于文本的 GPT 模型的性能。在 `evaluation/evaluate_gpt_text_models.py` 中设置您的 `API_KEY` 和 `OPENAI_URL`，然后运行命令 `python3 -m evaluation.evaluate_gpt_text_models` 以获取基于文本的 GPT 模型的评估结果。

### 微调您自己的模型
- 我们提供了一个简单的脚本用于微调您自己的 TS-MLLM 模型：https://github.com/xiezhe-24/ChatTS-Training（基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 修改）。请参考此仓库了解更多详情。

## 评估数据集
- 我们提供了两个评估数据集，如论文所述。您可以在 `evaluation/dataset` 文件夹中找到这些数据集。每个样本包含多个部分：`timeseries`，即时间序列数据本身；`question`，与时间序列相关的问题；`answer`，提供的标准答案仅作参考；`attributes`，用于评估结果的结构化标签；以及 `ability_types`，指示问题涉及的任务类型。
**请注意**：为了减少评估成本，我们将针对同一时间序列的不同问题合并成一个 `question`。我们使用编号区分这些不同的问题。因此，在查看评估数据集时，实际问题的数量可能会超过 `timeseries` 条目的数量。另一个需要注意的是，某些归纳推理和对齐任务被组合在一起。这是因为归纳推理任务通常需要解释时间序列属性的物理意义。
- `MCQ2` 数据集来自第三方且开源。但由于许可限制，我们无法在此仓库中提供它。您可以直接通过 https://github.com/behavioral-data/TSandLanguage 下载。

## 案例研究
![image](figures/case_studies.png)
在 `ChatTS` 中，我们主要关注**理解和推理**时间序列，类似于视觉/视频/音频-MLLM 所做的工作，而不是进行时间序列预测、异常检测和分类任务。
您可以通过修改 `demo_hf.ipynb` 中的时间序列和问题文本来尝试更多 `ChatTS` 的应用场景！

## 第三方依赖
=======

## 参考文献
>>>>>>> 46b5b51 (translate)
- QWen (https://github.com/QwenLM/Qwen2.5)
- DeepSpeed (https://www.deepspeed.ai/)
- RAGAS (https://github.com/explodinggradients/ragas)
- vLLM (https://github.com/vllm-project/vllm)
- Flash Attention (https://github.com/Dao-AILab/flash-attention)

<<<<<<< HEAD
## 安全
如果您发现此项目存在潜在的安全问题，或认为您可能发现了安全问题，请通过我们的 [安全中心](https://security.bytedance.com/src) 或 [漏洞报告邮箱](sec@bytedance.com) 通知字节跳动安全团队。

请**不要**为安全漏洞创建公共 GitHub 问题。
=======
## 安全性
如果您发现本项目存在潜在的安全问题，或认为可能存在安全问题，请通过我们的 [安全中心](https://security.bytedance.com/src) 或 [漏洞报告邮箱](sec@bytedance.com) 通知字节跳动安全团队。

请勿为安全漏洞创建公共 GitHub 问题。
>>>>>>> 46b5b51 (translate)

## 许可证
此项目遵循 [MIT 许可证](LICENSE)。

## 引用
```bibtex
@article{xie2024chatts,
  title={ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning},
  author={Xie, Zhe and Li, Zeyan and He, Xiao and Xu, Longlong and Wen, Xidao and Zhang, Tieying and Chen, Jianjun and Shi, Rui and Pei, Dan},
  journal={arXiv preprint arXiv:2412.03104},
  year={2024}
}
```