# ChatTS: 理解、对话和推理时间序列数据
`ChatTS` 专注于**理解和推理**时间序列数据，类似于视觉/视频/音频多模态大模型（MLLMs）的功能。
该仓库提供了 `ChatTS` 的代码、数据集和模型：[ChatTS: 通过合成数据对齐时间序列与大模型以增强理解和推理](https://arxiv.org/pdf/2412.03104)。

以下是一个 `ChatTS` 应用程序的例子，允许用户与大模型交互以理解和推理时间序列数据：
![Chat](figures/chat_example.png)

## 简介
本仓库提供了生成合成数据的工具包以及用于重现评估的代码和数据集：
- 生成合成时间序列数据及其对应属性的工具包：`chatts/ts_generator.py`。
- 使用预定义模板生成训练数据集的示例代码：`chatts/generate_template_qa.py`，可以进一步用作 TSEvol 的种子 QA。
- 使用大模型生成训练数据集的示例代码：`chatts/generate_llm_qa`，可以进一步用作 TSEvol 的种子 QA。
- 使用生成的种子 QA 实现 `TSEvol`：`chatts/evol/evol_instruct.py`。
- 评估代码实现：`evaluation/`。
- 训练好的 `ChatTS` 模型和评估数据集（更多详情请参阅下文）。
- 推理演示：`demo.ipynb`。
- 训练脚本用于训练自己的模型。

我们还提供了收集到的评估数据集。您可以从 [Zenodo](https://doi.org/10.5281/zenodo.14349206) 下载评估数据集，并将其放在 `evaluation/dataset` 目录下（如 `evaluation/dataset/dataset_a.json` 和 `evaluation/dataset/dataset_b.json`）。
一个微调后的 `ChatTS` 模型已经在 [HuggingFace](https://huggingface.co/bytedance-research/ChatTS-14B) 开源。您可以下载并试用！

## 如何使用
### 安装
- 模型推理的基本要求：`python>=3.11`，`deepspeed`，`vllm`，`flash-attn`（参考 `requirements.txt`）。
- 从 [Zenodo](https://doi.org/10.5281/zenodo.14349206) 下载评估数据集，并将其放在 `evaluation/dataset` 目录下（如 `evaluation/dataset/dataset_a.json` 和 `evaluation/dataset/dataset_b.json`）。
- 从 [HuggingFace](https://huggingface.co/bytedance-research/ChatTS-14B) 下载训练好的模型权重，解压后将所有文件放在 `ckpt/` 目录下（如 `ckpt/config.json` 等）。
- **注意：** `ChatTS` 是基于一个 14B 参数量的基础模型训练的，因此您需要确保 GPU 具有足够的内存进行推理。此外，由于模型的要求，`Flash-Attention` 是必需的，因此您需要确保您的 GPU 符合 Flash-Attention 的安装要求。推荐 GPU：A100/A800。

### 尝试 ChatTS 模型
- 按照 `Installation` 中的步骤下载训练好的 `ChatTS` 模型并将其放在 `ckpt` 目录下。
- `ChatTS` 模型可以直接使用 `transformers` 库加载。但是，由于输入为时间序列数据，API 使用方式与标准实现不同。**请参考 `demo.ipynb` 获取更多信息。**
- **关于 `sp` 编码。** 为了便于处理变长批次的时间序列，我们在编码时间序列时采用了名为 `sp` 编码的方法。对于每个时间序列数据点，添加一个额外的数值 1.0 作为掩码。为了方便起见，我们提供了一系列函数来规范化和转换时间序列和文本（值保留时间序列编码）。请参考 `demo.ipynb` 获取更多信息。
- `ChatTS` 示例用法：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from chatts.encoding_utils import eval_prompt_to_encoding

# 加载模型
model = AutoModelForCausalLM.from_pretrained("./ckpt", trust_remote_code=True, device_map=0, torch_dtype='float16')
tokenizer = AutoTokenizer.from_pretrained("./ckpt", trust_remote_code=True)

# 创建时间序列和提示
timeseries = np.sin(np.arange(256) / 10) * 5.0
timeseries[100:] -= 10.0
prompt = f"我有一个长度为 256 的时间序列：<ts><ts/>。请分析这个时间序列的局部变化。"
prompt, timeseries = eval_prompt_to_encoding(prompt, [timeseries], 'sp')

# 分词并转换为张量
inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(device=0)
timeseries = torch.tensor(timeseries, dtype=torch.float16, device=0)

# 模型生成
outputs = model.generate(**inputs, timeseries=timeseries, max_new_tokens=300)
print(tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True))
```

### 训练数据生成
1. **使用模板生成 QA 数据集**。使用 `python3 -m chatts.generate_template_qa` 生成带有预定义模板的训练数据集。
2. **使用大模型生成 QA 数据集**。您需要一个可以使用 `vLLM` 加载的本地大模型来执行此步骤。在 `chatts/generate_llm_qa.py` 中设置 `[LOCAL_LLM_PATH]` 为本地大模型路径（例如 QWen2.5-32B-Instruct，**不是 ChatTS 模型**) 并设置 `num_gpus` 和 `gpu_per_model`。使用 `python3 -m chatts.generate_llm_qa` 生成带有大模型的训练数据集。
3. **TSEvol**。您需要一个可以使用 `vLLM` 加载的本地大模型来执行此步骤。步骤 1 和步骤 2 生成的数据集将作为 TSEvol 的种子 QA，因此请确保已成功生成这些数据集后再运行 TSEvol。然后，按照 `chatts/evol/evol_instruct.py` 中的步骤：
    1. 在 `evol_instruct.py` 中设置 `[LOCAL_LLM_PATH]` 为本地大模型路径（例如 QWen2.5-32B-Instruct，**不是 ChatTS 模型**) 并设置 `num_gpus` 和 `gpu_per_model`。
    2. 运行 `python3 -m chatts.evol.evol_instruct`。
    3. 输出将保存到 `OUTPUT_FILE` 中指定的文件。

### 使用 Deepspeed 进行模型推理和评估
- 我们提供了一个简单的脚本 `chatts/inference_tsmllm_deepspeed.py`，用于使用 `deepspeed` 进行 `ChatTS` 推理。安装 `deepspeed` 后，请设置 `WORKDIR`（当前目录的绝对路径）和评估数据集。然后，运行以下命令进行模型推理：
```sh
deepspeed --num_gpus [YOUR_NUM_GPUS] --master_port 12345 chatts/inference_tsmllm_deepspeed.py
```
您应该可以在 `exp/` 文件夹中找到推理结果，这些结果将用于后续评估。

### 评估
- 安装 `ragas==0.1.9` (https://github.com/explodinggradients/ragas)，用于评估归纳推理结果。
- 设置 `evaluation/ragas/config/config.toml` 中的 `API_KEY` 和 `OPENAI_URL`（参考 https://platform.openai.com/docs/api-reference）。
- 运行 `python3 -m evaluation.evaluate_tsmllm_models` 以评估 `ChatTS`（确保已完成模型推理）。
- 我们还提供了一个简单的演示来评估基于文本的 GPT 模型性能。设置 `evaluation/evaluate_gpt_text_models.py` 中的 `API_KEY` 和 `OPENAI_URL` 后，运行命令 `python3 -m evaluation.evaluate_gpt_text_models` 以获取基于文本的 GPT 模型的评估结果。

### 微调自己的模型
- 我们提供了一个简单的脚本用于微调自己的 TS-MLLM 模型：https://github.com/xiezhe-24/ChatTS-Training（基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 修改）。请参考此仓库获取更多详情。

## 评估数据集
- 我们提供了两个评估数据集，如论文所述。您可以在 `evaluation/dataset` 文件夹中找到这些数据集。每个样本包含多个部分：`timeseries`（时间序列数据本身）、`question`（与时间序列相关的查询）、`answer`（提供的标准答案，仅作参考）、`attributes`（用于评估结果的结构化标签）和 `ability_types`（表示问题涉及的任务类型）。
**请注意：** 为了减少评估成本，我们将针对同一时间序列的不同问题合并成一个 `question`。我们使用编号来区分这些问题。因此，在查看评估数据集时，实际问题的数量可能比 `timeseries` 条目数量多。另一点需要注意的是，某些归纳推理和对齐任务被组合在一起。这是因为归纳推理任务通常需要解释时间序列属性的物理意义。
- `MCQ2` 数据集来自第三方且开源。但由于许可限制，我们无法在此仓库中提供。您可以直接从 https://github.com/behavioral-data/TSandLanguage 下载。

## 案例研究
![image](figures/case_studies.png)
在 `ChatTS` 中，我们主要关注**理解和推理**时间序列数据，而不是进行时间序列预测、异常检测和分类任务。
您可以通过修改 `demo.ipynb` 中的时间序列和问题文本来尝试更多 `ChatTS` 的应用场景！

## 注意事项
- 您可以使用 CPU 进行推理。然而，由于我们当前的 `ChatTS` 模型尚未实现 `kv_cache`（我们计划尽快实现），推理速度可能会显著较慢。
- 当前不支持 `vLLM` 推理。您可以使用 `deepspeed` 进行推理。

## 第三方依赖
- QWen (https://github.com/QwenLM/Qwen2.5)
- DeepSpeed (https://www.deepspeed.ai/)
- RAGAS (https://github.com/explodinggradients/ragas)
- VLLM (https://github.com/vllm-project/vllm)
- Flash Attention (https://github.com/Dao-AILab/flash-attention)

## 安全
如果您发现该项目存在潜在的安全问题，或认为可能存在安全问题，请通过我们的 [安全中心](https://security.bytedance.com/src) 或 [漏洞报告邮箱](sec@bytedance.com) 通知字节跳动安全团队。

请**不要**为安全漏洞创建公共 GitHub 问题。

## 许可证
本项目采用 [MIT License](LICENSE) 许可证。

## 引用
```bibtex
@article{xie2024chatts,
  title={ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning},
  author={Xie, Zhe and Li, Zeyan and He, Xiao and Xu, Longlong and Wen, Xidao and Zhang, Tieying and Chen, Jianjun and Shi, Rui and Pei, Dan},
  journal={arXiv preprint arXiv:2412.03104},
  year={2024}
}
```