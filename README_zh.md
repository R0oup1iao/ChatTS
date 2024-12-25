# ChatTS: 理解、对话和推理时间序列

`ChatTS` 专注于**理解和推理**时间序列，类似于视觉/视频/音频多模态大模型（MLLMs）的功能。此仓库提供了 `ChatTS` 的代码、数据集和模型：[ChatTS: 通过合成数据对齐时间序列与大模型以增强理解和推理](https://arxiv.org/pdf/2412.03104)。

以下是一个 `ChatTS` 应用程序的示例，用户可以通过与大模型交互来理解和推理时间序列数据：
![对话](figures/chat_example.png)

## 介绍
本仓库提供了生成合成数据的工具包，以及用于重现评估的代码和数据集：
- 生成合成时间序列数据及其属性的工具包：`chatts/ts_generator.py`。
- 使用预定义模板生成训练数据集的示例代码：`chatts/generate_template_qa.py`，可以进一步用作 TSEvol 的种子 QA。
- 使用大模型生成训练数据集的示例代码：`chatts/generate_llm_qa`，可以进一步用作 TSEvol 的种子 QA。
- 使用生成的种子 QA 实现 `TSEvol`：`chatts/evol/evol_instruct.py`。
- 评估代码实现：`evaluation/`。
- 训练好的 `ChatTS` 模型和评估数据集（更多详情请参阅下方部分）。
- 推理演示：`demo.ipynb`。
- 训练脚本用于训练自己的模型。

我们还提供了收集到的评估数据集。您可以从 [Zenodo](https://doi.org/10.5281/zenodo.14349206) 下载评估数据集，并将其放在 `evaluation/dataset` 目录下（例如 `evaluation/dataset/dataset_a.json` 和 `evaluation/dataset/dataset_b.json`）。训练脚本可以在 [ChatTS-Training](https://github.com/xiezhe-24/ChatTS-Training) 中找到。
一个微调的 `ChatTS` 模型已经在 [HuggingFace](https://huggingface.co/bytedance-research/ChatTS-14B) 开源。您可以下载并试用！

## 如何使用
### 安装
- 模型推理的基本要求：`python>=3.11`，`deepspeed`，`vllm`，`flash-attn`（参考 `requirements.txt`）。
- 从 [Zenodo](https://doi.org/10.5281/zenodo.14349206) 下载评估数据集，并将其放在 `evaluation/dataset` 目录下（例如 `evaluation/dataset/dataset_a.json` 和 `evaluation/dataset/dataset_b.json`）。
- 从 [HuggingFace](https://huggingface.co/bytedance-research/ChatTS-14B) 下载训练好的模型权重，解压后将所有文件放在 `ckpt/` 目录下（例如 `ckpt/config.json` 等）。
- **注意：** `ChatTS` 是基于一个 14B 大小的基础模型训练的，因此您需要确保有足够显存的 GPU 进行推理。此外，由于模型的要求，`Flash-Attention` (https://github.com/Dao-AILab/flash-attention) 是必需的，因此您需要确保您的 GPU 满足 Flash-Attention 的安装要求。推荐 GPU：A100/A800。

### 训练数据生成
1. **使用模板生成 QA**。使用 `python3 -m chatts.generate_template_qa` 生成带有预定义模板的训练数据集。
2. **使用大模型生成 QA**。您需要一个可以使用 `vLLM` 加载的本地大模型来执行此步骤。在 `chatts/generate_llm_qa.py` 中设置 `[LOCAL_LLM_PATH]` 为本地大模型路径（例如 QWen2.5-32B-Instruct，**不是 ChatTS 模型**) 并设置 `num_gpus` 和 `gpu_per_model`。使用 `python3 -m chatts.generate_llm_qa` 生成带有大模型的训练数据集。
3. **TSEvol**。您需要一个可以使用 `vLLM` 加载的本地大模型来执行此步骤。步骤 1 和步骤 2 生成的数据集将作为 TSEvol 的种子 QA，因此请确保已成功生成这些数据集后再运行 TSEvol。然后，按照 `chatts/evol/evol_instruct.py` 中的步骤：
    1. 在 `evol_instruct.py` 中设置 `[LOCAL_LLM_PATH]` 为本地大模型路径（例如 QWen2.5-32B-Instruct，**不是 ChatTS 模型**) 并设置 `num_gpus` 和 `gpu_per_model`。
    2. 运行 `python3 -m chatts.evol.evol_instruct`。
    3. 输出将保存到 `OUTPUT_FILE` 中指定的文件。

### 尝试 ChatTS 模型
- 按照 `安装` 部分的步骤下载训练好的 `ChatTS` 模型并将其放在 `ckpt` 目录下。
- `ChatTS` 模型可以直接使用 `transformers` 库加载。但是，由于输入是时间序列数据，API 使用方式与标准实现不同。**请参考 `demo.ipynb` 获取更多信息。**
- **关于 `sp` 编码。** 为了便于处理可变长度批次的时间序列输入，我们在编码时间序列时采用了名为 `sp` 编码的方法。对于每个时间序列数据点，添加一个额外的数值 1.0 作为掩码。为了方便起见，我们提供了一系列函数来规范化和转换时间序列和文本（值保留的时间序列编码）。请参考 `demo.ipynb` 获取其使用方法的更多信息。

### 使用 Deepspeed 进行模型推理和评估
- 我们提供了一个简单的脚本 `chatts/inference_tsmllm_deepspeed.py` 用于使用 `deepspeed` 进行 `ChatTS` 推理。安装 `deepspeed` 后，请设置 `WORKDIR`（当前目录的绝对路径）和评估数据集。然后，运行以下命令进行模型推理：
```sh
deepspeed --num_gpus [YOUR_NUM_GPUS] --master_port 12345 chatts/inference_tsmllm_deepspeed.py
```

### 评估
- 安装 `ragas==0.1.9` (https://github.com/explodinggradients/ragas)，用于评估归纳推理结果。
- 在 `evaluation/ragas/config/config.toml` 中设置 `API_KEY` 和 `OPENAI_URL`（参考 https://platform.openai.com/docs/api-reference）。
- 运行 `python3 -m evaluation.evaluate_tsmllm_models` 评估 `ChatTS`（确保在评估前已完成模型推理）。
- 我们还提供了一个简单的演示来评估基于文本的 GPT 模型性能。设置 `API_KEY` 和 `OPENAI_URL` 后，运行命令 `python3 -m evaluation.evaluate_gpt_text_models` 获取基于文本的 GPT 模型的评估结果。

### 微调自己的模型
- 我们提供了一个简单的脚本用于微调自己的 TS-MLLM 模型：https://github.com/xiezhe-24/ChatTS-Training（基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 修改）。请参考该仓库获取更多详情。

## 评估数据集
- 我们提供的两个评估数据集如论文中所述，可以在 `evaluation/dataset` 文件夹中找到。每个样本包含多个部分：`timeseries`，即时间序列数据本身；`question`，与时间序列相关的查询；`answer`，提供的标准答案（仅供参考）；`attributes`，用于评估结果的结构化标签；`ability_types`，指示问题涉及的任务类型。
**请注意：** 为了减少评估成本，我们将针对同一时间序列的不同问题合并为一个 `question`。我们使用编号来区分这些问题。因此，在查看评估数据集时，实际问题数量可能多于 `timeseries` 条目数量。另一个需要注意的是，某些归纳推理和对齐任务被组合在一个问题中。这是因为归纳推理任务通常需要解释时间序列属性的物理意义。
- `MCQ2` 数据集来自第三方且开源。但由于许可限制，我们无法在此仓库中提供。您可以通过 https://github.com/behavioral-data/TSandLanguage 直接下载。

## 案例研究
![案例研究](figures/case_studies.png)
在 `ChatTS` 中，我们主要关注**理解和推理**时间序列，而不是进行时间序列预测、异常检测和分类任务。
您可以通过修改 `demo.ipynb` 中的时间序列和问题文本来尝试更多 `ChatTS` 的应用场景！

## 注意事项
- 您可以使用 CPU 进行推理。然而，由于我们当前的 `ChatTS` 模型尚未实现 `kv_cache`（我们计划尽快实现），推理速度可能会显著变慢。

## 第三方依赖
- QWen (https://github.com/QwenLM/Qwen2.5)
- DeepSpeed (https://www.deepspeed.ai/)
- RAGAS (https://github.com/explodinggradients/ragas)
- VLLM (https://github.com/vllm-project/vllm)
- Flash Attention (https://github.com/Dao-AILab/flash-attention)

## 安全性
如果您发现项目中存在潜在的安全问题，或认为可能存在安全问题，请通过我们的 [安全中心](https://security.bytedance.com/src) 或 [漏洞报告邮箱](sec@bytedance.com) 通知字节跳动安全团队。

请**不要**为安全漏洞创建公共 GitHub 问题。

## 许可证
本项目采用 [MIT 许可证](LICENSE)。

## 引用
```bibtex
@article{xie2024chatts,
  title={ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning},
  author={Xie, Zhe and Li, Zeyan and He, Xiao and Xu, Longlong and Wen, Xidao and Zhang, Tieying and Chen, Jianjun and Shi, Rui and Pei, Dan},
  journal={arXiv preprint arXiv:2412.03104},
  year={2024}
}
```