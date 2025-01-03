USE_Q8_QUANTIZATION = False

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, AutoProcessor
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

if USE_Q8_QUANTIZATION:
    from transformers import BitsAndBytesConfig 

from chatts.encoding_utils import eval_prompt_to_encoding

# Set Environment and Load model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if USE_Q8_QUANTIZATION:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained("./ckpt", trust_remote_code=True, device_map='cuda:0', quantization_config=quantization_config, torch_dtype="auto")
else:
    model = AutoModelForCausalLM.from_pretrained("/home/acme/hfd_ckpt/Qwen2.5-Math-1.5B-Instruct/checkpoint-1000", trust_remote_code=True, device_map='cuda:0', torch_dtype='float16')
tokenizer = AutoTokenizer.from_pretrained("./ckpt", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("./ckpt", trust_remote_code=True, tokenizer=tokenizer)

# Generate a time series example
SEQ_LEN = 256
x = np.arange(SEQ_LEN)
# TS1: A simple sin signal with a sudden decrease
ts1 = np.sin(x / 10) * 5.0
ts1[100:] -= 10.0
# TS2: A increasing trend with a upward spike
ts2 = x * 0.05
ts2[103] += 10.0

# Convert time series to encoding
# prompt = f"I have 2 time series. TS1 is of length {SEQ_LEN}: <ts><ts/>; TS2 is of length {SEQ_LEN}: <ts><ts/>. Please analyze the local changes in these time series first and then conclude if these time series showing local changes near the same time? 结果用中文输出"
# prompt = f"我有两条时间序列。TS1 的长度为 {SEQ_LEN}: <ts><ts/>；TS2 的长度为 {SEQ_LEN}: <ts><ts/>。请先分析这两条时间序列的局部变化，然后判断它们是否在相近的时间点上显示出局部变化。 结果用中文输出"
prompt = f"我有两条时间序列。TS1 的长度为 {SEQ_LEN}: <ts><ts/>；TS2 的长度为 {SEQ_LEN}: <ts><ts/>。请尽可能详细的描述TS2的特征，并给出总结。"

# Apply Chat Template
prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{prompt}<|im_end|><|im_start|>assistant\n"

# Convert to tensor
inputs = processor(text=[prompt], timeseries=[ts1, ts2], padding=True, return_tensors="pt")
streamer = TextStreamer(tokenizer)
device = model.device
print(f"Model is on device: {device}")

# 将输入张量移动到与模型相同的设备，并打印每个张量的设备信息
for key, value in inputs.items():
    if isinstance(value, torch.Tensor):
        inputs[key] = value.to(device)
        print(f"Moved {key} to device: {device}")
# Input into model
print('Generating...')
outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                streamer=streamer
            )

# Show output
input_len = inputs['attention_mask'][0].sum().item()
output = outputs[0][input_len:]
text_out = tokenizer.decode(output, skip_special_tokens=True)
print('--------> Generated Answer')
print(text_out)