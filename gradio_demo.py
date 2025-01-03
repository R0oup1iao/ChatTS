import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, AutoProcessor, BitsAndBytesConfig
import matplotlib.pyplot as plt
import numpy as np
import os
from chatts.encoding_utils import eval_prompt_to_encoding

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained("./ckpt", trust_remote_code=True, device_map='cuda:0', quantization_config=quantization_config, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("./ckpt", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("./ckpt", trust_remote_code=True, tokenizer=tokenizer)

def generate_response(prompt, *ts_strs):
    # Convert input strings to numpy arrays
    ts_list = [np.array([float(x) for x in ts_str.split(',')]) for ts_str in ts_strs if ts_str]
    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|><|im_start|>user\n{prompt}<|im_end|><|im_start|>assistant\n"
    # Convert to tensor
    inputs = processor(text=[prompt], timeseries=ts_list, padding=True, return_tensors="pt")
    streamer = TextStreamer(tokenizer)
    device = model.device
    
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
    
    # Plot time series
    fig, ax = plt.subplots(len(ts_list), 1, figsize=(10, 6 * len(ts_list)))
    if len(ts_list) == 1:
        ax.plot(ts_list[0], label=f'TS1')
        ax.legend()
    else:
        for i, ts in enumerate(ts_list):
            ax[i].plot(ts, label=f'TS{i+1}')
            ax[i].legend()
    plt.tight_layout()
    
    return text_out, fig

# Create Gradio interface
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## 时间序列分析")
        gr.Markdown("输入提示和一个到五个时间序列，生成分析结果并可视化时间序列。")
        
        with gr.Row():
            prompt = gr.Textbox(lines=2, placeholder="请输入提示...")
        
        with gr.Row():
            num_ts = gr.Slider(1, 5, step=1, label="选择时间序列数量")
        
        with gr.Row():
            ts_inputs = [gr.Textbox(lines=1, placeholder=f"请输入时间序列{i+1}，用逗号分隔...") for i in range(5)]
        
        with gr.Row():
            submit_button = gr.Button("生成")
        
        with gr.Row():
            response = gr.Textbox(label="生成的回答")
        
        with gr.Row():
            plot = gr.Plot(label="时间序列可视化")
        
        # def update_ts_inputs(num_ts):
        #     return {ts_inputs[i](visible=(i < num_ts)) for i in range(5)}
        
        # num_ts.change(update_ts_inputs, num_ts, ts_inputs)
        
        def generate(prompt, *ts_strs):
            return generate_response(prompt, *ts_strs[:num_ts.value])
        
        submit_button.click(generate, [prompt] + ts_inputs, [response, plot])
        
        examples = [
            [   
                "我有两条时间序列。TS1 的长度为 50: <ts><ts/>；TS2 的长度为 50: <ts><ts/>。请先分析这两条时间序列的局部变化，然后判断它们是否在相近的时间点上显示出局部变化。",
                {
                    "prompt":"我有两条时间序列。TS1 的长度为 50: <ts><ts/>；TS2 的长度为 50: <ts><ts/>;。请先分析这两条时间序列的局部变化，然后判断它们是否在相近的时间点上显示出局部变化。",
                    "timeseries":[
                        "0.0,0.4991670832341408,0.9933466539753061,1.4776010333066978,1.9470917115432527,2.397127693021015,2.8232123669751767,3.221088436188455,3.586780454497614,3.916634548137417,4.207354924039483,4.456036800307177,4.660195429836131,4.817790927085965,4.9272486499423005,4.987474933020272,4.997868015207525,4.958324052262343,4.869238154390976,4.731500438437072,4.546487134128409,4.316046833244369,4.0424820190979505,3.7285260608836013,3.377315902755755",
                        "0.0,0.05,0.1,0.15000000000000002,0.2,0.25,0.30000000000000004,0.35000000000000003,0.4,0.45,0.5,0.55,0.6000000000000001,0.65,0.7000000000000001,0.75,0.8,0.8500000000000001,0.9,0.9500000000000001,1.0,1.05,1.1,1.1500000000000001,1.2000000000000002,1.25,1.3,1.35,1.4000000000000001,1.4500000000000002,1.5,1.55,1.6,1.6500000000000001,1.7000000000000002,1.75,1.8,1.85,1.9000000000000001,1.9500000000000002,12.0,2.0500000000000003,2.1,2.15,2.2,2.25,2.3000000000000003,2.35,2.4000000000000004,2.45",
                    ]
                }
            ],
            [
                "我有一条北京京通快速路的流量数据时间序列。TS1 的长度为 168: <ts><ts/>。请尽可能详细的描述TS1的特征，并给出总结。",
                {
                    "prompt":"我有一条北京京通快速路的流量数据时间序列。TS1 的长度为 168: <ts><ts/>。请尽可能详细的描述TS1的特征，并给出总结。",
                    "timeseries":[
                        "0.0,1.4933466539753062,2.9470917115432527,4.323212366975177,5.586780454497614,6.707354924039483,7.660195429836131,8.4272486499423,8.997868015207526,9.369238154390976,9.546487134128409,9.54248201909795,9.377315902755754,9.07750685910732,8.674940750779525,8.205600040299336,7.7081292828620995,7.222294489865844,6.787397783525737,6.4407105452864055,6.215987523460359,6.142121137932059,6.24198963055242,6.531544981832678,7.019176955820797,7.705378626684308,8.582726721399235,9.636177562220064,10.843666810638393,12.176989102931213,13.60292250900537,15.084552985912518,16.58274602425247,18.05770681756689,19.470566755693042,20.784932993593944,21.968339319245764,22.993540479058133,23.83959836015743,24.492716726873024,24.946791233116908,25.203652783398866,25.272994540441402,25.171985489370567,24.924585964458807,24.560592426208782,24.11444957050124,23.623877127266788,23.128366093885102,22.66760435374036,22.27989444555315,22.000626562032288,21.86086765457173,21.88612289193597,22.095318849667542,22.50004896724648,23.104111354243415,23.90335737167662,24.885857025156454,26.03237457611439,27.317135409997825,28.708853588815856,30.171979122758454,31.668115236105685,33.15754912550769,34.6008351841332,35.96036757353612,37.201879449762245,38.29580907428248,39.218478347220525,39.95303677847435,40.49013326358181,40.828288882746385,40.97395586070252,40.94126033687658,40.75143920078558,40.431993444269,40.01559178372851,39.53876826149722,39.04046574886159,38.56048341667467,38.13789006800767,37.809466588260264,37.60823960732851,37.56216483209248,37.69301254060222,38.01549966979202,38.536703097646836,39.25577751040938,40.163989102572096,41.24506376614162,42.47583588796858,43.82717188964052,45.265131691316896,46.75232179560816,48.24938604831476,49.716574644099474,51.11532882578848,52.40981810034068,53.568368687535525,54.56472625363814,55.379102588834876,55.99896450071335,56.41953347309308,56.6439761703862,56.68327819268028,56.555806114529915,56.286575267588304,55.9062524582747,55.449933378977185,54.95574345354798,54.4633178969662,54.01222158439283,53.64037172445218,53.382526219778775,53.26889797912415,53.3239504240273,53.56542220939676,54.003620039316864,54.641007771280684,55.47210818996688,56.483721366530226,57.655450898109876,58.96051704804689,60.36682436947639,61.838241249511135,63.33604036262737,64.82044260692234,66.25220297137695,67.59417511060019,68.81279225239801,69.87940539905445,70.77142547246349,71.47322386938919,71.9767555245578,72.28187964202252,72.39636530825362,72.33558176317766,72.12188567708196,71.78372986072097,71.35452894153934,70.87132722792181,70.37332186951785,69.900300190579,69.49105349489713,69.18183057893516,69.00489260670193,68.98722695894907,69.14947133146407,69.50509097526525,70.0598418795357,70.8115412984986,71.75015477060336,72.85819616084203,74.11142477777135,75.47981177338467,76.92873729852056,78.42037068699949,79.91517864490294,81.3735013182473,82.75713340620845,84.03084728590167,85.1637974265389,86.13075010340265,86.9130893868207,87.49955930053633,87.88671256196129,88.07904801445409",
                    ]
                }
                
            ]
        ]
        
        
        def show_selected_example(selected_example):
            # 根据用户选择返回相应的例子
            return selected_example

        with gr.Row():

            dropdown = gr.Dropdown(choices=examples, label="选择一个例子")
            output_text = gr.Textbox(label="所选例子")

            # 当用户在下拉菜单中做出选择时，触发更新输出文本框
            dropdown.change(fn=show_selected_example, inputs=[dropdown], outputs=output_text)

                
        return demo

# Launch the interface
demo = create_interface()
demo.launch(server_name="0.0.0.0", server_port=7860)