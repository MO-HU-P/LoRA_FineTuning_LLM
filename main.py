import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import json
import numpy as np
import urllib.parse
import io
import wave

model_name = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"
peft_name = "lora-rinna"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto", 
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

model = PeftModel.from_pretrained(
    model,
    peft_name,
    device_map="auto"
)

model.eval()

def generate_prompt(data_point):
    if data_point["input"]:
        result = f"""### 指示:
        {data_point["instruction"]}

        ### 入力:
        {data_point["input"]}

        ### 回答:
        """
    else:
        result = f"""### 指示:
        {data_point["instruction"]}

        ### 回答:
        """

    result = result.replace('\n', '<NL>')
    return result

def generate(instruction, input=None, maxTokens=256) -> str:
    prompt = generate_prompt({'instruction': instruction, 'input': input})
    input_ids = tokenizer(prompt,
                          return_tensors="pt",
                          truncation=True,
                          add_special_tokens=False).input_ids.cuda()
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=maxTokens,
        do_sample=True,
        temperature=0.9,
        top_p=0.75,
        top_k=40,
        no_repeat_ngram_size=2,
    )

    generated_text = tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True)
    generated_text = generated_text.replace("<NL>", "\n")  
    
    return generated_text


def tts_vvox(text):
    encoded_text = urllib.parse.quote(text)
    response = requests.post("http://localhost:50021/audio_query?text=" + encoded_text + "&speaker=3")
    synthesis = requests.post("http://localhost:50021/synthesis?speaker=3", json=response.json())
    voice = synthesis.content
    wav_data = np.frombuffer(voice, dtype=np.int16)

    return wav_data

def generate_output(query, use_tts):
    generated_text = generate(query)
    
    if use_tts:
        wav_data = tts_vvox(generated_text)
        wav_file = io.BytesIO()
        with wave.open(wav_file, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(wav_data.tobytes())
        return generated_text, wav_file.getvalue()
    else:
        return generated_text, None


with gr.Blocks() as demo:
    
    gr.Interface(
        fn=generate_output,
        inputs=["text", gr.Checkbox(label="Use Text-to-Speech")],
        outputs=["text", gr.Audio(type="filepath")],
        title="ChatBot: Rinna & LoRA Model",
        theme=gr.themes.Soft(), 
        allow_flagging=False
    ),

    gr.Markdown("Text-to-SpeechにはVOICEVOXずんだもんを使用しております。")


demo.launch(share=True)
