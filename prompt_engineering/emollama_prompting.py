import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import LlamaTokenizer, LlamaForCausalLM
import pandas as pd
from constants import *
import torch

def classify_emotion(text):

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).to(int)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():  
        output_ids = model.generate(
            input_ids, 
            max_length=100, 
            attention_mask=attention_mask,
            do_sample=False,  
            pad_token_id=tokenizer.eos_token_id
        )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = output_text.replace(prompt, "").strip()
        
        return response
    
if __name__ == "__main__":
    
    tokenizer = LlamaTokenizer.from_pretrained(EMO_LLAMA_MODEL_PATH)
    model = LlamaForCausalLM.from_pretrained(EMO_LLAMA_MODEL_PATH, device_map="auto")

    df = pd.read_csv(NRC_EN_TR_PATH)
    results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for _, row in df.iterrows():
        word = str(row['word_sr_final'])
        pos = str(row['pos_sr'])
    
        prompt=f"""
        Task: Categorize the text's emotional tone as either 'neutral' or identify the presence of one or more of the given emotions (anger, anticipation, disgust, fear, joy, sadness, surprise, trust).
        Text: {word}
        This text contains emotions:
        """
        
        output_text = classify_emotion(prompt)
        print("Word: ", word, " PoS: ", pos)
        print("Model Output:", output_text)