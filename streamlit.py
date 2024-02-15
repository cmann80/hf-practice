import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

torch.cuda.empty_cache()

def generation_function(prompt: str) -> str:
    
    tokenizer = AutoTokenizer.from_pretrained("mobiuslabsgmbh/aanaphi2-v0.1")
    model = AutoModelForCausalLM.from_pretrained("mobiuslabsgmbh/aanaphi2-v0.1")
    
    device = "cuda"
    
    model = model.half()
    
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    model.to(device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=200, do_sample=True)
    
    return tokenizer.batch_decode(generated_ids)[0]

# Streamlit app layout
st.title("Language model test platform")

# User input
user_input = st.text_input("Prompt:")

if user_input:
    # Call the generation function
    result = generation_function(user_input)
    
    # Display the result
    st.text_area("Completion:", value=result, height=400, max_chars=None, key=None)
