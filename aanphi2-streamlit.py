from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st

# Placeholder for chat history
chat_history = []

device = "cpu"
tokenizer = AutoTokenizer.from_pretrained("mobiuslabsgmbh/aanaphi2-v0.1")
model = AutoModelForCausalLM.from_pretrained("mobiuslabsgmbh/aanaphi2-v0.1")



# Text input for user message
user_message = st.text_input("You:", key="user_message")

def generate_response(prompt):

    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    model.to(device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=50, do_sample=True)
    return tokenizer.batch_decode(generated_ids)[0]

# Streamlit UI
st.title('Chatbot Interface')

# Button to send message
if st.button('Send'):
    if user_message:
        # Add user message to chat history
        chat_history.append(f"You: {user_message}")
        # Get chatbot response and add to chat history
        chatbot_response = generate_response(user_message)
        chat_history.append(f"Chatbot: {chatbot_response}")
        # Clear input
        st.rerun()

# Display chat history
for message in chat_history:
    st.text(message)