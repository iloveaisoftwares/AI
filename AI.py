import streamlit as st
import google.generativeai as palm
import textwrap
import numpy as np
import pandas as pd
import time
palm.configure(api_key='AIzaSyCLBwjzHA-aXhWlIOoUwqRiyeJZLq7LmXE')
models = [m for m in palm.list_models() if 'embedText' in m.supported_generation_methods]
model = models[0]
st.set_page_config(page_title="AI", page_icon="https://www.jccotp.org/app/uploads/2022/07/neuroscience_event.jpg")
st.markdown(
    f"""
    <style>
        .reportview-container {{
            background: url("https://www.jccotp.org/app/uploads/2022/07/neuroscience_event.jpg");
            background-size: cover;
        }}
    </style>
    """,
    unsafe_allow_html=True
)
st.title("AI  ")
st.write("Sophie")
# Save chat history to a local file
def save_chat_history(chat_history):
    with open(r'C:\Users\Shree\Desktop\ChatHistory.txt', 'w', encoding='utf-8') as f:
        for message in chat_history:
            f.write(f"{message}\n")
# Load chat history from a local file
def load_chat_history():
    try:
        with open(r'C:\Users\Shree\Desktop\ChatHistory.txt', 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        return []
# Load chat history from a local file when the app starts
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()
# Create a sidebar and display the chat history
st.sidebar.title("Chat History")
for message in st.session_state.chat_history:
    st.sidebar.write(message)
# Add a button to save the chat history to a local file
if st.sidebar.button("Save Chat History"):
    save_chat_history(st.session_state.chat_history)
    st.sidebar.success("Chat history saved!")
document1 = "sumit(i am a boy)"
document2 = "sum"
document3 = "S"
texts = [document1, document2, document3]
df = pd.DataFrame(texts)
df.columns = ['Text']
def embed_fn(text):
    chunk_size = 2048
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    embeddings = []
    for chunk in chunks:
        embedding = palm.generate_embeddings(model=model, text=chunk)['embedding']
        embeddings.append(embedding)
    return np.mean(embeddings, axis=0)
df['Embeddings'] = df['Text'].apply(embed_fn)
query = st.text_input("Enter your question here:", "Hi")
# Create a small temperature slider in the top right corner of the screen
col1, col2 = st.columns([4, 1])
with col2:
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
def find_best_passage(query, dataframe):
    query_embedding = palm.generate_embeddings(model=model, text=query)
    dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding['embedding'])
    idx = np.argmax(dot_products)
    return dataframe.iloc[idx]['Text']
passage = find_best_passage(query, df)
def make_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ") 
    # Include conversation history in prompt
    conversation_history_str = ""
    for message in st.session_state.chat_history:
        conversation_history_str += f"{message}\n"
    prompt = textwrap.dedent("""You are a helpful and informative bot, master personal assistant who is a multitaasker, countinuously adapting at service that answers in brief, answer the  questions in detail and simple words(in 1000 words exactly from the ncert broken down into umderstandable context as if understanding it to a kid in story like format,
                              also mention from which chapter and topic the question is from of ncert only 11th and 12th, provide image link at the end of the answer too from https://archive.org/ )
                              using text from the reference of NCERT textbooks or passage included below. \
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
                            always refer information from internet before giving a response \
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
    CONVERSATION HISTORY:
    {conversation_history}
    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'
        ANSWER:
    """).format(conversation_history=conversation_history_str, query=query, relevant_passage=escaped)
    return prompt
prompt = make_prompt(query, passage)
defaults = {
  'model': 'models/text-bison-001',
  'temperature': temperature,
  'candidate_count': 1,
  'top_k': 40,
  'top_p': 0.95,
  'max_output_tokens': 1024,
  'stop_sequences': [],
  'safety_settings': [{"category":"HARM_CATEGORY_DEROGATORY","threshold":1},{"category":"HARM_CATEGORY_TOXICITY","threshold":1},{"category":"HARM_CATEGORY_VIOLENCE","threshold":3},{"category":"HARM_CATEGORY_SEXUAL","threshold":3},{"category":"HARM_CATEGORY_MEDICAL","threshold":3},{"category":"HARM_CATEGORY_DANGEROUS","threshold":2}],
}
answer = palm.generate_text(
  **defaults,
  prompt=prompt
)
# Process user input and generate response
if query:
    response = answer.result
    st.session_state.chat_history.append(f"User: {query}")
    st.session_state.chat_history.append(f"AI: {response}")
    # Display the answer progressively
    expander = st.expander("Show more", expanded=False)
    with expander:
        container = st.empty()
        for i in range(len(response)):
            time.sleep(0.01)
            container.write(response[:i+1])
from streamlit.components.v1 import iframe
# Add this code where you want to display the webpage
url = "https://www.bing.com"  # URL of the webpage you want to display
width = 900  
height = 300 
iframe(url, width=width, height=height)


from PIL import Image

# Open the image file
image = Image.open("C:\\Users\\Shree\\Downloads\\WhatsApp Image 2024-01-25 at 11.46.11 PM.jpeg")

# Display the image
st.image(image, caption='Plz donate to keep this project alive ;) ')
