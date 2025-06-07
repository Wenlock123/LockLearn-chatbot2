import os
import sys
import re
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import together
from dotenv import load_dotenv

# --- Patch sqlite3 สำหรับ Streamlit Cloud ---
# เพื่อแก้ปัญหา sqlite3 ไม่ compatible บน Streamlit Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- Load environment variables ---
load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY")
if not together_api_key:
    st.error("Please set the TOGETHER_API_KEY in your environment variables.")
    st.stop()

# --- Load multilingual embedding model on CPU ---
embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device="cpu"
)

# --- Load Chroma DB persistent client ---
chroma_client = chromadb.PersistentClient(path="chromadb_database_v2")
collection = chroma_client.get_or_create_collection("recommendations")

# --- Streamlit app config ---
st.set_page_config(page_title="LockLearn Coach", page_icon="🧠")
st.title("🧠 LockLearn: Life Coaching Chatbot")

# --- Initialize session state for chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Utility functions ---

def is_gibberish_or_typo(text):
    text = text.strip()
    if len(text) <= 2:
        return True
    words = text.split()
    if len(words) == 1 and not re.search(r"[a-zA-Zก-๙]", words[0]):
        return True
    return False

def is_closing_message(text):
    patterns = [
        r"\b(thank you|thanks|bye|goodbye)\b",
        r"(ขอบคุณ|บาย|ลาก่อน|แค่นี้ก่อน)"
    ]
    return any(re.search(pattern, text.lower()) for pattern in patterns)

def detect_language(text):
    thai_chars = re.findall(r"[\u0E00-\u0E7F]", text)
    # หากอักษรไทยมากกว่า 30% ของข้อความ ให้ถือเป็นภาษาไทย
    return "th" if len(thai_chars) / max(len(text), 1) > 0.3 else "en"

def retrieve_recommendations(query_embedding, top_k=10):
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    if results and "documents" in results and results["documents"]:
        return results["documents"][0]
    return []

def query_llm_with_chat(prompt, api_key):
    together.api_key = api_key
    response = together.Chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": "You are a supportive female life coach who helps users improve their lives."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()

# --- Main Chat UI ---

user_input = st.chat_input("Type your concern or question here...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    lang = detect_language(user_input)

    if is_gibberish_or_typo(user_input):
        reply = {
            "th": "😅 ดิฉันไม่แน่ใจว่าคุณหมายถึงอะไร ลองพิมพ์ใหม่อีกครั้งนะคะ",
            "en": "😅 I'm not sure what you mean. Could you try rephrasing it?"
        }[lang]
    elif is_closing_message(user_input):
        reply = {
            "th": "😊 ยินดีเสมอค่ะ หากต้องการคำแนะนำเพิ่มเติมสามารถถามได้ตลอดเลยนะคะ",
            "en": "😊 You're always welcome! Feel free to ask if you need more support!"
        }[lang]
    else:
        with st.spinner("Thinking..."):
            question_embedding = embedding_model.encode(user_input).tolist()
            recommendations = retrieve_recommendations(question_embedding, top_k=10)

            prompt = f"""
User message: "{user_input}"

Step 1: Briefly analyze the user's feelings or situation based on the message above.
Step 2: Using your analysis and the recommendations below, generate a supportive and practical response.

Recommendations:
"""
            for rec in recommendations:
                prompt += f"- {rec}\n"

            prompt += f"""

Please respond in {'Thai' if lang == 'th' else 'English'} with a {'polite and warm tone, ending sentences with "ค่ะ"' if lang == 'th' else 'kind and uplifting tone like a supportive female life coach'}.

Your response should:
- Reflect understanding of the user's feelings or situation.
- Naturally incorporate relevant recommendations.
- Avoid repeating the user's exact words or the recommendations verbatim.
- Be concise (1–2 sentences) and encouraging.
"""

            reply = query_llm_with_chat(prompt, together_api_key)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant", avatar="🧘‍♀️"):
        st.markdown(reply)

# --- Show previous chat messages ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
