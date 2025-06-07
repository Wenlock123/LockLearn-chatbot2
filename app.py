import os
import zipfile
import shutil
import gdown
import streamlit as st
import requests
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# ตั้งค่าหน้า Streamlit
st.set_page_config(page_title="LockLearn Lifecoach", page_icon="💖", layout="centered")

# โหลด embedding model
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# API Key
api_key = st.secrets["TOGETHER_API_KEY"]

# โหลดฐานข้อมูลจาก zip (ถ้ายังไม่มี)
db_path = "chromadb_database_v2"
if not os.path.exists(db_path):
    st.info("📦 กำลังดาวน์โหลดฐานข้อมูล Vector DB...")
    gdown.download(id="13MOEZbfRTuqM9g2ZJWllwynKbItB-7Ca", output="db.zip", quiet=False)
    with zipfile.ZipFile("db.zip", "r") as zip_ref:
        zip_ref.extractall(db_path)
    os.remove("db.zip")
    st.success("✅ โหลด Vector DB สำเร็จ!")

# เรียก ChromaDB
client = PersistentClient(path=db_path)
try:
    collection = client.get_collection("recommendations")
except:
    collection = client.create_collection("recommendations")

# ตรวจ gibberish หรือปิดบทสนทนา
import re
def is_gibberish(text):
    return len(text.strip()) < 3 or re.fullmatch(r"[^\wก-๙]+", text.strip())

def is_closing(text):
    return re.match(r"^(ขอบคุณ|ok|โอเค|got it|thank).*", text.strip().lower())

def detect_language(text):
    return "th" if len(re.findall(r"[\u0E00-\u0E7F]", text)) > 5 else "en"

# LLM เรียก Together AI
def query_llm(prompt):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=20)
    return response.json()["choices"][0]["message"]["content"].strip()

# เรียก context จาก Vector DB
def retrieve_context(user_input):
    embedding = embedding_model.encode(user_input).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=10)
    return results["documents"][0] if results["documents"] else []

# Session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat UI
st.title("💖 LockLearn Lifecoach")
for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("อยากให้ช่วยอะไรดีคะ?")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if is_gibberish(user_input):
        response = "ขอโทษค่ะ ไม่แน่ใจว่าหมายถึงอะไร ลองพิมพ์ใหม่ได้นะคะ 😊"
    elif is_closing(user_input):
        response = "ยินดีเสมอค่ะ ขอให้เป็นวันที่ดีนะคะ ☀️"
    else:
        lang = detect_language(user_input)
        with st.spinner("กำลังคิดคำตอบที่ดีที่สุด..."):
            docs = retrieve_context(user_input)
            prompt = f"""
User message: "{user_input}"

Step 1: Analyze the user's feelings.
Step 2: Use the recommendations below to craft a kind, supportive response.

Recommendations:
""" + "\n".join(f"- {d}" for d in docs) + f"""

Respond in {'Thai' if lang == 'th' else 'English'} as a warm female life coach.
End all sentences with {"'ค่ะ'" if lang == 'th' else 'kind tone'}.
"""

            response = query_llm(prompt)

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant", avatar="🧘‍♀️"):
        st.markdown(response)
