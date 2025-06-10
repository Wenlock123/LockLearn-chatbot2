# --- ส่วนที่ 1: Import ---
import os
import re
import shutil
import zipfile
import requests
import gdown
import streamlit as st

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# --- ส่วนที่ 2: ตั้งค่า Streamlit ---
st.set_page_config(page_title="LockLearn Lifecoach", page_icon="💖", layout="centered")

# --- ส่วนที่ 3: ดาวน์โหลดและโหลดฐานข้อมูล ---
folder_path = "./chromadb_database_v2"
zip_file_path = "./chromadb_database_v2.zip"
gdrive_file_id = "1czCTZUvq-ooRt6_-YL_hzYTYDuOdmNpB"

if os.path.exists(folder_path):
    st.info("📂 พบฐานข้อมูลอยู่แล้ว กำลังโหลด...")
else:
    st.info("📦 กำลังดาวน์โหลดฐานข้อมูลจาก Google Drive...")
    gdown.download(id=gdrive_file_id, output=zip_file_path, quiet=False, use_cookies=False)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(folder_path)
    os.remove(zip_file_path)
    st.success("✅ ดาวน์โหลดและแตกไฟล์เรียบร้อยแล้ว!")

# --- ส่วนที่ 4: โหลด ChromaDB ---
try:
    client = PersistentClient(path=folder_path)
    collection = client.get_or_create_collection(name="recommendations")
except Exception as e:
    st.error(f"❌ ไม่สามารถโหลดฐานข้อมูลได้: {e}")
    st.stop()

# --- ส่วนที่ 5: โหลด Embedding Model ---
embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# --- ส่วนที่ 6: API Key ---
api_key = st.secrets["TOGETHER_API_KEY"]

# --- ส่วนที่ 7: เรียก LLM ---
def query_llm_with_chat(prompt, api_key):
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
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=15)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"❌ API Error {res.status_code}: {res.text}"
    except Exception as e:
        return f"❌ Request failed: {e}"

# --- ส่วนที่ 8: ฟังก์ชันช่วย ---
def retrieve_recommendations(embedding, top_k=10):
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    return results.get('documents', [[]])[0]

def is_closing_message(text):
    patterns = [r"^ขอบคุณ.*", r"^โอเค.*", r"^เข้าใจ.*", r"^รับทราบ.*", r"^thank.*", r"^ok.*", r"^noted.*"]
    return any(re.match(p, text.strip().lower()) for p in patterns if len(text.split()) <= 5)

def is_gibberish_or_typo(text):
    text = text.strip()
    return len(text) <= 2 or (len(text.split()) == 1 and not re.search(r"[a-zA-Zก-๙]", text))

def detect_language(text):
    return "th" if len(re.findall(r"[\u0E00-\u0E7F]", text)) / max(len(text), 1) > 0.3 else "en"

# --- ส่วนที่ 9: Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- ส่วนที่ 10: UI ---
st.title("💖 LockLearn Lifecoach")

# แสดงประวัติสนทนา
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

# รับข้อความใหม่จากผู้ใช้
user_input = st.chat_input("How can I support you today?")

# --- ส่วนที่ 11: ประมวลผลข้อความผู้ใช้ ---
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    lang = detect_language(user_input)

    if is_gibberish_or_typo(user_input):
        reply = {
            "th": "😅 ผมไม่แน่ใจว่าคุณหมายถึงอะไร ลองพิมพ์ใหม่อีกครั้งนะครับ",
            "en": "😅 I'm not sure what you mean. Could you try rephrasing it?"
        }[lang]
    elif is_closing_message(user_input):
        reply = {
            "th": "😊 ยินดีเสมอครับ หากต้องการคำแนะนำเพิ่มเติมสามารถถามได้ตลอดเลยนะครับ!",
            "en": "😊 You're always welcome! Feel free to ask if you need more support!"
        }[lang]
    else:
        with st.spinner("Thinking..."):
            question_embedding = embedding_model.encode(user_input).tolist()
            recommendations = retrieve_recommendations(question_embedding, top_k=10)

            prompt = f"""
User message: "{user_input}"

Based on the user's message and the recommendations below, please generate a concise, supportive, and practical lifecoach response in {'Thai' if lang == 'th' else 'English'}.
Your reply should be 1-3 sentences, warm and encouraging {'and end each sentence with "ค่ะ"' if lang == 'th' else 'like a supportive female life coach'}.
Naturally incorporate relevant recommendations without repeating them verbatim.
Avoid stepwise answers; just give a unified, smooth answer.

Recommendations:
"""
            for rec in recommendations:
                prompt += f"- {rec}\n"

            prompt += "\nPlease respond accordingly."
            reply = query_llm_with_chat(prompt, api_key)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant", avatar="🧘‍♀️"):
        st.markdown(reply)
