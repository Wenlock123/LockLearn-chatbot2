import os
import sys
import re
import shutil
import gdown
import zipfile
import streamlit as st
import requests

# Patch sqlite3 for Streamlit Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# ตั้งค่า Streamlit
st.set_page_config(page_title="LockLearn Lifecoach", page_icon="💖", layout="centered")

# Path สำหรับฐานข้อมูล
folder_path = "./chromadb_database_v2"
zip_file_path = "./chromadb_database_v2.zip"

# ลบฐานข้อมูลเก่า
if os.path.exists(folder_path):
    shutil.rmtree(folder_path)

# ดาวน์โหลดไฟล์ zip จาก Google Drive
st.info("📦 กำลังดาวน์โหลดฐานข้อมูลคำแนะนำ (Vector DB) จาก Google Drive...")
gdrive_file_id = "1czCTZUvq-ooRt6_-YL_hzYTYDuOdmNpB"
gdown.download(id=gdrive_file_id, output=zip_file_path, quiet=False, use_cookies=False)

# แตก zip
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(folder_path)
os.remove(zip_file_path)

st.success("✅ ดาวน์โหลดและแตกไฟล์ฐานข้อมูลเรียบร้อยแล้ว!")

# โหลด ChromaDB
try:
    client = PersistentClient(path=folder_path)
    collections = client.list_collections()
    collection_names = [c.name for c in collections]
    if "recommendations" in collection_names:
        collection = client.get_collection(name="recommendations")
    else:
        st.warning("⚠️ ไม่พบ collection 'recommendations' — ระบบจะสร้าง collection เปล่าใหม่แทน")
        collection = client.create_collection(name="recommendations")
except Exception as e:
    st.error(f"❌ ไม่สามารถโหลด ChromaDB ได้: {e}")
    st.stop()

# โหลด embedding model
try:
    embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device='cpu')
    st.info("✅ โหลด embedding model paraphrase-multilingual-mpnet-base-v2 สำเร็จแล้ว (CPU mode)")
except Exception as e:
    st.error(f"❌ โหลด embedding model ล้มเหลว: {e}")
    st.stop()

# โหลด API Key
api_key = st.secrets["TOGETHER_API_KEY"]

# เรียก LLM
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
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"❌ API Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"❌ Request failed: {e}"

# ค้นหาคำแนะนำ
def retrieve_recommendations(question_embedding, top_k=10):
    results = collection.query(query_embeddings=[question_embedding], n_results=top_k)
    return results['documents'][0] if results and results.get('documents') else []

# ตรวจ closing message
def is_closing_message(text):
    patterns = [r"^ขอบคุณ.*", r"^โอเค.*", r"^เข้าใจ.*", r"^รับทราบ.*", r"^thank.*", r"^ok.*", r"^noted.*"]
    return any(re.match(p, text.strip().lower()) for p in patterns if len(text.split()) <= 5)

# ตรวจ gibberish/typo
def is_gibberish_or_typo(text):
    text = text.strip()
    return len(text) <= 2 or (len(text.split()) == 1 and not re.search(r'[a-zA-Zก-๙]', text))

# ตรวจภาษา
def detect_language(text):
    return "th" if len(re.findall(r'[\u0E00-\u0E7F]', text)) / max(len(text), 1) > 0.3 else "en"

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# UI
st.title("💖 LockLearn Lifecoach")

for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

user_input = st.chat_input("How can I support you today?")

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
            try:
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

                prompt += """
Please generate a supportive, practical, and encouraging response based on the suggestions above.
Respond in the same language as the user's question:
- Thai if the question is in Thai.
- English if the question is in English.

If the user's question is in Thai, respond in a polite and feminine tone using "ค่ะ" at the end of the sentence, as if you are a female life coach giving kind, warm, and caring motivation.
If the user's question is in English, keep a kind and uplifting tone like a supportive female life coach.

Keep the response concise (1–2 sentences), natural, and motivating.
"""

                reply = query_llm_with_chat(prompt, api_key)
            except Exception as e:
                reply = f"❌ เกิดข้อผิดพลาดระหว่างประมวลผล: {e}"

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant", avatar="🧘‍♀️"):
        st.markdown(reply)
