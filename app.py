import os
import sys
import re
import shutil
import gdown
import zipfile
import streamlit as st
import requests

# --- Patch sqlite3 สำหรับ Streamlit Cloud ---
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# --- ตั้งค่าหน้า Streamlit ---
st.set_page_config(page_title="LockLearn Lifecoach", page_icon="💖", layout="centered")

# --- กำหนด path สำหรับฐานข้อมูล ---
folder_path = "./chromadb_database_v2"
zip_file_path = "./chromadb_database_v2.zip"

# --- ลบฐานข้อมูลเก่า (ถ้ามี) ---
if os.path.exists(folder_path):
    shutil.rmtree(folder_path)

# --- ดาวน์โหลดฐานข้อมูลจาก Google Drive ---
st.info("📦 กำลังดาวน์โหลดฐานข้อมูลคำแนะนำ (Vector DB)...")

gdrive_file_id = "13MOEZbfRTuqM9g2ZJWllwynKbItB-7Ca"
gdown.download(id=gdrive_file_id, output=zip_file_path, quiet=False, use_cookies=False)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(folder_path)
os.remove(zip_file_path)

st.success("✅ ดาวน์โหลดและแตกไฟล์ฐานข้อมูลเรียบร้อยแล้ว!")

# --- โหลด ChromaDB ---
try:
    client = PersistentClient(path=folder_path)
    collection = client.get_collection(name="recommendations")
except Exception as e:
    st.error(f"❌ โหลด ChromaDB ไม่สำเร็จ: {e}")
    st.stop()

# --- โหลด embedding model ---
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# --- โหลด API Key ---
api_key = st.secrets["TOGETHER_API_KEY"]

# --- ฟังก์ชันเรียก LLaMA 4 Scout ---
def query_llm(prompt, api_key):
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
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"❌ API Error {r.status_code}: {r.text}"
    except Exception as e:
        return f"❌ Request failed: {e}"

# --- ดึงคำแนะนำจาก ChromaDB ---
def retrieve_recommendations(embedding, top_k=10):
    try:
        result = collection.query(query_embeddings=[embedding], n_results=top_k)
        return result['documents'][0] if result and result.get('documents') else []
    except Exception:
        return []

# --- ตรวจปิดข้อความ / gibberish / ภาษา ---
def is_closing(text):
    patterns = [r"^ขอบคุณ", r"^ok", r"^เข้าใจ", r"^โอเค", r"^รับทราบ", r"^got it", r"^noted"]
    return any(re.match(p, text.strip().lower()) for p in patterns if len(text.split()) <= 5)

def is_typo(text):
    return len(text.strip()) <= 2 or (len(text.split()) == 1 and not re.search(r'[a-zA-Zก-๙]', text))

def detect_language(text):
    th_chars = re.findall(r'[\u0E00-\u0E7F]', text)
    return "th" if len(th_chars) / max(len(text), 1) > 0.3 else "en"

# --- Session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- UI ---
st.title("💖 LockLearn Lifecoach")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("พิมพ์ข้อความของคุณที่นี่...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    lang = detect_language(user_input)

    if is_typo(user_input):
        reply = "😅 ผมไม่แน่ใจว่าคุณหมายถึงอะไร ลองพิมพ์ใหม่อีกครั้งนะค่ะ" if lang == "th" else \
                "😅 I'm not sure what you mean. Could you try rephrasing it?"
    elif is_closing(user_input):
        reply = "😊 ยินดีเสมอค่ะ ถ้าต้องการคำแนะนำเพิ่มเติมสามารถถามได้ตลอดเลยนะคะ!" if lang == "th" else \
                "😊 You're always welcome! Feel free to ask anytime if you need more support!"
    else:
        with st.spinner("🧘‍♀️ กำลังคิดคำตอบที่ดีที่สุด..."):
            embedding = embedding_model.encode(user_input).tolist()
            recs = retrieve_recommendations(embedding)

            prompt = f"""
User message: "{user_input}"

Step 1: Briefly analyze the user's feelings or situation.
Step 2: Using the recommendations below, respond with a short, supportive, and empathetic answer.

Recommendations:
{chr(10).join(f"- {r}" for r in recs)}

Please respond in {'Thai' if lang == 'th' else 'English'} with a {'warm and polite female tone, ending sentences with "ค่ะ"' if lang == 'th' else 'kind and supportive female coach tone'}.
Make sure your message:
- Reflects understanding of the user.
- Naturally incorporates useful advice.
- Feels personal and encouraging.
- Is concise (1–2 sentences).
"""

            reply = query_llm(prompt, api_key)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant", avatar="🧘‍♀️"):
        st.markdown(reply)
