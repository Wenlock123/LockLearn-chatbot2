import os
import sys
import re
import shutil
import gdown
import zipfile
import streamlit as st
import requests

# --- Patch sqlite3 สำหรับ Streamlit Cloud ด้วย pysqlite3 เวอร์ชันใหม่ ---
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# --- ตั้งค่าหน้า Streamlit ---
st.set_page_config(page_title="LockLearn Lifecoach", page_icon="💖", layout="centered")

# --- กำหนด path สำหรับฐานข้อมูล ---
folder_path = "./chromadb_database_v2"
zip_file_path = "./chromadb_database_v2.zip"

# --- ฟังก์ชันลบฐานข้อมูลเก่า ---
def clear_old_database(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        st.warning("🗑️ ลบฐานข้อมูลเก่าเรียบร้อยแล้ว")

# --- ฟังก์ชันดาวน์โหลดและแตก zip ---
def download_and_extract():
    st.info("📦 กำลังดาวน์โหลดฐานข้อมูลคำแนะนำ (Vector DB) จาก Google Drive...")
    gdrive_file_id = "13MOEZbfRTuqM9g2ZJWllwynKbItB-7Ca"  # เปลี่ยนตามไฟล์จริงของคุณ
    gdown.download(id=gdrive_file_id, output=zip_file_path, quiet=False, use_cookies=False)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(folder_path)
    os.remove(zip_file_path)
    st.success("✅ ดาวน์โหลดและแตกไฟล์ฐานข้อมูลเรียบร้อยแล้ว!")

# --- ฟังก์ชันสร้างฐานข้อมูลใหม่ (empty) ---
def create_empty_collection(client):
    try:
        collection = client.create_collection(name="recommendations")
        st.info("🆕 สร้างฐานข้อมูลใหม่เรียบร้อยแล้ว")
        return collection
    except Exception as e:
        st.error(f"❌ สร้าง collection ไม่ได้: {e}")
        st.stop()

# --- โหลด embedding model ---
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# --- โหลด API Key จาก secrets.toml ---
api_key = st.secrets["TOGETHER_API_KEY"]

# --- ฟังก์ชันเรียก LLaMA 4 Scout ผ่าน Together AI ---
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

# --- ฟังก์ชันดึงคำแนะนำจาก ChromaDB ---
def retrieve_recommendations(collection, question_embedding, top_k=10):
    try:
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k
        )
        if results and results.get('documents'):
            return results['documents'][0]
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการดึงข้อมูลจากฐานข้อมูล: {e}")
    return []

# --- ฟังก์ชันตรวจข้อความปิดท้าย ---
def is_closing_message(text):
    closing_patterns = [
        r"^ขอบคุณ.*", r"^ขอบใจ.*", r"^โอเค.*", r"^เข้าใจ.*", r"^ได้เลย.*", r"^รับทราบ.*",
        r"^thank(s| you).*", r"^ok.*", r"^got it.*", r"^noted.*", r"^understood.*"
    ]
    text = text.strip().lower()
    if len(text.split()) <= 5:
        for pattern in closing_patterns:
            if re.match(pattern, text):
                return True
    return False

# --- ฟังก์ชันตรวจ gibberish หรือ typo ง่ายๆ ---
def is_gibberish_or_typo(text):
    text = text.strip()
    if len(text) <= 2:
        return True
    words = text.split()
    if len(words) == 1 and not re.search(r'[a-zA-Zก-๙]', words[0]):
        return True
    return False

# --- ฟังก์ชันตรวจภาษาแบบง่าย ---
def detect_language(text):
    thai_chars = re.findall(r'[\u0E00-\u0E7F]', text)
    return "th" if len(thai_chars) / max(len(text), 1) > 0.3 else "en"

# --- Session state สำหรับเก็บประวัติแชท ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- เริ่มโหลดฐานข้อมูล ChromaDB แบบ persistent client ---
try:
    if not os.path.exists(folder_path):
        # ถ้าไม่มีโฟลเดอร์ฐานข้อมูล ให้ดาวน์โหลดและแตกไฟล์ก่อน
        download_and_extract()
    client = PersistentClient(path=folder_path)
except Exception as e:
    st.error(f"❌ ไม่สามารถโหลด ChromaDB ได้: {e}")
    # ลองลบฐานข้อมูลเก่าแล้วสร้างใหม่
    clear_old_database(folder_path)
    st.info("ลองสร้างฐานข้อมูลใหม่จากศูนย์...")
    try:
        client = PersistentClient(path=folder_path)
        collection = create_empty_collection(client)
    except Exception as e2:
        st.error(f"❌ สร้างฐานข้อมูลใหม่ก็ล้มเหลว: {e2}")
        st.stop()
else:
    # โหลดหรือสร้าง collection ชื่อ "recommendations"
    try:
        collection = client.get_collection(name="recommendations")
    except Exception:
        collection = create_empty_collection(client)

# --- UI ---
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
            question_embedding = embedding_model.encode(user_input).tolist()
            recommendations = retrieve_recommendations(collection, question_embedding, top_k=10)

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

            reply = query_llm_with_chat(prompt, api_key)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant", avatar="🧘‍♀️"):
        st.markdown(reply)
