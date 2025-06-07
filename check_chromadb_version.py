import streamlit as st
import sqlite3
import os

st.title("🔍 ตรวจสอบฐานข้อมูล ChromaDB")

db_path = "chromadb_database_v2/chroma.sqlite3"

# ตรวจว่าไฟล์มีอยู่จริงหรือไม่
if not os.path.exists(db_path):
    st.error(f"❌ ไม่พบไฟล์ฐานข้อมูลที่: {db_path}")
else:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # ดึง schema ของตาราง collections
        cursor.execute("PRAGMA table_info(collections);")
        columns = cursor.fetchall()

        st.subheader("📋 คอลัมน์ในตาราง 'collections'")
        for col in columns:
            st.write(f"• {col[1]}")

        # ตรวจว่ามีคอลัมน์ topic หรือไม่
        if any(col[1] == "topic" for col in columns):
            st.success("✅ พบคอลัมน์ 'topic' → ต้องใช้ ChromaDB เวอร์ชัน **>= 0.4.24**")
        else:
            st.warning("⚠️ ไม่พบคอลัมน์ 'topic' → ใช้ ChromaDB เวอร์ชัน **< 0.4.24** เท่านั้น")

        conn.close()
    except Exception as e:
        st.error(f"❌ ไม่สามารถเชื่อมต่อฐานข้อมูล: {e}")
