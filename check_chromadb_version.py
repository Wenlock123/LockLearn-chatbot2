import sqlite3

# ชี้ไปยัง path ของไฟล์ฐานข้อมูล
conn = sqlite3.connect("chromadb_database_v2/chroma.sqlite3")
cursor = conn.cursor()

# ดู schema ของตาราง collections
cursor.execute("PRAGMA table_info(collections);")
columns = cursor.fetchall()

print("🧱 คอลัมน์ในตาราง 'collections':")
for col in columns:
    print(f"- {col[1]}")

# ตรวจว่ามี 'topic' ไหม
if any(col[1] == "topic" for col in columns):
    print("\n✅ พบคอลัมน์ 'topic' → ต้องใช้ ChromaDB >= 0.4.24")
else:
    print("\nℹ️ ไม่พบคอลัมน์ 'topic' → ใช้ ChromaDB < 0.4.24 ก็ได้")
