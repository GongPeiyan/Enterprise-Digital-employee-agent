# -*- coding: utf-8 -*-
# 阶段3进阶：SQLite 记忆分区（多用户隔离 + 同主题覆盖更新）
# 运行：.venv\Scripts\python.exe stage3_memory_partition.py

import sqlite3, os
from openai import OpenAI

api_key = os.getenv("DEEPSEEK_API_KEY", "")
if not api_key:
    raise SystemExit("未找到 DEEPSEEK_API_KEY：请先设置环境变量，或在项目根目录 .env 里配置")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

DB = "E:/数字员工项目/memory.db"

# ========== 1. 初始化 SQLite ==========
def init_db():
    conn = sqlite3.connect(DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,      -- 谁的记忆（分区关键）
            topic TEXT,         -- 主题（如"名字""职业"，更新关键）
            fact TEXT,          -- 事实内容
            ts TEXT
        )""")
    conn.commit()
    conn.close()

# ========== 2. 记忆读写（按 username 隔离） ==========
def load_memories(username):
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT fact FROM memories WHERE username=? ORDER BY id", (username,)
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]

def upsert_memory(username, topic, fact):
    """同一个人、同一主题 → 覆盖旧事实（这是'更新'的关键）"""
    conn = sqlite3.connect(DB)
    conn.execute("DELETE FROM memories WHERE username=? AND topic=?", (username, topic))
    conn.execute(
        "INSERT INTO memories(username, topic, fact, ts) VALUES(?,?,?,datetime('now'))",
        (username, topic, fact),
    )
    conn.commit()
    conn.close()

# ========== 3. 用 LLM 提取"主题|事实" ==========
def extract_memory(user_msg, assistant_msg):
    prompt = (
        "从下面这段对话里，提取值得长期记住的、关于用户的事实。\n"
        "以下这些【一定要记，宁可多记不可漏记】：\n"
        "- 用户的个人信息：名字、职业、身份、年龄、生日、所在城市等\n"
        "- 用户的偏好和喜好：爱吃什么喝什么、习惯、讨厌什么（比如'我爱喝美式'）\n"
        "- 用户正在做的事、目标、重要决定\n"
        "- 用户明确要求记的（说了'记住''别忘了'等）\n"
        "输出格式：主题|事实（例如：咖啡偏好|爱喝美式不加糖）。每行一条。\n"
        "只有确实没出现任何关于用户的信息时，才回复：无\n"
        f"用户：{user_msg}\nAI：{assistant_msg}"
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=200,
    )
    return resp.choices[0].message.content.strip()

# ========== 主流程：输入用户名模拟登录 ==========
init_db()

username = input("请输入用户名（模拟登录）：").strip()
memories = load_memories(username)
memory_text = "\n".join(f"- {m}" for m in memories) if memories else "（暂无）"

messages = [
    {"role": "system", "content": f"你是数字员工。你记得关于 {username} 的这些事：\n{memory_text}"},
]

print(f"当前用户：{username}，已有记忆 {len(memories)} 条")
print("输入 exit 退出")

while True:
    q = input("\n你：")
    if q.strip().lower() == "exit":
        break
    messages.append({"role": "user", "content": q})
    resp = client.chat.completions.create(
        model="deepseek-chat", messages=messages, temperature=0.7, max_tokens=1000
    )
    reply = resp.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    print("AI：", reply)

    result = extract_memory(q, reply)
    if result and result != "无":
        for line in result.split("\n"):
            if "|" in line:
                topic, fact = line.split("|", 1)
                topic, fact = topic.strip(), fact.strip()
                upsert_memory(username, topic, fact)
                print(f"  [🧠 记住：{topic} | {fact}]")
