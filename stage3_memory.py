# -*- coding: utf-8 -*-
# 阶段3：对话记忆 —— 让 AI 跨会话记住"人"和"事"
# 运行：.venv\Scripts\python.exe stage3_memory.py

import os, json
from openai import OpenAI

api_key = os.getenv("DEEPSEEK_API_KEY", "")
if not api_key:
    raise SystemExit("未找到 DEEPSEEK_API_KEY：请先设置环境变量，或在项目根目录 .env 里配置")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

MEMORY_FILE = "E:/数字员工项目/memories.json"

# ========== 长期记忆的"读"和"写" ==========
def load_memories():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memories(mems):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(mems, f, ensure_ascii=False, indent=2)

# ========== 用 LLM 从一轮对话里提取"值得记的事实" ==========
def extract_memory(user_msg, assistant_msg):
    prompt = (
        "从下面这段对话里，提取一条\"值得长期记住的事实\"（比如用户的名字、身份、偏好、重要决定、正在做的事）。\n"
        "如果这段对话里没有值得记的，就只回复两个字：无\n"
        f"用户：{user_msg}\n"
        f"AI：{assistant_msg}\n"
        "值得记住的事实："
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()

# ========== 启动时：把长期记忆读出来，注入 system prompt ==========
memories = load_memories()
memory_text = "\n".join(f"- {m}" for m in memories) if memories else "（暂无）"

messages = [
    {"role": "system", "content": f"你是用户的数字员工。你记得关于用户的这些事：\n{memory_text}"},
]

print(f"已加载 {len(memories)} 条记忆")
print("开始对话，输入 exit 退出")

while True:
    user_input = input("\n你：")
    if user_input.strip().lower() == "exit":
        break

    # 短期记忆：累积 messages（和阶段0一样）
    messages.append({"role": "user", "content": user_input})
    resp = client.chat.completions.create(
        model="deepseek-chat", messages=messages, temperature=0.7, max_tokens=2000
    )
    reply = resp.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    print("AI：", reply)

    # 每轮结束后：判断这轮有没有值得记的，有就存进长期记忆
    fact = extract_memory(user_input, reply)
    if fact and fact != "无":
        memories.append(fact)
        save_memories(memories)
        print(f"  [🧠 记住了一条：{fact}]")
