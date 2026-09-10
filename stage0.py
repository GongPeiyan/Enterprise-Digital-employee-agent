# 阶段 0：跑通大模型多轮对话（最小可运行版本）
# 目标：能调用 DeepSeek 完成多轮对话，理解"模型无状态、靠 messages 累积"这件事。
#
# 用法：
#   1) 装库：pip install openai -i https://pypi.tuna.tsinghua.edu.cn/simple
#   2) 配 key：在项目根目录 .env 里写 DEEPSEEK_API_KEY=sk-xxxx，或设为环境变量
#   3) 运行：python stage0.py

import os

from openai import OpenAI

# 1. API key 从环境变量读取，不要写死在代码里
api_key = os.getenv("DEEPSEEK_API_KEY", "")
if not api_key:
    raise SystemExit("未找到 DEEPSEEK_API_KEY：请先设置环境变量，或在项目根目录 .env 里配置")

# 2. 初始化客户端：DeepSeek 提供 OpenAI 兼容接口，换 base_url 即可复用 openai 库
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# 3. messages 装着整段对话，role 只有 system / user / assistant
messages = [
    {"role": "system", "content": "你是一个简洁的助手，回答尽量简短。"},
]

print("开始对话，输入 exit 退出")
while True:
    user_input = input("\n你：")
    if user_input.strip().lower() == "exit":
        break

    # 4. 用户这句话追加进历史
    messages.append({"role": "user", "content": user_input})

    # 5. 把整段历史一起发给模型（模型本身不记得上一句，是 messages 让它"记得"）
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.7,
        max_tokens=2000,
    )

    reply = resp.choices[0].message.content

    # 6. AI 这句话也追加进历史，下一轮就能看到自己说过什么
    messages.append({"role": "assistant", "content": reply})

    print("AI：", reply)
