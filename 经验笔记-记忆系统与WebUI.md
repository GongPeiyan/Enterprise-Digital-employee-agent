# 经验笔记：多用户记忆系统 + WebUI

> 覆盖：SQLite 记忆分区、对话历史持久化、多用户 WebUI 架构，以及一路踩过的坑。

---

## 一、记忆系统的两个独立问题

记忆 = 判断（什么该记）+ 存储（怎么存、存哪、怎么隔离）。两者要分开想：

- **判断**：LLM 决定"这条要不要进记忆"。prompt 要写死判断标准（个人信息/偏好/目标/明确要求记的），
  并强调"宁可多记不可漏记"——漏记是永久丢失，多记只是噪音可清理。
- **存储**：SQLite 表 + 字段隔离 + 同主题覆盖。

## 二、SQLite 记忆分区（多用户隔离 + 覆盖更新）

关键就一张表的两列：

```sql
CREATE TABLE memories(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    emp_id TEXT,   -- 谁的记忆（分区关键：多用户隔离靠它）
    topic TEXT,    -- 主题（更新关键：同主题覆盖靠它）
    fact TEXT,     -- 事实内容
    ts TEXT
);
```

- **多用户隔离**：查记忆都带 `WHERE emp_id=?`，A 永远读不到 B 的。
- **覆盖更新**：`DELETE FROM memories WHERE emp_id=? AND topic=?` 先删旧值再插新值，
  这样"我叫张三"→"我叫小明"是覆盖，不是两条并存（否则 prompt 里看到矛盾信息，LLM 随机答）。

```python
def upsert_memory(emp_id, topic, fact):
    conn.execute("DELETE FROM memories WHERE emp_id=? AND topic=?", (emp_id, topic))
    conn.execute("INSERT INTO memories(emp_id, topic, fact, ts) VALUES(?,?,?,datetime('now'))", ...)
```

## 三、账号设计：花名表 + 工号（不是身份证后六位）

- **花名表（白名单）**：roster 表存"工号+姓名"，只有名单上的人能注册。
- **登录唯一 key 用工号**：唯一、稳定、不敏感。身份证后六位有两个坑——①不唯一（跨区县可重复）
  ②涉及隐私，不宜做登录凭证。
- 关键认知：**账号（username/工号）和"记忆里的名字"是两码事**。账号是固定 key（像身份证号），
  "我叫什么"只是记忆里一条可变的事实。改名字不影响登录。

## 四、对话历史持久化（产品完整性）

完整对话记录要落盘，否则刷新/重启就丢。做法：

```sql
CREATE TABLE messages(id INTEGER PRIMARY KEY AUTOINCREMENT, emp_id TEXT, role TEXT, content TEXT, ts TEXT);
```

每轮对话存两条（user + assistant），登录时 `load_history()` 恢复。关键设计：
- **存"原始问题"，不存带 RAG 参考资料的版本**——参考资料只在调用 LLM 时临时拼上，
  这样历史记录干净、可直接展示。
- 三层记忆的状态：长期事实（SQLite 永久）、对话上下文（内存，重启丢）、完整记录（SQLite，需主动存）。

## 五、多用户 WebUI 架构

```
浏览器（每人登录）→ FastAPI 后端 → SQLite（roster/users/memories/messages 四张表）+ 共享 RAG 索引
```

- 技术栈：FastAPI + 原生 HTML/JS + SQLite + 标准库 hash（最小依赖）。
- 会话：内存 dict `conversations[emp_id] = messages`，每工号独立。
- 认证：登录发 token（secrets.token_hex），前端存 localStorage，请求带 Authorization 头。

## 六、踩坑记录（都是真实遇到的）

1. **faiss 的 write_index 不支持中文路径**（底层 C++ IO）。改用
   `faiss.serialize_index(index).tobytes()` 存 + `np.frombuffer(..., dtype=np.uint8)` 读，绕开。
2. **pymupdf 默认"列优先"提取打乱表格** → 见《经验笔记-表格处理.md》。
3. **改后端代码必须重启服务器**，光刷新浏览器（F5）没用——旧进程还是旧代码。
4. **忘 import uvicorn** → NameError。写完代码先跑一遍语法检查。
5. **主题不一致**：LLM 提取记忆时主题词不统一（"姓名"vs"个人信息"），导致同主题覆盖失效、
   存冗余矛盾。产品级需做"主题规范化"（固定主题字典或让 LLM 用统一措辞）。

## 七、关键文件位置（本机）

- 项目根：E:\数字员工项目\
- WebUI 后端：webui\app.py（FastAPI）
- 前端：webui\static\login.html + chat.html
- 数据库：webui\data\app.db（roster/users/memories/messages 四张表）
- 共享知识库：faiss.index + chunks.json（docs\ 目录放 PDF）
- 端口：8644（app.py 里可改）
