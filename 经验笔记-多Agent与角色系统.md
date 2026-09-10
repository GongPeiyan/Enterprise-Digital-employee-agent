# 经验笔记：多Agent协作 + 角色系统 + 即梦API踩坑

> 覆盖：多 Agent 调度者、角色系统、知识沉淀、流式输出、火山引擎即梦 API 接入（重点踩坑）。

---

## 一、多 Agent 调度者（管家模式）

核心架构：一个"调度者" + 多个"专家 agent"。

```
用户 → 管家(调度者) → 拆解任务 → 分发给专家(妙妙/准准/图图...) → 各专家执行 → 汇总成最终结果
```

关键函数三个：
- `orchestrate_plan(task)`：管家拆解任务，输出"专家名|子任务"列表（用 LLM 拆解）
- `expert_run(role_prompt, sub_task)`：专家用自己的人设 + 子任务，生成结果
- `summarize(task, results)`：管家汇总各专家结果

**关键认知**：区分三个层次——
1. 用户 → agent（单角色问答）
2. agent → 工具（调函数，工具是"哑巴"）
3. agent → agent（多 agent 协作，有对话协商）

"写字"是 LLM 天生会做的（不调工具）；"画图/查数据"是 LLM 做不到的（要调工具）。

## 二、角色系统（叠字名 + 侧边栏 + 召唤）

- 角色 = 人设（system prompt）+ 专属职责。用叠字名好记（妙妙/准准/码码/图图/采采/安安/顺顺）。
- 前端侧边栏：emoji 头像 + 名字，点击切换，当前高亮。
- 自然语言召唤：detect_role() 用 LLM 判断"用户消息是否在召唤角色"（关键词预筛 + LLM 精确判断），召唤后返回 role，前端 switchRole 高亮。
- 后端 done 消息里带 `role` 字段，前端据此高亮。

## 三、知识沉淀（自动提炼 + 管理员审核）

- 每轮对话后 extract_knowledge() 判断"有没有公共知识"，提炼成"问题|答案"，存 pending。
- 管理员审核通过 → 向量化进全局知识库（add_text_to_global）。
- 与"记忆"的区别：记忆是"某个人的私有事实"，知识是"全公司共享的公共知识"。

## 四、流式输出（SSE）踩坑

**关键坑：同步 openai client + async generator 会导致响应被缓冲**（攒到最后一次性发出）。

解法：把 generate 从 `async def` 改成 `def`（同步 generator），FastAPI 会自动放线程池跑，流式正常。

前端：getReader 逐块读 + buffer 累积 + 按 `\n\n` 切 SSE 消息 + JSON.parse + 追加显示（打字机）。

## 五、火山引擎即梦 API 接入（重点踩坑，以后还会遇到）

### 凭据编码坑（最耗时）
- 从密码管理器/导出工具复制的 AK/SK 会被 **base64 编码**。
- 实测：SK 被**双重** base64 编码（`TlRBMk...==` → `NTA2MzIy...` → `5063225673c945d688f7c073568454b0` 32位hex）。
- AK 被**单次** base64 编码（`AKLT` + 43字符 base64 → `AKLT` + 32位hex）。
- 教训：让用户直接在控制台网页上复制凭据，别经过密码管理器。

### 签名
- **别手写 SigV4 签名**（我手写三次都 401），直接用官方 SDK：`pip install volcengine`。
- SDK 用法：
```python
from volcengine.visual.VisualService import VisualService
vs = VisualService()
vs.set_ak(ak); vs.set_sk(sk)
resp = vs.cv_sync2async_submit_task({"req_key": "jimeng_t2i_v31", "prompt": "...", "width": 1024, "height": 1024})
# 拿 task_id，轮询查询
vs.cv_sync2async_get_result({"req_key": "jimeng_t2i_v31", "task_id": task_id, "req_json": json.dumps({"return_url": True})})
```

### 即梦文生图接口要点
- 接口：https://visual.volcengineapi.com（视觉智能 cv 服务，Region=cn-north-1）
- 两步异步：CVSync2AsyncSubmitTask（提交，返回 task_id）→ CVSync2AsyncGetResult（轮询查结果，返回 image_urls）
- req_key 固定 `jimeng_t2i_v31`（文生图 3.1）

### 报错含义
- SignatureDoesNotMatch = 签名错（SK 不对）
- InvalidAccessKey = 签名对但 AK 无效（AK 值错/失效/账号没开通服务）

## 六、其他坑
- faiss 的 write_index 不支持中文路径，用 serialize_index().tobytes() + frombuffer 绕开。
- 改后端代码必须重启服务器，光刷新浏览器没用。
- FastAPI 文件上传（UploadFile）需要装 python-multipart。
- docx 读取用 python-docx（段落 + 表格）。
