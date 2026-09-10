# 数字员工 · 企业级 RAG 多智能体工作台

> 给一家城市燃气公司做的**内网数字员工**：员工用工号登录，向不同"专家角色"提问，回答自动检索公司资料并标注出处。
> 全栈从零实现——**不依赖 LangChain / LlamaIndex 等框架**，便于看清每一步在做什么、哪里会出问题。

![对话主界面](screenshots/1-chat-light.png)

## 这是什么

企业内部知识助手 + 文书助手。9 个专家角色（文档、数据、代码、图纸、采购、人事、生产、总调度）共用一套知识库，可以上传资料、沉淀问答知识、共享提示词技能。

- **面向真实使用**：账号由管理员在花名表里发放，普通员工只能检索自己的资料，管理员上传的资料进共享库
- **全本地部署**：检索、向量化、重排、索引全在本机跑，只有生成回答调用大模型 API（DeepSeek）
- **数据不出内网**：知识库、会话、记忆都存在本地 SQLite 与文件里

## 功能

| 模块 | 说明 |
|---|---|
| RAG 问答 | BM25 + CrossEncoder 重排 + FAISS，回答带出处标注 |
| 多智能体编排 | 总调度"管家"拆解任务 → 分派专家 → 汇总结果（自实现 function calling 循环） |
| 长期记忆 | 按工号分区，自动从对话中抽取事实并写入 SQLite，下次对话带记忆 |
| 知识沉淀 | 有价值的问答自动入库，由管理员审核后生效 |
| 技能共享 | 把好用的提示词沉淀成"技能"，全公司下载复用 |
| 文档生成 | 回答可一键保存为 Word 并下载 |
| 索引自动重建 | 文档丢进 `docs/`，服务自动检测变化 → 后台重建索引 → 热加载，不用敲命令 |
| 权限 | 工号注册/登录、管理员审核、个人知识库与共享知识库分离 |

## 实测效果

检索质量用 58 题评估集（真实业务问题）评测，MRR@10 对比多种方案：

| 检索方案 | MRR@10 | Recall@10 |
|---|---|---|
| **BM25 + 重排（生产配置）** | **0.875** | **0.96** |
| 纯 BM25 | 0.837 | 0.96 |
| 向量检索（bge-small + FAISS） | 0.541 | 0.72 |
| 混合检索（向量+BM25 RRF 融合） | 0.726 | 0.96 |

结论：**这批"事实型"数据上，BM25 + 重排就是最优，混合检索反而更差**（向量噪声打乱排序）。详见下文"关键工程决策"。

端到端实测（7 题：6 个事实题 + 1 个库里没有的负样本）：检索命中 7/7，答案正确 6/6，负样本明确回答"资料中未记载、建议查原始合同"，**没有编造**。

## 技术架构

```
员工浏览器 ──► FastAPI (webui/app.py)
                 │
                 ├─ 账号 / 花名表 / 会话 / 记忆  ──► SQLite (webui/data/app.db)
                 │
                 ├─ 多智能体编排：管家拆解 → 专家执行 → 汇总
                 │
                 ├─ 检索：retrieval.py
                 │    ├─ 召回  BM25 (jieba 分词)  ┐
                 │    ├─ 召回  向量 (bge-small-zh) ├─ 策略可插拔
                 │    ├─ 召回  混合 (RRF)         ┘
                 │    └─ 重排  bge-reranker-base (cross-encoder)
                 │
                 ├─ 索引：rebuild_index.py + index_meta.py
                 │    └─ docs/ 指纹变化 → 后台子进程重建 → 原子替换 → 热加载
                 │
                 └─ 生成：DeepSeek API（OpenAI 兼容协议）
```

## 快速开始

```bash
# 1) 依赖（建议 Python 3.11）
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install fastapi uvicorn openai faiss-cpu rank-bm25 jieba sentence-transformers \
            pymupdf python-docx openpyxl numpy

# 2) 放入你自己的资料（示例文档可直接用来试跑）
copy sample_docs\* docs\

# 3) 配置大模型 key（只放密钥，不要写进代码）
echo DEEPSEEK_API_KEY=sk-xxxxxxxx > .env

# 4) 建索引（首次会下载 bge 模型）
.venv\Scripts\python.exe rebuild_index.py

# 5) 启动
.venv\Scripts\python.exe -m uvicorn app:app --app-dir webui --host 0.0.0.0 --port 8644
```

打开 <http://127.0.0.1:8644>，用工号注册登录（管理员账号 `admin` 可管理花名表与知识审核）。

评估检索效果：

```bash
.venv\Scripts\python.exe eval_retrieval.py --strategies bm25,vector,hybrid --rerank both
```

Windows 用户也可以直接双击 `启动数字员工.bat`（内含环境检查、就绪轮询、自动开浏览器）。

## 目录结构

```
retrieval.py          检索核心：策略可插拔（bm25 / vector / hybrid）+ 重排
rebuild_index.py      全量重建索引（多格式解析、切分策略、原子写盘）
index_meta.py         docs 目录指纹 + 索引元信息（自动重建的判据）
eval_retrieval.py     检索评估脚本（MRR / Recall / Precision）
start_webui.py        启动器（已运行检测、端口占用提示、就绪轮询）
reset_pwd.py          重置/新建登录密码
webui/app.py          FastAPI 后端：账号、记忆、编排、知识沉淀、上传下载
webui/static/         前端三页：登录 / 对话工作台 / 技能库
sample_docs/          示例业务文档（虚构公司，用于试跑）
评估集_示例.json        示例评估集
screenshots/          界面截图
stage*.py             学习阶段的递进脚本（从最简对话 → 最简 RAG → 重排 → 记忆）
```

## 关键工程决策（都有数据支撑）

**1. 切分：单层 400 字符，别的都更差**

对比了 4 种切分（16 题封闭测试集）：单层 400 得 0.667，父子文档 200/800 得 0.674（打平），按行切分只有 0.235（-43%），按行+工程名前缀 0.296（-37%）。原因是这批资料是"小型分层结算表"，400 字符累积恰好把"工程名 + 子项 + 金额"装进同一块。

**2. 检索：BM25 + 重排，混合检索是负优化**

向量检索最弱（0.541）；混合检索虽然 Recall 与 BM25 相同，但 MRR 更差（0.726 vs 0.837）——向量噪声打乱了原本正确的排序。重排（cross-encoder）是最大单项提升（+4.5% MRR）。

**3. 一个真实 bug 带来的最大提升：docx 合并单元格去重**

python-docx 读取合并单元格时会把同一单元格重复返回 4~8 次，导致工程名在一个块里反复出现、BM25 词频虚高，把真正的答案挤出候选。加"相邻单元格去重"后：索引块数 11837 → 7264（-38.6%），MRR 0.860 → 0.875，Recall 0.940 → 0.960。**"问工期检索不到"的真根因是它，不是检索算法。**

**4. 索引自动重建：让"丢文件"这件事零操作**

给 `docs/` 目录算指纹（文件路径 + 大小 + mtime），服务启动时和运行中定时比对：变了就起一个子进程重建，重建完热加载，服务全程不中断。索引文件先写 `.tmp` 再原子替换，保证重建过程中不会有"半个索引"被读到。

**5. 自实现 function calling 循环**

没有用框架的 agent 封装，自己实现"模型返回 tool_calls → 执行工具 → 结果回灌 → 继续对话"的循环。多智能体编排也是自写的：总调度拆解任务、分派给专家角色、再汇总。

## 关于数据与隐私

本仓库**只包含代码**，不包含任何真实业务数据：

- 公司资料（`docs/`）、知识库派生文件（`chunks.json` / `faiss.index`）、评估集、数据库、上传文件、密钥均已在 `.gitignore` 中排除
- `sample_docs/` 与 `评估集_示例.json` 中的公司名、人名、金额**全部为虚构**，仅用于让仓库下载后可以直接跑通

如果你要把它用于自己的场景：把资料放进 `docs/`，配好 `.env` 里的 key，跑一次 `rebuild_index.py` 即可。

## License

MIT
