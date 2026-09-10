# -*- coding: utf-8 -*-
"""
检索核心：策略可插拔。生产(app.py)与评估(eval_retrieval.py)共用同一套逻辑，
保证"评估测的就是生产在跑的"。

检索策略（candidate_ids 的 strategy 参数）：
- bm25    纯关键词召回（默认，当前事实型数据最优）
- vector  纯向量语义召回
- hybrid  向量+BM25 的 RRF 融合（当前数据上实验证明是负优化，留作数据异构时备选）

切换判据（"答案装不装得进一个块"）：
- 装得进（事实型：数字/名词/术语，答案在单块内）→ bm25
- 装不下（长文本/答案跨段/问法千变万化）→ 才切 hybrid/vector，并配合父子文档 chunk

全局切换：改 RETRIEVAL_STRATEGY 一处，生产 + 评估同时生效。
"""
import numpy as np
import jieba

# 全局默认检索策略
RETRIEVAL_STRATEGY = "bm25"


def candidate_ids(query, bm25, faiss_idx=None, embed_model=None, n=20, strategy=None):
    """按策略召回 top-n 候选块 id（未 rerank）。

    bm25 / faiss_idx / embed_model 由调用方传入，避免对全局对象的隐式依赖。
    """
    strategy = strategy or RETRIEVAL_STRATEGY

    if strategy == "bm25":
        scores = bm25.get_scores(list(jieba.cut(query)))
        return np.argsort(scores)[::-1][:n].tolist()

    if strategy == "vector":
        q_vec = embed_model.encode([query]).astype("float32")
        _, ids = faiss_idx.search(q_vec, n)
        return ids[0].tolist()

    if strategy == "hybrid":
        q_vec = embed_model.encode([query]).astype("float32")
        _, vids = faiss_idx.search(q_vec, n)
        vids = vids[0].tolist()
        bids = np.argsort(bm25.get_scores(list(jieba.cut(query))))[::-1][:n].tolist()
        rrf = {}
        for rank, i in enumerate(vids):
            rrf[i] = rrf.get(i, 0) + 1.0 / (60 + rank)
        for rank, i in enumerate(bids):
            rrf[i] = rrf.get(i, 0) + 1.0 / (60 + rank)
        return sorted(rrf, key=rrf.get, reverse=True)[:n]

    raise ValueError(f"未知检索策略 {strategy!r}，可选 bm25 / vector / hybrid")


def rerank_scores(query, texts, reranker):
    """cross-encoder 重排打分，返回与 texts 等长的分数列表。rerank 是最大单项提升，别去掉。"""
    if not texts:
        return []
    return list(reranker.predict([(query, t) for t in texts]))


def search(query, bm25, texts, reranker, faiss_idx=None, embed_model=None,
           k=10, n=20, strategy=None):
    """单库完整检索：召回(策略) + 重排(rerank)，返回 top-k 块 id 列表。评估脚本用。"""
    ids = candidate_ids(query, bm25, faiss_idx, embed_model, n, strategy)
    scores = rerank_scores(query, [texts[i] for i in ids], reranker)
    ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked[:k]]
