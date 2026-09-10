import os
os.environ["HF_HOME"] = "E:/hf_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
from sentence_transformers import CrossEncoder
print("开始下载 bge-reranker-base ...")
model = CrossEncoder("BAAI/bge-reranker-base")
print("下载完成")
