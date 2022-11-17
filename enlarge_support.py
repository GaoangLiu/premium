# 语义搜索
# https://www.sbert.net/examples/applications/semantic-search/README.html
# 给定少数样本，扩充支撑集
from sentence_transformers import SentenceTransformer, util
import torch
import codefast as cf
from typing import List

def encode(corpus: List[str]):
    rsp = cf.net.post('http://localhost:5678/encode', json={'corpus': corpus})
    result_path = rsp.json()['result_path']
    return torch.load(result_path)


def enlarge_support(support: List[str], query: str, top_k: int = 5):
    # 1. encode support
    support_embeddings = encode(support)

    # 2. encode query
    query_embedding = encode([query])

    # 3. compute cosine similarity
    cos_scores = util.pytorch_cos_sim(query_embedding, support_embeddings)[0]
    cos_scores = cos_scores.cpu()

    # 4. get top_k
    top_results = torch.topk(cos_scores, k=top_k)

    # 5. return
    return [support[idx] for idx in top_results.indices]