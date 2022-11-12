# 语义搜索
# https://www.sbert.net/examples/applications/semantic-search/README.html
from sentence_transformers import SentenceTransformer, util
import torch
import codefast as cf
from typing import List
# embedder = SentenceTransformer('bert-base-chinese')


def encode(corpus: List[str]):
    rsp = cf.net.post('http://localhost:5678/encode', json={'corpus': corpus})
    result_path = rsp.json()['result_path']
    return torch.load(result_path)


# Corpus with example sentences
corpus = ['技术现实主义将被历史视为一场悲剧性的运动。',
          '我脑子里一直在想一个主意，我觉得现在该提出来了。',
          '是啊，这几天听起来像约翰尼。']
corpus = cf.io.read('/tmp/train.txt').data
queries = cf.l(cf.io.walk('/tmp/talks')).map(cf.js).flatten().map(lambda x: x['content']).data

corpus += queries
curpus = list(set(corpus))

corpus_embeddings = encode(corpus)

# Query sentences:
# queries = ['A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.',
#            'A cheetah chases prey on across a field.']
queries = list(set(queries))
queries += ['我有一辆油车自己开，给我爸买的。', '19款轩逸，我去接小孩去', '接小孩用的。对171819也行。', '我要开车去接孩子了', '给我媳妇买来通勤用的', '女朋友上班通勤时间长，有辆车方便', '车是给我爸买的 前两天相中一个vv7 他说油耗高[破涕为笑]']

query_embeddings = encode(queries)

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(10, len(corpus))
for i, query in enumerate(queries):
    query_embedding = query_embeddings[i]

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("-"*77)
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        if score < 0.99:
            print("{:.4f}, {}".format(score, corpus[idx]))

    """
    # Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
    hits = hits[0]      #Get the hits for the first query
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
    """
