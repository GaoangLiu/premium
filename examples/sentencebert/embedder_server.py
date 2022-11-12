# 语义搜索
# https://www.sbert.net/examples/applications/semantic-search/README.html
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import torch
import codefast as cf

embedder = SentenceTransformer('bert-base-chinese')
embedder.encode("Great, thanks! I will check it out.")
app = Flask(__name__)
import gc 

@app.route('/encode', methods=['POST'])
def get_encoder():
    corpus = request.json['corpus']
    tensors = embedder.encode(corpus, convert_to_tensor=False)
    uid = cf.uuid()
    path = f'/tmp/{uid}.pt'
    torch.save(tensors, path)
    return {'result_path': path}

app.run(host='0.0.0.0', port=5678)
