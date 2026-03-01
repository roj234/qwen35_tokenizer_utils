
import argparse
import csv

from tqdm import tqdm
import numpy as np

LLAMA_POOLING_TYPE_NONE = 0
LLAMA_POOLING_TYPE_MEAN = 1
LLAMA_POOLING_TYPE_CLS = 2
LLAMA_POOLING_TYPE_LAST = 3
LLAMA_POOLING_TYPE_RANK = 4
from llama_cpp import Llama
from llama_cpp.llama_embedding import LlamaEmbedding

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def analyze_semantic_alignment(tokenizer: Llama, model: LlamaEmbedding, text: bytes, token_id: int):
    full_tokens = [token_id]
    split_tokens = []

    prev_tokens = tokenizer.tokenize(text)
    if len(prev_tokens) == 1:
        return

    for t in prev_tokens:
        #print(tokenizer.detokenize([t]).decode("utf-8"))
        split_tokens += model.tokenize(tokenizer.detokenize([t]))

    emb0 = np.squeeze(model.embed([
        full_tokens,
    ]))
    emb1 = np.squeeze(model.embed([
        split_tokens,
    ]))

    # 4. 计算相似度
    similarity = cosine_similarity(
        emb0,
        emb1
    )

    return {
        "Alignment": similarity,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="语义对齐算法")

    # 模型路径参数
    parser.add_argument("--ref_model", type=str, help="参考模型路径 (用于分词)", required=True)
    parser.add_argument("--source_model", type=str, help="源模型路径 (用于计算 Embedding)", required=True)

    # 文件输入输出参数
    parser.add_argument("--input", type=str, default="tokens_preview.csv", help="输入 CSV 文件路径")
    parser.add_argument("--output", type=str, default="tokens_embedding_alignment.csv", help="输出 CSV 文件路径")
    parser.add_argument("--total", type=int, default=None, help="tqdm 显示的总行数")

    # 模型运行参数
    parser.add_argument("--gpu_layers", type=int, default=-1, help="加载到 GPU 的层数")
    parser.add_argument("--threshold_bad", type=float, default=0.6, help="判定为 BAD 的阈值")
    parser.add_argument("--threshold_critical", type=float, default=0.5, help="判定为 CRITICAL 的阈值")

    args = parser.parse_args()

    tokenizer = Llama(
        model_path=args.ref_model,
        vocab_only=True,
        verbose=False
    )

    llm = LlamaEmbedding(
        model_path=args.source_model,
        verbose=False,
        embeddings=True,
        pooling_type=LLAMA_POOLING_TYPE_LAST,
        n_ctx=512,
        n_gpu_layers=args.gpu_layers
    )

    with open(args.input, mode='r', encoding='utf-8', errors='ignore') as infile:
        reader = csv.DictReader(infile)

        with open(args.output, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ID", "Text", "Alignment", "Label"])
            writer.writeheader()

            for row in tqdm(reader, total=args.total):
                if int(row['Delete']): continue

                token_id = int(row['ID'])
                text = row['Text'].encode("utf-8")

                res = analyze_semantic_alignment(tokenizer, llm, text, token_id)
                if not res:
                    writer.writerow({
                        "ID": row["ID"],
                        "Text": row["Text"],
                        "Alignment": 1,
                        "Label": "Same",
                    })
                    continue

                status = "BAD" if res['Alignment'] < args.threshold_bad else "OK"
                if res['Alignment'] < args.threshold_critical: status = "CRITICAL"

                writer.writerow({
                    "ID": row["ID"],
                    "Text": row["Text"],
                    "Alignment": res['Alignment'],
                    "Label": status,
                })
