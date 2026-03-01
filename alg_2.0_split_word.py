
import argparse
import csv
import json

from tqdm import tqdm
import numpy as np

from llama_cpp import Llama

#from llama_cpp._ggml import libggml
#libggml.ggml_backend_load("ggml-cuda.dll".encode("utf-8"))
#libggml.ggml_backend_load("ggml-cpu-alderlake.dll".encode("utf-8"))

from typing import List

def calculate_ppl(model: Llama, tokens: List[int], only_last=True):
    """计算 PPL"""
    n_tokens = len(tokens)

    model.reset()
    model.eval(tokens)

    nll = 0.0
    count = 0

    for i in range(n_tokens - 1 if only_last else 1, n_tokens):
        logits = model._scores[i-1, :]

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        token_id = tokens[i]
        token_prob = probs[token_id]

        nll += -np.log(token_prob + 1e-10)
        count += 1

    return np.exp(nll / count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2.0算法")
    parser.add_argument("--model", type=str, help="模型路径", required=True)
    parser.add_argument("--input", type=str, help="输入 CSV 文件路径")
    parser.add_argument("--output", type=str, help="输出 CSV 文件路径")
    parser.add_argument("--gpu_layers", type=int, default=-1, help="GPU 层数")
    parser.add_argument("--ctx_size", type=int, default=256, help="上下文大小")
    parser.add_argument("--resume_id", type=int, default=None, help="跳过直到指定的 ID (用于断点续传)")
    parser.add_argument("--total", type=int, default=None, help="tqdm 显示的总行数")
    parser.add_argument("--always_compute_ppl", action="store_true", help="tqdm 显示的总行数")

    args = parser.parse_args()

    llm = Llama(
        model_path=args.model,
        verbose=False,
        n_ctx=args.ctx_size,
        logits_all=True,
        n_gpu_layers=args.gpu_layers
    )

    # 注意，这是我在测符号token的时候改的，之前是"每个字是什么"，以及双引号
    backtick = "`"
    before = llm.tokenize(f"<|im_start|>user\n{backtick}".encode("utf-8"), special=True)
    after = llm.tokenize(f"{backtick}每个字符是什么，以JSON数组回复<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n[\"".encode("utf-8"), special=True)
    after_no_json = llm.tokenize(f"{backtick}每个字符是什么，以JSON数组回复<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".encode("utf-8"), special=True)
    eos_id = llm.token_eos()
    last_progress = args.resume_id

    input_file = args.input or "tokens_preview.csv"
    output_file = args.output or "black_hole.csv"

    with open(input_file, mode='r', encoding='utf-8', errors='ignore') as infile:
        reader = csv.DictReader(infile)

        with open(output_file, mode='a' if last_progress else 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=["ID", "Text", "Split", "PPL"])
            if not last_progress:
                writer.writeheader()

            for row in tqdm(reader, total = args.total):
                if int(row['Delete']): continue

                if last_progress:
                    if int(row["ID"]) == last_progress:
                        last_progress = None
                    continue

                completion_tokens = []
                input = [*before, int(row["ID"]), *after]
                first_token = llm.tokenize(row["Text"][0].encode("utf-8"))[0]
                res = None

                llm._logits_all = False
                for token in llm.generate(
                        tokens=input,
                        temp=0,
                        present_penalty=1.5,
                ):
                    completion_tokens.append(token)

                    if first_token != None and token != first_token:
                        res = "!EARLY_STOP "+llm.detokenize(completion_tokens).decode("utf-8", errors="ignore")
                        break
                    first_token = None

                    if token == eos_id or args.ctx_size - len(completion_tokens) - len(input) == 1:
                        res = "[\""+llm.detokenize(completion_tokens).decode("utf-8", errors="ignore")
                        try:
                            data = json.loads(res)
                            match = "".join(data) == row["Text"]
                            if match:
                                res = "True"
                        except:
                            pass
                        break

                row_out = {
                    "ID": row["ID"],
                    "Text": row["Text"],
                    "Split": res,
                    "PPL": -1
                }

                if args.always_compute_ppl or res != "True":
                    input = [
                        *before,
                        int(row["ID"]),
                        *after_no_json,
                        *llm.tokenize(json.dumps(list(row["Text"]), ensure_ascii=False).encode("utf-8"))
                    ]

                    llm._logits_all = True
                    ppl = calculate_ppl(llm, input, only_last=True)
                    row_out["PPL"] = ppl

                writer.writerow(row_out)

