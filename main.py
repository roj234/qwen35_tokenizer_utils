
import sys

if len(sys.argv) == 1:
    print("""
        read README.md or throw me to a LLM
""")
    exit()

options = int(sys.argv[1]) # 1, 2, 3

REFERENCE_MODEL = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
SOURCE_MODEL = "Qwen3.5-35B-A3B-Q4_K_M.gguf"
OUTPUT_MODEL = "final_model.gguf"      # 生成的结果

import re
import io
import json
import csv
from collections import OrderedDict

from tqdm import tqdm
import numpy as np
import pandas as pd

LLAMA_POOLING_TYPE_NONE = 0
LLAMA_POOLING_TYPE_MEAN = 1
LLAMA_POOLING_TYPE_CLS = 2
LLAMA_POOLING_TYPE_LAST = 3
LLAMA_POOLING_TYPE_RANK = 4
from llama_cpp import Llama
from llama_cpp.llama_embedding import LlamaEmbedding

#from llama_cpp._ggml import libggml
#libggml.ggml_backend_load("ggml-cuda.dll".encode("utf-8"))
#libggml.ggml_backend_load("ggml-cpu-alderlake.dll".encode("utf-8"))

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def analyze_semantic_alignment(tokenizer, model, text, token_id):
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
        "alignment": similarity,
    }

def bytes_to_unicode():
    """
    生成一个字典，映射 0..255 字节到可见的 Unicode 字符。
    这是为了让 BPE 算法在处理词表时，不会遇到像空格、换行符这种不可见或有歧义的字符。
    """
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    # 2. 对于 0-255 之间不在上述范围内的“危险字节”（如控制字符、空格、换行）
    # 将它们映射到 256 之后不常用的 Unicode 区域
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs)), dict(zip(cs, bs))

enc_map, dec_map = bytes_to_unicode()

def decode_token(s: str) -> bytearray:
    byte_data = bytearray()
    for char in s:
        if char in dec_map:
            byte_data.append(dec_map[char])
        else:
            byte_data += char.encode("utf-8")

    return byte_data

def calculate_ppl(model, text):
    tokens = [id] if id else model.tokenize(text.encode("utf-8"))
    n_tokens = len(tokens)

    model.reset()
    model.eval(tokens)

    # 3. 计算负对数似然 (NLL)
    # PPL = exp(平均 NLL)
    nll = 0.0
    count = 0

    # 逐个 token 计算概率（从第 2 个 token 开始预测）
    for i in range(1, n_tokens):
        # 得到预测第 i 个 token 时的 logits (基于前 i-1 个 token)
        # 注意：llama.cpp 的 eval 是增量的，为了简单起见，这里我们用简化的概率获取方式
        # 在实际高性能测试中，通常使用 llama-cpp 自带的 ppl 示例工具

        # 获取第 i-1 位置输出的 logits
        logits = model._scores[i-1, :]

        # Softmax 归一化
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # 目标 token 的概率
        token_id = tokens[i]
        token_prob = probs[token_id]

        # 累加负对数概率
        nll += -np.log(token_prob + 1e-10)
        count += 1

    ppl = np.exp(nll / count)
    return ppl

# Phase 1 generate Aligment
if options == 1 and __name__ == "__main__":
    with open("tokenizer_qwen3.5.json", "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    rows = []
    for k, v in tokenizer_config["model"]["vocab"].items():
        s = decode_token(k)
        rows.append({
            "ID": v,
            "Length": len(s),
            "Text": s.decode("utf-8", errors="replace"),
            "Bytes": bytes(s),
            "Raw_Key": k
        })

    # 按长度排序 (从长到短)
    rows.sort(key=lambda x: x["Length"], reverse=True)
    i = 0
    for item in rows:
        if len(item["Text"]) < 4:
            rows = rows[:i]
            break
        i += 1

    print(f"tokens to check: {len(rows)}")

    tokenizer = Llama(
        model_path=REFERENCE_MODEL,
        vocab_only=True,
        verbose=False
    )

    llm = LlamaEmbedding(
        model_path=SOURCE_MODEL,
        verbose=False,
        embeddings=True,
        pooling_type=LLAMA_POOLING_TYPE_LAST,
        n_ctx=512,
        n_gpu_layers=20
    )

    print(llm.tokenize("给我写一个能开灯的简易程序".encode("utf-8")))
    print(llm.detokenize([188317]))

    fieldnames = ["ID", "Text", "alignment", "Label"]
    with open("tokens_alignment.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in tqdm(rows):
            res = analyze_semantic_alignment(tokenizer, llm, item["Bytes"], item["ID"])
            if not res:
                writer.writerow({
                    "ID": item["ID"],
                    "Text": item["Text"],
                    "alignment": -1,
                    "Label": "Unknown",
                })
                continue

            status = "BAD" if res['alignment'] < 0.6 else "OK"
            if res['alignment'] < 0.5: status = "CRITICAL"

            writer.writerow({
                "ID": item["ID"],
                "Text": item["Text"],
                "alignment": res['alignment'],
                "Label": status
            })


def is_confusing_token(text):
    # 处理纯空格 Token
    if re.match(r'^[!@#$%^&*()_+-=[\]\\{}|:";\', ./<>?]+$', text):
        # 如果是纯空格，且长度不在 [1, 2, 4, 8] 之中，则删除 (变成 Reserved)
        if len(text) not in [1, 2, 4, 8]:
            return True
        return False # 保留 1, 2, 4, 8 个空格

    return False

def is_removeable_token(text):
    # 1. 定义排除范围：如果包含泰语或俄语字符，直接返回 False
    # \u0e00-\u0e7f: 泰语
    # \u0400-\u04ff: 西里尔字母 (俄语等)
    if re.search(r'[\u0e00-\u0e7f\u0400-\u04ff]', text):
        return False

    # 2. 检查是否包含中文字符
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
    if has_chinese:
        return True

    # 3. 检查是否全是“特殊符号”
    # 我们移除掉所有 Unicode 字母和数字 (\w) 以及空白符 (\s)
    # 如果剩下的内容不为空，且原本不含普通英文字母，则视为特殊符号
    # 这里用一种更直观的方法：匹配常见的标点、符号、Emoji
    # 或者判断：如果不含任何字母数字，但包含内容，就是纯符号
    clean_text = re.sub(r'[\w\s]', '', text)
    if len(text) > 0 and len(clean_text) == len(text):
        return True

    return False

def generate_reserved_dict(csv_path):
    # 读取CSV
    df = pd.read_csv(csv_path)

    reserved_dict = {}
    counter = 0

    for idx, row in df.iterrows():
        token_id = int(row['ID'])
        text = str(row['Text'])
        label = row['Label']

        #if is_confusing_token(text):
        #    reserved_dict[token_id] = f"<|RESERVED_{counter}|> #{text}"
        #    counter += 1

        if label == 'CRITICAL' and is_removeable_token(text):
            reserved_dict[token_id] = f"<|RESERVED_{counter}|>"
            counter += 1

    return reserved_dict

# Phase 2 dummies tokens
if options == 2 and __name__ == "__main__":
    # From HF Repo
    with open("tokenizer_qwen3.5.json", "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    vocab = tokenizer_config["model"]["vocab"]
    merges = tokenizer_config["model"]["merges"]

    to_remove = generate_reserved_dict('tokens_alignment.csv')

    print(f"{len(to_remove)} tokens will be removed")

    copy_vocab = OrderedDict()
    removed_vocab = dict()
    for token, id in vocab.items():
        if id in to_remove:
          removed_vocab[token] = 1
    #      token = to_remove[id]

    #    copy_vocab[token] = id

    #tokenizer_config["model"]["vocab"] = copy_vocab

    copy_merges = []
    for merge in merges:
        if merge.replace(" ", "") in removed_vocab:
            print("remove merge rule "+decode_token(merge).decode("utf-8", errors="replace"))
        else:
            copy_merges.append(merge)

    tokenizer_config["model"]["merges"] = copy_merges

    with open("tokenizer_qwen3.5_new.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent='\t')



from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGUFValueType

def mutate_gguf(
    source_model,
    target_model,
    kv_from_model=None,
    keys_to_copy=None,
    manual_kv_overrides=None,
    keys_to_remove=None
):
    print(f"[*] 正在读取目标模型 (Base): {source_model}")
    reader = GGUFReader(source_model)

    arch_val = reader.get_field("general.architecture")
    arch = str(arch_val.contents())
    writer = GGUFWriter(target_model, arch=arch)

    # 1. 准备要复制的属性池
    kv_overrides = {}
    if keys_to_copy is not None:
        print(f"[*] 正在读取属性来源模型 (Source): {kv_from_model}")
        source_reader = GGUFReader(kv_from_model)
        for key in keys_to_copy:
            kv = source_reader.get_field(key)
            if kv:
                kv_overrides[key] = kv
                print(f"  [克隆准备] 读取到 {key}")
            else:
                print(f"  [警告] 源文件中未找到 Key: {key}")
    else:
        keys_to_copy = {}

    # 2. 处理元数据 (KV Pairs)
    print("[*] 正在合并元数据...")

    # 记录已经处理过的覆盖键，避免重复写入
    processed_overrides = set()

    for kv in reader.fields.values():
        name = kv.name

        if name.startswith("GGUF."):
            continue

        # A. 如果在删除列表中，跳过
        if keys_to_remove and name in keys_to_remove:
            print(f"  [删除] -> {name}")
            continue

        # B. 如果在覆盖列表中，从 Source 池中取值
        if name in kv_overrides:
            new_kv = kv_overrides[name]
            writer.add_key_value(name, new_kv.contents(), new_kv.types[0])
            processed_overrides.add(name)
            print(f"  [覆盖] -> {name} (来自 Source)")
            continue

        # 先不管了
        if name in manual_kv_overrides:
            new_kv = manual_kv_overrides[name]
            writer.add_key_value(name, new_kv["value"], new_kv["type"])
            processed_overrides.add(name)
            print(f"  [覆盖] -> {name} (来自 Manual)")
            continue

        # C. 保持原样写入
        writer.add_key_value(name, kv.contents(), kv.types[0])

    # D. 处理那些在 Target 中不存在，但在 Source 中存在需要新增的键
    for key in keys_to_copy:
        if key not in processed_overrides and key in kv_overrides:
            new_kv = kv_overrides[key]
            writer.add_key_value(new_kv.name, new_kv.contents(), new_kv.types[0])
            print(f"  [新增] -> {key} (来自 Source)")

    total_bytes = 0

    for tensor in reader.tensors:
        total_bytes += tensor.n_bytes
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)

    bar = tqdm(desc="Writing", total=total_bytes, unit="byte", unit_scale=True)

    # 4. 执行写入
    print(f"[*] 正在保存到: {target_model}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
        bar.update(tensor.n_bytes)

    writer.close()
    print("[+] 成功！")

# Phase 3 Apply GGUF Patch (Dynamic)
if options == 3 and __name__ == "__main__":
    with open("tokenizer_qwen3.5_new.json", "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    vocab = tokenizer_config["model"]["vocab"]
    merges = tokenizer_config["model"]["merges"]

    reader = GGUFReader(SOURCE_MODEL)

    ggml_tokens: list[str] = []
    ggml_merges: list[str] = merges

    for kv in reader.fields.values():
        name = kv.name

        # We dont change "tokenizer.ggml.token_type"
        if not (name == "tokenizer.ggml.tokens" or name == "tokenizer.ggml.merges"):
            continue

        s = kv.contents()
        if name == "tokenizer.ggml.tokens":
            ggml_tokens = s
        #else:
        #    ggml_merges = s

        for k, v in vocab.items():
            ggml_tokens[v] = k

        with open(name+".json", 'w', encoding='utf8') as f:
            json.dump(s, f, ensure_ascii=False, indent='\t')

    mutate_gguf(
        source_model=SOURCE_MODEL,
        target_model=OUTPUT_MODEL,

        manual_kv_overrides={
            "tokenizer.ggml.tokens": {
                "value": ggml_tokens,
                "type": GGUFValueType.ARRAY
            },
            "tokenizer.ggml.merges": {
                "value": ggml_merges,
                "type": GGUFValueType.ARRAY
            },
        }
    )

# Phase 4 Apply GGUF Patch (Static)