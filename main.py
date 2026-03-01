
import argparse
import re
import json
import csv
from collections import OrderedDict

from tqdm import tqdm
import numpy as np

from bpe_tokenizer import decode_token
from symbol_utils import symbol_whitelist

from typing import Dict, List

from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGUFValueType

def mutate_gguf(
    source_model,
    target_model,
    kv_from_model=None,
    keys_to_copy=None,
    kv_overrides=None,
    keys_to_remove=None
):
    print(f"[*] 正在读取目标模型 (Base): {source_model}")
    reader = GGUFReader(source_model)

    arch_val = reader.get_field("general.architecture")
    arch = str(arch_val.contents())
    writer = GGUFWriter(target_model, arch=arch)

    if kv_overrides is None: kv_overrides = {}

    if keys_to_copy is not None:
        print(f"[*] 正在读取属性来源模型 (Source): {kv_from_model}")
        source_reader = GGUFReader(kv_from_model)
        for key in keys_to_copy:
            kv = source_reader.get_field(key)
            if kv:
                kv_overrides[key] = {
                    "value": kv.contents(),
                    "type": kv.types[0]
                }

    print("[*] 正在合并元数据...")

    for kv in reader.fields.values():
        name = kv.name
        if name.startswith("GGUF."): continue
        if keys_to_remove and name in keys_to_remove:
            print(f"  [删除] -> {key}")
            continue
        if name in kv_overrides: continue

        writer.add_key_value(name, kv.contents(), kv.types[0])

    for key, value in kv_overrides.items():
        writer.add_key_value(key, value["value"], value["type"])
        print(f"  [新增] -> {key}")

    total_bytes = 0

    for tensor in reader.tensors:
        total_bytes += tensor.n_bytes
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)

    bar = tqdm(desc="Writing", total=total_bytes, unit="byte", unit_scale=True)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
        bar.update(tensor.n_bytes)

    writer.close()


def generate_changeset(data_csv: str, symbol_remove_type: int):
    with open("tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    rows = []
    for k, v in tokenizer_config["model"]["vocab"].items():
        s = decode_token(k)
        rows.append({
            "ID": v,
            "Length": len(s),
            "Text": s.decode("utf-8", errors="replace"),
            "Bytes": bytes(s),
        })

    # 按长度排序 (从长到短)
    rows.sort(key=lambda x: x["Length"], reverse=True)
    print(f"tokens total (no special): {len(rows)}")

    i = 0
    j = 0

    def get_token_type(token):
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', token))
        if has_chinese:
            return 1

        if symbol_remove_type == 0:
            return 0

        if re.match(r'^\s+$', token):
            # 如果是纯空格，且长度不在 [1, 2, 4, 8] 之中，则删除 (变成 Reserved)
            if len(token) not in [1, 2, 4, 8]:
                return 3

            return 3 if len(set(token)) != 1 else 0

        if len(token.strip()) == 1:
            return 0

        if token.strip() in symbol_whitelist:
            return 0

        if re.match(r'^[ `~!@#$%^&*()_+\-=[\]\\{}|:";\',./<>?]+$', token):
            if len(token) > 8:
                return 3

            unique_chars = len(set(token))
            if unique_chars == 1:
                return 3 if len(token) > 4 else 2

            if symbol_remove_type == 2:
                # DELETE
                return 3

            return 2 # CHECK_PPL

        return 0

    with open(data_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Text", "Delete"])
        writer.writeheader()

        for item in rows:
            token_type = get_token_type(item["Text"])
            if token_type == 0:
                continue

            if token_type == 1 and len(item["Text"]) < 4:
                continue

            writer.writerow({
                "ID": item["ID"],
                "Text": item["Text"],
                "Delete": int(token_type == 3)
            })

            if token_type != 3:
                i += 1
            else:
                j += 1

    print(f"ready to check: {i}, ready to remove {j}")

def remove_tokens(data_csvs: List[str]):
    # From HF Repo
    with open("tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    vocab = tokenizer_config["model"]["vocab"]
    merges = tokenizer_config["model"]["merges"]

    to_remove = {}
    replace_with_reserved = False

    def rm(token):
        if token in to_remove:
            return
        to_remove[token] = f"<|RESERVED_{len(to_remove)}|>"

    # 读取CSV
    for data_csv in data_csvs:
        with open(data_csv, mode='r', encoding='utf-8', errors='ignore') as infile:
            reader = csv.DictReader(infile)

            for row in reader:
                if delete_criteria(row):
                    rm(int(row["ID"]))

    copy_vocab = OrderedDict()
    removed_vocab = dict()
    for token, id in vocab.items():
        if id in to_remove:
            removed_vocab[token] = 1
            token = to_remove[id]

        copy_vocab[token] = id

    if replace_with_reserved:
        tokenizer_config["model"]["vocab"] = copy_vocab

    copy_merges = []
    for merge in merges:
        if merge.replace(" ", "") in removed_vocab:
            print("remove merge rule "+decode_token(merge).decode("utf-8", errors="replace"))
        else:
            copy_merges.append(merge)

    tokenizer_config["model"]["merges"] = copy_merges

    print(f"{len(to_remove)} tokens removed")

    with open("tokenizer_patched.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent='\t')

def patch_gguf(source_model: str, target_model: str):
    with open("tokenizer_patched.json", "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    vocab = tokenizer_config["model"]["vocab"]
    merges = tokenizer_config["model"]["merges"]

    reader = GGUFReader(source_model)

    ggml_tokens: list[str] = []

    for kv in reader.fields.values():
        name = kv.name

        # We dont change "tokenizer.ggml.token_type"
        if not (name == "tokenizer.ggml.tokens" or name == "tokenizer.ggml.merges"):
            continue

        if name == "tokenizer.ggml.tokens":
            ggml_tokens = kv.contents()

            for k, v in vocab.items():
                ggml_tokens[v] = k

    mutate_gguf(
        source_model=source_model,
        target_model=target_model,

        kv_overrides={
            "tokenizer.ggml.tokens": {
                "value": ggml_tokens,
                "type": GGUFValueType.ARRAY
            },
            "tokenizer.ggml.merges": {
                "value": merges,
                "type": GGUFValueType.ARRAY
            },
            "tokenizer.patched_by": {
                "value": "roj234/qwen35_tokenizer_utils",
                "type": GGUFValueType.STRING
            }
        }
    )

# 最高Last Token PPL以保留token
LAST_TOKEN_PPL_MAX = 1.01

def delete_criteria(row: Dict[str, str]):
    """
    是否删除一行
    :param row:
    :return:
    """

    is_delete = row.get("Delete", "0")
    if is_delete == "True" or is_delete == "1":
        return True

    embedding_v1_score = row.get("Label", "OK")
    if embedding_v1_score == "CRITICAL": return True

    split_word_v2_score = row.get("Split", "True")
    if split_word_v2_score != "True":
        v2_ppl = float(row.get("PPL", "-1"))
        if v2_ppl > LAST_TOKEN_PPL_MAX:
            return True

    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenizer 处理工具")

    def list_of_strings(arg):
        return arg.split(',')

    # 必须参数：运行哪个阶段
    parser.add_argument("phase", type=int, choices=[1, 2, 3],
                        help="运行阶段: 1-生成待检查的tokens列表, 2-生成分词器配置, 3-应用Patch到GGUF")

    # 文件路径参数
    parser.add_argument("--source", type=str, default="Qwen3.5-35B-A3B-MXFP4_MOE.gguf",
                        help="原始 GGUF 模型路径")
    parser.add_argument("--output", type=str, default="patched_model.gguf",
                        help="生成的 GGUF 模型路径")
    parser.add_argument("--csv", type=list_of_strings, default="tokens_preview.csv",
                        help="CSV 文件")

    # 逻辑参数
    parser.add_argument("--symbol-remove-type", type=int, choices=[0, 1, 2], default=0,
                        help="0=不删除符号token，1=计算PPL，2=按白名单保留符号token")
    parser.add_argument("--ppl-max", type=float, default=1.005,
                        help="最高 Last Token PPL 阈值")
    args = parser.parse_args()

    if args.phase == 1:
        generate_changeset(args.csv[0], args.symbol_remove_type)
    if args.phase == 2:
        LAST_TOKEN_PPL_MAX = args.ppl_max
        remove_tokens(args.csv)
    if args.phase == 3:
        patch_gguf(args.source, args.output)
