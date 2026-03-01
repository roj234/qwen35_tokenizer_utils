
def make_bpe_mapping():
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

enc_map, dec_map = make_bpe_mapping()

def decode_token(s: str) -> bytearray:
    byte_data = bytearray()
    for char in s:
        if char in dec_map:
            byte_data.append(dec_map[char])
        else:
            byte_data += char.encode("utf-8")

    return byte_data
