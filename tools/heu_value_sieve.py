import re
import math
import torch
from torch.nn import functional as F


def process(form):
    res = []
    for ch in form:
        res += "[" + ch + "," + ch.upper() + "]"
    return "".join(res) + "$"


def name2re(name):
    possible = [name[:i] for i in range(3, len(name) + 1)]
    processed = map(process, possible)
    reg = re.compile("|".join(processed))
    return reg


def embed(s, t_example):
    res = t_example.new_zeros(300)
    months = ["january", "february", "march", "april", "may", "june",
              "july", "august", "september", "october", "november", "december"]
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    month_re = map(name2re, months)
    day_re = map(name2re, days)

    for i, m in enumerate(month_re):
        if m.match(s):
            res.fill_(math.sin(i * math.pi / 6))
            yield {"type": months[i], "tensor": res}

    for i, m in enumerate(day_re):
        if m.match(s):
            res.fill_(math.sin(i * 2 * math.pi / 7))
            yield {"type": days[i], "tensor": res}

    clk = re.compile(r"(\d{1,2}):(\d{1,2})([aApP][mM])?$")
    m = clk.match(s)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        if m.group(3) is not None and m.group(3)[0] in ['P', 'p']:
            hour += 1
        res.fill_(math.sin((hour / 12 + minute / 720) * math.pi))
        yield {"type": format(hour, '02d') + ':' + format(minute, '02d'), "tensor": res}

    try:
        x = float(s)
        try:
            res.fill_(x / 10)
        except RuntimeError:
            res.fill_(float("+inf"))
        res = 100 * torch.sigmoid(res)
        if not math.isnan(res.sum().item()):
            yield {"type": str(x), "tensor": res}
    except ValueError:
        pass


def value_sieve(ocr_tokens, s_emb, mask_s):
    """
    :param ocr_tokens: list of str, len == 50 >= <num_ocr>
    :param s_emb: [50, 300]
    :param mask_s: int, <num_ocr>
    :return: value_tokens: list of str, len==50 >= <num_val>
    :return: v_emb, [50, 300]
    :return: mask_v, []
    """
    value_tokens = ["<pad>"] * len(ocr_tokens)
    v_emb = s_emb.new_zeros(s_emb.size())
    mask_v = mask_s.new_zeros(())
    for i in range(mask_s):
        res_it = embed(ocr_tokens[i], s_emb)
        for res in res_it:
            value_tokens[mask_v] = res["type"]
            v_emb[mask_v] = res["tensor"]
            mask_v += 1
    return value_tokens, v_emb, mask_v


if __name__ == "__main__":
    s = input("ocr_token: \n")
    t_example = torch.Tensor(7)
    while s != "q":
        for item in embed(s, t_example):
            print(item)

        s = input("ocr_token: \n")
    print("Quited")
