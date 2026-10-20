"""Helper bersama: kode bahasa, format input NLLB, terjemahan, dan loader."""
import json

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_MODEL = "facebook/nllb-200-distilled-600M"

# ind_Latn sudah ada di NLLB. rej_Latn adalah token kustom (bahasa Rejang
# tidak ada dalam daftar 200 bahasa NLLB), ditambahkan ke vocabulary saat training.
TAG = {"rej": "rej_Latn", "ind": "ind_Latn"}


def build_example(tok, src, tgt, src_text, tgt_text, max_len=128):
    """Format NLLB: input = [lang_src] X [eos], label = [lang_tgt] Y [eos]."""
    src_ids = tok(src_text, add_special_tokens=False,
                  truncation=True, max_length=max_len - 2).input_ids
    tgt_ids = tok(tgt_text, add_special_tokens=False,
                  truncation=True, max_length=max_len - 2).input_ids
    input_ids = [tok.convert_tokens_to_ids(TAG[src])] + src_ids + [tok.eos_token_id]
    labels = [tok.convert_tokens_to_ids(TAG[tgt])] + tgt_ids + [tok.eos_token_id]
    return {"input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels}


@torch.no_grad()
def translate(model, tok, texts, src, tgt, batch_size=16,
              num_beams=4, max_new_tokens=128):
    """Terjemahkan daftar kalimat dari src ke tgt."""
    device = next(model.parameters()).device
    src_id = tok.convert_tokens_to_ids(TAG[src])
    tgt_id = tok.convert_tokens_to_ids(TAG[tgt])
    outputs = []
    for i in range(0, len(texts), batch_size):
        seqs = []
        for t in texts[i:i + batch_size]:
            ids = tok(t, add_special_tokens=False,
                      truncation=True, max_length=126).input_ids
            seqs.append([src_id] + ids + [tok.eos_token_id])
        batch = tok.pad({"input_ids": seqs}, return_tensors="pt").to(device)
        gen = model.generate(input_ids=batch["input_ids"],
                             attention_mask=batch["attention_mask"],
                             forced_bos_token_id=tgt_id,
                             num_beams=num_beams,
                             max_new_tokens=max_new_tokens)
        outputs.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return outputs


def load(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    return tok, model


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
