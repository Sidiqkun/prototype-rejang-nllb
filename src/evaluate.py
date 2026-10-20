"""Evaluasi model pada data test dengan BLEU dan chrF (kedua arah)."""
import argparse
import json
import os

import sacrebleu
import torch

from common import load, read_jsonl, translate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--test", default="data/processed/test.jsonl")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--batch_size", type=int, default=16)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load(a.model_dir)
    model.to(device).eval()

    rows = read_jsonl(a.test)
    os.makedirs(a.out_dir, exist_ok=True)
    metrics = {}
    for src, tgt in (("rej", "ind"), ("ind", "rej")):
        refs = [r[tgt] for r in rows]
        hyps = translate(model, tok, [r[src] for r in rows], src, tgt, a.batch_size)
        name = f"{src}2{tgt}"
        with open(os.path.join(a.out_dir, f"pred_{name}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(hyps) + "\n")
        metrics[name] = {
            "bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
            "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
            "n": len(rows),
        }

    with open(os.path.join(a.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
