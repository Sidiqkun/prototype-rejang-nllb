"""Membersihkan korpus paralel dan membaginya menjadi train/valid/test.

Input (pilih salah satu):
  --tsv FILE                  satu file TSV: kolom1=rejang, kolom2=indonesia
  --rej FILE --ind FILE       dua file teks dengan jumlah baris yang sama
"""
import argparse
import json
import os
import random
import re
import unicodedata


def norm(s):
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s).strip()


def load_pairs(args):
    if args.tsv:
        with open(args.tsv, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    yield parts[0], parts[1]
    else:
        with open(args.rej, encoding="utf-8") as fr, \
             open(args.ind, encoding="utf-8") as fi:
            for r, i in zip(fr, fi, strict=True):
                yield r.rstrip("\n"), i.rstrip("\n")


def clean(pairs, max_chars=400, max_ratio=3.0):
    seen, out = set(), []
    for r, i in pairs:
        r, i = norm(r), norm(i)
        if not r or not i:
            continue
        if len(r) > max_chars or len(i) > max_chars:
            continue
        ratio = max(len(r), len(i)) / max(1, min(len(r), len(i)))
        if ratio > max_ratio:  # pasangan dengan panjang tidak seimbang
            continue
        key = (r.lower(), i.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({"rej": r, "ind": i})
    return out


def split(rows, valid_frac, test_frac, seed):
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    n = len(rows)
    n_test = max(1, round(n * test_frac))
    n_valid = max(1, round(n * valid_frac))
    return {
        "test": rows[:n_test],
        "valid": rows[n_test:n_test + n_valid],
        "train": rows[n_test + n_valid:],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tsv")
    p.add_argument("--rej")
    p.add_argument("--ind")
    p.add_argument("--out_dir", default="data/processed")
    p.add_argument("--valid_frac", type=float, default=0.05)
    p.add_argument("--test_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if not a.tsv and not (a.rej and a.ind):
        p.error("pakai --tsv, atau --rej dan --ind")

    rows = clean(load_pairs(a))
    if len(rows) < 10:
        raise SystemExit(f"Hanya {len(rows)} pasangan valid, minimal 10.")

    splits = split(rows, a.valid_frac, a.test_frac, a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    for name, data in splits.items():
        with open(os.path.join(a.out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name:6s}: {len(data)} pasangan")


if __name__ == "__main__":
    main()
