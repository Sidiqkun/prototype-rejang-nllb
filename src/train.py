"""Fine-tuning NLLB-200 untuk Rejang <-> Indonesia (dua arah).

Jalankan lokal:   python src/train.py --epochs 1
Jalankan multi-GPU: torchrun --standalone --nproc_per_node=2 src/train.py
"""
import argparse
import os

import torch
from datasets import Dataset
from transformers import (DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, set_seed)

from common import BASE_MODEL, TAG, build_example, load, read_jsonl


def expand(rows, tok, max_len):
    """Setiap pasangan dipakai dua kali: rej->ind dan ind->rej."""
    examples = []
    for r in rows:
        for src, tgt in (("rej", "ind"), ("ind", "rej")):
            examples.append(build_example(tok, src, tgt, r[src], r[tgt], max_len))
    return Dataset.from_list(examples)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default=BASE_MODEL)
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--output_dir", default="models/nllb-rejang")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    set_seed(a.seed)

    tok, model = load(a.model_name)

    # Tambahkan token bahasa Rejang lalu inisialisasi embedding-nya
    # dari embedding ind_Latn agar training mulai dari titik yang wajar.
    tok.add_tokens([TAG["rej"]], special_tokens=True)
    model.resize_token_embeddings(len(tok))
    emb = model.get_input_embeddings().weight
    with torch.no_grad():
        emb[tok.convert_tokens_to_ids(TAG["rej"])] = \
            emb[tok.convert_tokens_to_ids(TAG["ind"])]

    train_rows = read_jsonl(os.path.join(a.data_dir, "train.jsonl"))
    valid_rows = read_jsonl(os.path.join(a.data_dir, "valid.jsonl"))
    train_ds = expand(train_rows, tok, a.max_len)
    valid_ds = expand(valid_rows, tok, a.max_len)
    print(f"Contoh training: {len(train_ds)}, validasi: {len(valid_ds)}")

    collator = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100)
    args = Seq2SeqTrainingArguments(
        output_dir=a.output_dir,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size,
        learning_rate=a.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=a.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(a.output_dir)
    if trainer.is_world_process_zero():
        tok.save_pretrained(a.output_dir)
        print(f"Model tersimpan di {a.output_dir}")


if __name__ == "__main__":
    main()
