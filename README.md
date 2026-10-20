# Rejang ↔ Indonesia: Korpus Paralel dan Fine-tuning NLLB-200

Prototype sederhana untuk membangun korpus paralel bahasa Rejang–Indonesia,
lalu melakukan fine-tuning `facebook/nllb-200-distilled-600M` dua arah
(Rejang→Indonesia dan Indonesia→Rejang). Bisa dijalankan lokal untuk uji
coba, dan di cluster HPC dengan SLURM untuk training sungguhan.

## Struktur

```
rejang-nllb/
├── data/
│   ├── raw/sample.tsv        # contoh format (data placeholder, BUKAN data asli)
│   └── processed/            # hasil prepare_data.py (train/valid/test .jsonl)
├── src/
│   ├── common.py             # kode bahasa, format input NLLB, fungsi terjemahan
│   ├── prepare_data.py       # pembersihan dan pembagian data
│   ├── train.py              # fine-tuning dengan Seq2SeqTrainer
│   └── evaluate.py           # BLEU dan chrF pada data test
├── scripts/train_hpc.slurm   # job SLURM untuk HPC
├── requirements.txt
└── README.md
```

## Format data

Korpus mentah berupa TSV, satu pasangan per baris, dipisah tab:

```
<kalimat Rejang>	<kalimat Indonesia>
```

Atau dua file teks dengan jumlah baris yang sama (`--rej` dan `--ind`).
Data placeholder di `data/raw/sample.tsv` hanya untuk menguji pipeline.
Ganti dengan korpus asli sebelum training.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # install PyTorch versi CUDA dari cluster bila perlu
```

## Uji coba lokal (smoke test)

```bash
python src/prepare_data.py --tsv data/raw/sample.tsv
python src/train.py --epochs 1 --batch_size 4
python src/evaluate.py --model_dir models/nllb-rejang
```

Hasil dengan data placeholder tidak bermakna. Tujuannya hanya memastikan
seluruh pipeline berjalan dari awal sampai akhir.

## Menjalankan di HPC

1. Di login node, unduh model ke cache (compute node biasanya tanpa internet):
   ```bash
   python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
   n='facebook/nllb-200-distilled-600M'; AutoTokenizer.from_pretrained(n); AutoModelForSeq2SeqLM.from_pretrained(n)"
   ```
2. Letakkan korpus di `data/raw/corpus.tsv`, lalu sesuaikan modul dan partisi
   di `scripts/train_hpc.slurm`.
3. Kirim job:
   ```bash
   mkdir -p logs
   sbatch scripts/train_hpc.slurm
   ```
4. Pantau dengan `squeue -u $USER` dan `tail -f logs/nllb-rejang-<jobid>.out`.

Hasil akhir ada di `results/metrics.json` dan `results/pred_*.txt`.

## Catatan teknis

- **Kode bahasa Rejang.** NLLB-200 tidak memiliki bahasa Rejang. Prototype ini
  memakai token kustom `rej_Latn` yang ditambahkan ke vocabulary. Embedding-nya
  diinisialisasi dari `ind_Latn` supaya training tidak mulai dari nol.
- **Format input.** Input dan label mengikuti format NLLB:
  `[kode_bahasa] teks [</s>]`. Saat inferensi, bahasa target dipaksa sebagai
  token awal output.
- **Data kecil.** Dengan korpus yang sedikit, model cenderung overfit. Pantau
  `eval_loss` dan gunakan epoch yang sedikit bila perlu.
- **Multi-GPU.** `torchrun` menangani DDP secara otomatis. Batch size yang
  tercatat adalah per GPU.

## Keterbatasan dan rencana lanjutan

- Pembersihan masih sederhana (aturan panjang dan duplikat). Belum ada
  deteksi bahasa atau filter kualitas.
- Evaluasi memakai BLEU dan chrF saja. Evaluasi manusia dengan penutur
  Rejang tetap diperlukan.
- Belum ada baseline tanpa fine-tuning, karena NLLB belum mengenal kode `rej_Latn`.
- Pertimbangkan `nllb-200-1.3B` atau `nllb-200-3.3B` jika GPU mencukupi.

## Catatan etika data

Korpus bahasa daerah sebaiknya dibuat bersama komunitas penutur dan dengan
izin yang jelas. Tentukan lisensi data sebelum dipublikasikan.
