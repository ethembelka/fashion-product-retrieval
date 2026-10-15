---
name: log-experiment
description: Tamamlanan bir öznitelik çıkarma, eğitim ya da değerlendirmeyi docs/EXPERIMENTS.md deney kaydına işlemek için kullan.
---

# Deney kaydı

1. `outputs/results/<run_id>/metrics.json` ve `config.yaml` dosyalarını oku.
2. `docs/EXPERIMENTS.md` tablosuna bir satır ekle: run_id, tarih, omurga, varyant (zeroshot / head), ayrım, ortam (yerel 3060 / Colab T4), R@1, R@10, mAP, AUROC (margin), kısa not.
3. Sayıları metrics.json'dan kopyala; yuvarlama 3 ondalık.
4. `python scripts/make_tables.py` ile `reports/tables/` altındaki karşılaştırma tablolarını yenile.
5. Commit: `exp: <run_id> sonuçları`.
