---
name: evaluate-retrieval
description: Çıkarılmış öznitelikleri protokole göre değerlendirmek için kullan — Recall@K, mAP, açık küme (katalogda yok) analizi, kırılımlar ve galeriler.
---

# Değerlendirme

`docs/EVALUATION_PROTOCOL.md`'yi oku; tanımların tek kaynağı orasıdır.

1. Komut: `python scripts/evaluate.py --features data/features/<model>/test --config configs/eval/default.yaml`.
2. Kapalı küme:
   - Galeriyi `IndexFlatIP`'e ekle, her sorgu için `search_k` sonuç al.
   - Recall@1/5/10/20 ve mAP hesapla. Sorgu başına sonucu `per_query.parquet`'e yaz.
3. Açık küme ("katalogda yok"):
   - İki skor: `abs = s1`, `margin = s1 − s2` (s2 = farklı match_key'e sahip ilk sonuç).
   - Kapalı küme sorguları "bilinen", openset sorguları "bilinmeyen" → AUROC (her iki skor için).
   - τ'yu **val_internal**'da %95 kapsamaya göre seç, test'te uygula: kapsama, kabul edilenlerde top-1 doğruluk, açık küme sorgularında yanlış eşleşme oranı.
   - Kapsama–kesinlik eğrisi → `coverage_curve.csv` + şekil.
4. Kırılımlar: occlusion, scale, zoom_in, viewpoint, kategori bazında R@1 (sorgu ürününün özelliğine göre). Örnek sayısı 30'dan az olan grupları işaretle.
5. Verimlilik: gömme boyutu, sorgu başına ms (warmup sonrası medyan), indeks MB.
6. Galeriler: en iyi ve en kötü örneklerden `gallery_examples` kadar (sorgu + top-5, doğrular yeşil çerçeve) → `figures/`.
7. Hepsi `outputs/results/<run_id>/` altına: `config.yaml`, `git_hash.txt`, `metrics.json`. Sonra `log-experiment` skill'ini çalıştır.
8. Sonuçları kullanıcıya Türkçe, kısa bir tabloyla özetle; beklenmedik bir sayı varsa (ör. R@1 > R@5) dur ve hatayı ara.
