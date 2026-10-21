# Deney Kaydı

## Plan (süre tahminleri, CLAUDE.md kuralı)
- 2026-10-21 Adım 4 — projeksiyon başlığı (siglip_vitb16, InfoNCE, önbellek öznitelikleri): train_fit 194.603 satır (60.894 user + 133.709 shop), 16.374 ürün;
  val_internal üzerinde 3 turluk deneme 0,5 s/tur (20k satır) → tam veri ≈ 5 s/tur, 30 tur + erken durdurma ≈ 3 dk, yerel 3060, VRAM < 1 GB. Colab gerekmez.
- Adım 4b — ArcFace karşılaştırması: aynı veri, aynı süre (≈ 3 dk). [yapıldı]
- 2026-10-22 Adım 5 — kısmi fine-tune (SigLIP son 4 blok + attn_pool + başlık, 35,4 M parametre; InfoNCE, P=64 K=2, AMP, 8 tur, lr 1e-5/1e-3):
  tur başına 32.748 görüntü ileri+geri (son 4 blok) + val_internal 20.769 görüntü ileri. L4 tahmini ≈ 2,5–4 dk/tur → 8 tur 20–35 dk; zip açma 5 dk; test+val çıkarımı 3 dk → toplam ≈ 40 dk.
  Colab L4, tam otomatik (notebooks/colab_finetune.ipynb, sonunda runtime.unassign). Yerel smoke test (64 ürün, 2 tur) geçti. Kullanıcı onayı: 2026-10-22.

Her deney bitince `log-experiment` skill'i ile bir satır eklenir. Sayılar `outputs/results/<run_id>/metrics.json` dosyasından alınır, elle yazılmaz.

| run_id | tarih | omurga | varyant | ayrım | ortam | R@1 | R@10 | mAP | AUROC (margin) | not |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261021_clip_vitb16_zeroshot_test_s0 | 2026-10-21 | clip_vitb16 | zeroshot | test | yerel 3060 | 0.079 | 0.172 | 0.058 | 0.554 | strict 0.562; doğru↔yanlış margin 0.789; kat.tut@5 0.354 |
| 20261021_clip_vitb16_zeroshot_val_internal_s0 | 2026-10-21 | clip_vitb16 | zeroshot | val_internal | yerel 3060 | 0.077 | 0.184 | 0.055 | 0.529 | strict 0.529; doğru↔yanlış margin 0.730; kat.tut@5 0.327 |
| 20261021_dinov2_vitb14_zeroshot_test_s0 | 2026-10-21 | dinov2_vitb14 | zeroshot | test | Colab L4 | 0.065 | 0.136 | 0.047 | 0.510 | strict 0.528; doğru↔yanlış margin 0.774; kat.tut@5 0.415 |
| 20261021_dinov2_vitb14_zeroshot_val_internal_s0 | 2026-10-21 | dinov2_vitb14 | zeroshot | val_internal | Colab L4 | 0.072 | 0.163 | 0.053 | 0.528 | strict 0.530; doğru↔yanlış margin 0.733; kat.tut@5 0.409 |
| 20261021_resnet50_zeroshot_test_s0 | 2026-10-21 | resnet50 | zeroshot | test | Colab L4 | 0.088 | 0.205 | 0.062 | 0.497 | strict 0.511; doğru↔yanlış margin 0.747; kat.tut@5 0.422 |
| 20261021_resnet50_zeroshot_val_internal_s0 | 2026-10-21 | resnet50 | zeroshot | val_internal | Colab L4 | 0.103 | 0.252 | 0.077 | 0.498 | strict 0.499; doğru↔yanlış margin 0.743; kat.tut@5 0.404 |
| 20261021_siglip_vitb16_zeroshot_test_s0 | 2026-10-21 | siglip_vitb16 | zeroshot | test | Colab L4 | 0.357 | 0.620 | 0.320 | 0.578 | strict 0.599; doğru↔yanlış margin 0.770; kat.tut@5 0.678 |
| 20261021_siglip_vitb16_zeroshot_val_internal_s0 | 2026-10-21 | siglip_vitb16 | zeroshot | val_internal | Colab L4 | 0.422 | 0.693 | 0.380 | 0.565 | strict 0.569; doğru↔yanlış margin 0.756; kat.tut@5 0.670 |
| 20261022_siglip_vitb16+head_head_val_internal_s0 | 2026-10-22 | siglip_vitb16 | head | val_internal | yerel 3060 | 0.656 | 0.876 | 0.641 | 0.695 | AUROC abs 0.758; doğru↔yanlış margin 0.829; kat.tut@5 0.812; dim 256 |
| 20261022_siglip_vitb16+head_head_test_s0 | 2026-10-22 | siglip_vitb16 | head | test | yerel 3060 | 0.559 | 0.808 | 0.542 | 0.641 | AUROC abs 0.676; doğru↔yanlış margin 0.813; kat.tut@5 0.814; dim 256 |
| 20261022_siglip_vitb16+arcface_arcface_val_internal_s0 | 2026-10-22 | siglip_vitb16 | arcface | val_internal | yerel 3060 | 0.640 | 0.855 | 0.608 | 0.707 | AUROC abs 0.811; doğru↔yanlış margin 0.830; kat.tut@5 0.775; dim 256 |
| 20261022_siglip_vitb16+arcface_arcface_test_s0 | 2026-10-22 | siglip_vitb16 | arcface | test | yerel 3060 | 0.567 | 0.804 | 0.529 | 0.666 | AUROC abs 0.700; doğru↔yanlış margin 0.822; kat.tut@5 0.788; dim 256 |
