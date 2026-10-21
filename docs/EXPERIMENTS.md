# Deney Kaydı

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
