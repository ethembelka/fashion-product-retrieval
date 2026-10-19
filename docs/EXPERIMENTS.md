# Deney Kaydı

Her deney bitince `log-experiment` skill'i ile bir satır eklenir. Sayılar `outputs/results/<run_id>/metrics.json` dosyasından alınır, elle yazılmaz.

| run_id | tarih | omurga | varyant | ayrım | ortam | R@1 | R@10 | mAP | AUROC (margin) | not |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261019_clip_vitb16_zeroshot_test_s0 | 2026-10-19 | clip_vitb16 | zeroshot | test | yerel 3060 | 0.079 | 0.172 | 0.058 | — | kat.tut@5 0.354; dim 512 |
| 20261019_clip_vitb16_zeroshot_val_internal_s0 | 2026-10-19 | clip_vitb16 | zeroshot | val_internal | yerel 3060 | 0.077 | 0.184 | 0.055 | — | kat.tut@5 0.327; dim 512 |
| 20261019_dinov2_vitb14_zeroshot_test_s0 | 2026-10-19 | dinov2_vitb14 | zeroshot | test | Colab L4 | 0.065 | 0.136 | 0.047 | — | kat.tut@5 0.415; dim 768 |
| 20261019_dinov2_vitb14_zeroshot_val_internal_s0 | 2026-10-19 | dinov2_vitb14 | zeroshot | val_internal | Colab L4 | 0.072 | 0.163 | 0.053 | — | kat.tut@5 0.409; dim 768 |
| 20261019_resnet50_zeroshot_test_s0 | 2026-10-19 | resnet50 | zeroshot | test | Colab L4 | 0.088 | 0.205 | 0.062 | — | kat.tut@5 0.422; dim 2048 |
| 20261019_resnet50_zeroshot_val_internal_s0 | 2026-10-19 | resnet50 | zeroshot | val_internal | Colab L4 | 0.103 | 0.252 | 0.077 | — | kat.tut@5 0.404; dim 2048 |
| 20261019_siglip_vitb16_zeroshot_test_s0 | 2026-10-19 | siglip_vitb16 | zeroshot | test | Colab L4 | 0.357 | 0.620 | 0.320 | — | kat.tut@5 0.678; dim 768 |
| 20261019_siglip_vitb16_zeroshot_val_internal_s0 | 2026-10-19 | siglip_vitb16 | zeroshot | val_internal | Colab L4 | 0.422 | 0.693 | 0.380 | — | kat.tut@5 0.670; dim 768 |
