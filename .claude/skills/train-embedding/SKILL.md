---
name: train-embedding
description: Projeksiyon başlığını ya da omurganın son bloklarını (InfoNCE/ArcFace, P×K örnekleme) eğitmek veya yarıda kalan eğitimi sürdürmek için kullan.
---

# Gömme eğitimi (projeksiyon başlığı ve kısmi fine-tune)

1. Önce Tablo 1 hazır olmalı (eğitimsiz omurga karşılaştırması). En iyi omurgayı `configs/train/head.yaml` içindeki `backbone` alanına yaz.
2. `use_cached_features: true` ise omurgadan geçiş yapılmaz; `train_fit` öznitelikleri `.npy`'den okunur → eğitim yerelde dakikalar sürer, Colab gerekmez.
   Öznitelikler yoksa önce `extract-features` skill'ini `train_fit` için çalıştır.
3. Komut: `python scripts/train_head.py --config configs/train/head.yaml`.
4. Eğitim kuralları:
   - P×K örnekleyici: her batch'te P farklı match_key, her birinden K görüntü (en az bir user + bir shop).
   - Kayıp: InfoNCE (user→shop yönünde), sıcaklık config'ten.
   - Her epoch sonunda `val_internal` üzerinde R@1 ölç; en iyiyi `outputs/checkpoints/<run_id>/best.pt` olarak sakla, `last.pt` ile devam ettirilebilir olsun.
   - Erken durdurma `patience`'a göre. **Test ayrımına eğitim sırasında hiç bakılmaz.**
5. Bitince eğitilmiş başlıkla `test` özniteliklerini dönüştür (`data/features/<backbone>+head/<split>/`) ve `evaluate-retrieval` skill'ine geç.
6. Kayıp ve val R@1 eğrisini `outputs/results/<run_id>/figures/training_curve.png` olarak kaydet.

## Kısmi fine-tune (plan madde 4)
- `use_cached_features: false`, omurganın son `unfreeze_blocks` bloğu açık; diğerleri dondurulmuş.
- 6 GB VRAM: fp16 (AMP), batch 32–48, gerekirse gradyan biriktirme; bellek hatasında batch'i yarıya indir.
- İlk epoch'un süresini ölç, toplam süreyi tahmin et ve kullanıcıya yerel / Kaggle seçeneğini sor.
