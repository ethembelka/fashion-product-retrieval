# Durum
_Son güncelleme: 2026-10-16 (cc)_

## Hazır
- Depo yapısı, mimari, değerlendirme protokolü, skill'ler
- GitHub (private): https://github.com/ethembelka/fashion-product-retrieval — `main` push'landı
- Senkronizasyon hook'u: `.claude/hooks/sync_check.sh` her istekte git durumu + origin/main farkını gösterir
- Veri `data/raw/DeepFashion2/deepfashion2_original_images/` altında, tam olduğu doğrulandı
- Proje önerisi PDF'i hazır (kullanıcı okuyup gönderecek)

## Sıradaki
1. `prepare-data`: kırpıntılar + ayrımlar + resmi test listeleri
2. `extract-features`: 4 omurga, eğitimsiz karşılaştırma (Tablo 1)
3. `evaluate-retrieval`: kapalı + açık küme

## Bekleyen
- Açık tutarsızlıklar (cc incelemesi): kırpma 224 vs `resize_short: 256`; val_internal galerisine shop style=0 dikkat dağıtıcıları eklenmeli (resmi galeri bunları içeriyor); `head.yaml`'da ArcFace/`unfreeze_blocks` yok; Colab defteri `src/train.py` çağırıyor; `requirements.txt`'te `pyarrow`, `pytest` eksik
- Araştırma önerisi (ILIAS + AMES, 1 sayfa) — teslim 1 Kasım
