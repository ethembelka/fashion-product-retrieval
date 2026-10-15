---
name: extract-features
description: Hazır bir omurgayla (ResNet, CLIP, SigLIP, DINOv2) galeri, sorgu ve açık küme kırpıntılarının özniteliklerini çıkarmak için kullan.
---

# Öznitelik çıkarma

1. `configs/models/backbones.yaml`'dan modeli seç. Yeni bir model eklenecekse sadece `src/fashret/models/` altına `FeatureExtractor`'ı uygulayan bir sınıf yaz ve `registry.py`'ye kaydet; değerlendirme koduna dokunma.
2. Komut: `python scripts/extract_features.py --model <ad> --split <train_fit|val_internal|test>`.
3. Kurallar:
   - `model.eval()`, `torch.inference_mode()`, fp16 (6 GB VRAM). Bellek hatasında batch'i yarıya indir ve kullanıcıya bildir.
   - Çıktı L2-normalize edilir.
   - Her liste (gallery, queries, openset) için `data/features/<model>/<split>/<liste>_emb.npy` + `<liste>_meta.parquet` (item_id, match_key, öznitelikler) yaz. Satır sırası iki dosyada aynı olmalı; bunu assert et.
   - Zaten var olan dosyayı `--overwrite` olmadan yeniden hesaplama.
4. Sonunda: kaç görüntü, süre, görüntü/saniye, VRAM tepe değeri yazdır.
5. Tablo 1 için dört omurganın hepsini `val_internal` ve `test` üzerinde çıkar. (Eğitim yok; her biri dakikalar sürer.)
