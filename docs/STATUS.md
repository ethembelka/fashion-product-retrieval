# Durum
_Son güncelleme: 2026-10-19 (cc)_

## Hazır
- Depo yapısı, mimari, değerlendirme protokolü, skill'ler
- GitHub (private): https://github.com/ethembelka/fashion-product-retrieval — `main` push'landı
- Senkronizasyon hook'u: `.claude/hooks/sync_check.sh` her istekte git durumu + origin/main farkını gösterir
- Veri `data/raw/DeepFashion2/deepfashion2_original_images/` altında, tam olduğu doğrulandı
- Proje önerisi teslim edildi (18 Ekim)
- **Adım 0 — kurulum:** `src/fashret/` (config, utils, data, reporting), `pyproject.toml`, `requirements.txt`, 5 pytest testi geçiyor
- **Adım 1 — veri hazırlama tamam** (`scripts/prepare_data.py`):
  - 312.110 train + 52.470 validation kırpıntı (`data/processed/crops/`, 8,2 GB); 96 kutu <32 px atıldı, 0 hata
  - Ayrımlar seed 0, pair_id düzeyinde: train_fit 13.099 / val_internal 1.456 pair_id
  - Resmi test: 12.375/12.377 sorgu, 36.958/36.961 galeri eşlendi (eksikler küçük kutu); 3.137 açık küme sorgusu
  - Rapor: `docs/reports/01_veri_hazirlama.docx` (üretici: `scripts/report_01_data.py`)
  - `data/crops.zip` (`scripts/pack_crops.py`) → kullanıcı Drive'a `CMPE9103_Retrieval/data/` altına yükleyecek

- **Adım 2 kodu hazır:** `src/fashret/models/` (FeatureExtractor, timm, open_clip, registry), `data/dataset.py`, `engine/extractor.py`, `scripts/extract_features.py`; 4 omurga 256 kırpıntıyla yerelde doğrulandı (3060, fp16: ~110–130 img/s, <1 GB VRAM). Colab defteri: `notebooks/colab_extract_features.ipynb` (crops.zip + GitHub token → features/ Drive'a)
- `data/crops.zip` üretildi: 7,72 GB, 364.592 dosya (crops + splits + crop_report), bütünlük testi temiz

- **Değerlendirme kodu hazır (kapalı küme):** `index/faiss_index.py`, `evaluation/closed_set.py` (Recall@K, mAP, kategori tutarlılığı, s1/s2), `evaluation/evaluator.py`, `scripts/evaluate.py`; 4 elle hesaplanmış metrik testi (toplam 9 test geçiyor). Uçtan uca 3.000 örnekle yerelde denendi.
- GitHub deposu **public** yapıldı (Colab token'sız çekiyor)

- **Adım 2 tamam — Tablo 1 (test, eğitimsiz):** SigLIP R@1 0,357 / R@10 0,620 / mAP 0,320 → ResNet 0,088 → CLIP 0,079 → DINOv2 0,065. SigLIP, makaledeki eğitimli Match R-CNN'i (GT kutu top-1 0,268) eğitimsiz geçiyor. Öznitelikler Colab L4'te (~9 dk) çıkarıldı, Drive `features/` + yerel `data/features/`. CLIP QuickGELU düzeltmesiyle yerelde yeniden çıkarıldı. 8 çalıştırma `outputs/results/`, `docs/EXPERIMENTS.md`, `reports/tables/table1_*`. Rapor: `docs/reports/02_omurga_karsilastirmasi.docx`.
- **Karar:** sonraki tüm adımlar `siglip_vitb16` üzerinde.

## Sıradaki
1. Adım 3: `evaluation/open_set.py` (abs vs margin, AUROC, kapsama–kesinlik, τ val_internal'da %95 kapsama), `breakdown.py` (occlusion/scale/zoom/viewpoint/kategori), `scripts/evaluate.py`'ye ekle; Rapor 3 — GPU gerekmez
2. Adım 4: `engine/trainer.py` projeksiyon başlığı (InfoNCE, P×K) — train_fit öznitelikleri gerekli → Colab'da `siglip_vitb16 / train_fit` çıkarımı (≈207k görüntü, L4'te ~6 dk)
4. Adım 3 — açık küme + kırılımlar; Rapor 3

## Bekleyen
- Protokol metni: "kısa kenar 224" → config `resize_short: 256` (disk 256, model 224'e indirir) — protokol metnini düzelt
- val_internal galerisine shop style=0 dikkat dağıtıcıları eklensin mi (resmi test galerisi içeriyor) — karar bekliyor
- `head.yaml`'da ArcFace/`unfreeze_blocks` alanları yok (Adım 4'te eklenecek); eski `colab_train.ipynb` kaldırıldı, eğitim defteri Adım 4'te yazılacak
- Araştırma önerisi (ILIAS + AMES, 1 sayfa) — teslim 1 Kasım
