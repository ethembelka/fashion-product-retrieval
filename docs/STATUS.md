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

## Sıradaki
1. Kullanıcı: `crops.zip` Drive'a yükleniyor → sonra `notebooks/colab_extract_features.ipynb` (L4, 4 model × 2 ayrım)
2. Öznitelikler Drive'dan yerel `data/features/` altına indirilince: 4 model × test → `scripts/evaluate.py` → Tablo 1; `scripts/make_tables.py`; Rapor 2
3. Adım 3: `evaluation/open_set.py` (AUROC, kapsama–kesinlik, τ seçimi), `breakdown.py`, galeriler; Rapor 3
4. Adım 3 — açık küme + kırılımlar; Rapor 3

## Bekleyen
- Protokol metni: "kısa kenar 224" → config `resize_short: 256` (disk 256, model 224'e indirir) — protokol metnini düzelt
- val_internal galerisine shop style=0 dikkat dağıtıcıları eklensin mi (resmi test galerisi içeriyor) — karar bekliyor
- `head.yaml`'da ArcFace/`unfreeze_blocks` alanları yok (Adım 4'te eklenecek); eski `colab_train.ipynb` kaldırıldı, eğitim defteri Adım 4'te yazılacak
- Araştırma önerisi (ILIAS + AMES, 1 sayfa) — teslim 1 Kasım
