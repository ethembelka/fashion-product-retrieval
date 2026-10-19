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

## Sıradaki
1. Kullanıcı: `crops.zip`'i Drive'a yükle
2. Adım 2 — `extract-features`: Colab defteri + `src/fashret/models/` + `engine/extractor.py`; 4 omurga, val_internal + test → `.npy` Drive'a
3. Adım 2 — `evaluate-retrieval` (yerel CPU): Recall@K, mAP → Tablo 1; Rapor 2
4. Adım 3 — açık küme + kırılımlar; Rapor 3

## Bekleyen
- Protokol metni: "kısa kenar 224" → config `resize_short: 256` (disk 256, model 224'e indirir) — protokol metnini düzelt
- val_internal galerisine shop style=0 dikkat dağıtıcıları eklensin mi (resmi test galerisi içeriyor) — karar bekliyor
- `head.yaml`'da ArcFace/`unfreeze_blocks` alanları yok; Colab defteri güncellenecek (Adım 2'de)
- Araştırma önerisi (ILIAS + AMES, 1 sayfa) — teslim 1 Kasım
