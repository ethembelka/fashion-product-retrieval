# Durum
_Son güncelleme: 2026-10-21 (cc)_

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

- **Adım 3 tamam (yerel CPU):** `evaluation/open_set.py`, `breakdown.py`; `scripts/evaluate.py` artık val_internal'da τ seçip test'e uyguluyor; çıktılar `per_openset.parquet`, `coverage_curve_{abs,margin}.csv`, `breakdown.csv`. 13 test geçiyor. Bulgular (SigLIP test): AUROC bilinen↔yok s1 0,45 / margin 0,58 (yok diyemiyor); AUROC doğru↔yanlış margin 0,77; %50 kapsamada kesinlik 0,53; kırılım: ağır kapanma 0,20, küçük ölçek 0,20, yandan/arkadan 0,27, trousers 0,17. Rapor: `docs/reports/03_acik_kume_ve_kirilimlar.docx`.

- **Adım 4 kodu hazır:** `models/projection_head.py`, `data/samplers.py` (P×K), `engine/losses.py` (InfoNCE user→shop, ArcFace), `engine/trainer.py` (val_internal R@1 ile seçim, best/last.pt, resume), `scripts/train_head.py` (başlığı val_internal+test özniteliklerine uygular → `data/features/siglip_vitb16+head/`). 16 test geçiyor. Windows'ta torch→pyarrow.dataset DLL çökmesi: `fashret` paketi pyarrow'u önce yükler, betikler `import fashret` ile başlar.
- Raporlar 1–3 öğretici biçimde baştan yazıldı (2026-10-21).

## Sıradaki
0. Kullanıcı: Colab'da `siglip_vitb16 / train_fit` çıkarımı → `data/features/siglip_vitb16/train_fit/` (bekleniyor)
1. Adım 4: `engine/trainer.py` projeksiyon başlığı (InfoNCE, P×K) — train_fit öznitelikleri gerekli → Colab'da `siglip_vitb16 / train_fit` çıkarımı (≈207k görüntü, L4'te ~6 dk)
2. Adım 4 sonrası: `scripts/evaluate.py --variant head` (val_internal → test), Tablo 2 (zeroshot vs head), Rapor 4; sonra ArcFace karşılaştırması (Adım 4b)

## Bekleyen
- Protokol metni: "kısa kenar 224" → config `resize_short: 256` (disk 256, model 224'e indirir) — protokol metnini düzelt
- val_internal galerisine shop style=0 dikkat dağıtıcıları eklensin mi (resmi test galerisi içeriyor) — karar bekliyor
- Kısmi fine-tune (plan madde 4) için `unfreeze_blocks` henüz yok; başlık sonuçlarına göre karar verilecek
- Araştırma önerisi (ILIAS + AMES, 1 sayfa) — teslim 1 Kasım
