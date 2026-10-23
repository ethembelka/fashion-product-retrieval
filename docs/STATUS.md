# Durum
_Son güncelleme: 2026-10-23 (cc)_

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

- **Adım 4 tamam (yerel 3060, ≈3 dk/eğitim):** train_fit SigLIP öznitelikleri Colab'da çıkarıldı (194.603 satır, 16.374 ürün). InfoNCE başlığı (768→1024→256): val R@1 0,422→0,656; **test R@1 0,357→0,559, mAP 0,320→0,542, AUROC var↔yok (s1) 0,45→0,68**, %50 kapsamada kesinlik 0,80. ArcFace (class_lr 1e-2, m 0,2): val 0,640, test R@1 0,567 / mAP 0,529 / AUROC 0,70 — başa baş; seçim val'e göre → InfoNCE ana yöntem. İlk ArcFace denemesi (class_lr = lr) 0,08'de çöktü. Öznitelikler `data/features/siglip_vitb16+{head,arcface}/`, checkpoint'ler `outputs/checkpoints/`. Rapor: `docs/reports/04_projeksiyon_basligi.docx`.

- **Adım 5 kodu hazır (kısmi fine-tune):** `engine/finetune.py` (FineTuneModel: son 4 blok + attn_pool + başlık; RandomResizedCrop+flip; AMP; warmup+cosine; Drive'a best/last.pt + resume; sonunda val/test öznitelik çıkarımı), `scripts/finetune.py` (--smoke), `configs/train/finetune.yaml`, `checkpoints/siglip_vitb16_head_infonce.pt` (başlangıç, git'te), `notebooks/colab_finetune.ipynb` (Tümünü çalıştır → bitince runtime.unassign). Yerel smoke geçti.

- **Adım 5 tamam (Colab L4, tam otomatik):** son 4 blok + başlık fine-tune. val R@1 0,745; **test R@1 0,633 / mAP 0,628 / R@20 0,898 / AUROC abs 0,714 / kat.tut@5 0,845** (başlık: 0,559 / 0,542 / 0,862 / 0,676). Öznitelikler `data/features/siglip_vitb16+ft/`. Rapor: `docs/reports/05_kismi_fine_tune.docx`. `history.json` henüz indirilmedi (eğitim eğrisi raporda eksik).

## Sıradaki
1. Kullanıcı: Drive `outputs/checkpoints/ft/history.json` + `train_summary.json` → `outputs/checkpoints/ft/` (Rapor 5'e eğitim eğrisi)
2. Tohum tekrarı (başlık 3 tohum, ucuz); IEEE rapor taslağı (LaTeX, Tablo 1–2, şekiller); sunum
2. Nihai rapor hazırlığı: tohum tekrarı (3 tohum, başlık), Tablo 1–2 LaTeX, IEEE taslak; sunum

## Bekleyen
- Protokol metni: "kısa kenar 224" → config `resize_short: 256` (disk 256, model 224'e indirir) — protokol metnini düzelt
- val_internal galerisine shop style=0 dikkat dağıtıcıları eklensin mi (resmi test galerisi içeriyor) — karar bekliyor
- Kısmi fine-tune (plan madde 4) için `unfreeze_blocks` henüz yok; başlık sonuçlarına göre karar verilecek
- Araştırma önerisi (ILIAS + AMES, 1 sayfa) — teslim 1 Kasım
