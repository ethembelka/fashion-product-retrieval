# Mimari

## İlkeler
- **Tek sorumluluk (S):** her modül tek bir iş yapar. Veri hazırlama, öznitelik çıkarma, indeksleme, değerlendirme ayrı paketlerdir.
- **Açık/kapalı (O):** yeni bir omurga eklemek için mevcut kod değişmez; `models/` altına yeni sınıf + registry kaydı yeterli.
- **Liskov (L):** her öznitelik çıkarıcı aynı arayüzü uygular; değerlendirme hangi model olduğunu bilmez.
- **Arayüz ayrımı (I):** değerlendirme sadece `(query_emb, gallery_emb, ids)` dizilerine bağlıdır, modele değil.
- **Bağımlılığın tersine çevrilmesi (D):** üst seviye akış (`engine`, `scripts`) somut sınıflara değil `FeatureExtractor` protokolüne bağlıdır.
- Fonksiyonlar kısa, saf (yan etkisiz) olmaya çalışır; G/Ç `utils/io.py` üzerinden. Tip ipuçları zorunlu. Sihirli sayı yok, her şey config'te.

## Klasör yapısı
```
src/fashret/
  config.py                 # YAML yükleme, dataclass'lar (DataCfg, ModelCfg, EvalCfg, TrainCfg)
  data/
    df2_parser.py           # DeepFashion2 JSON → ürün kayıtları (item_id, image, bbox, category, style, pair_id, source, attrs)
    cropper.py              # kutudan kırpma (+ kenar payı), diske yazma
    splits.py               # pair_id düzeyinde train_fit / val_internal; galeri/sorgu/açık küme listeleri
    dataset.py              # torch Dataset: kırpıntı + etiket (pair_id, style) → match_key
    samplers.py             # P×K örnekleyici (P ürün × K görüntü) — sadece eğitimde
  models/
    base.py                 # FeatureExtractor protokolü
    timm_backbone.py        # ResNet-50, DINOv2 (timm)
    openclip_backbone.py    # CLIP, SigLIP (open_clip)
    projection_head.py      # MLP projeksiyon başlığı
    registry.py             # isim → sınıf eşlemesi ("dinov2_vitb14" → ...)
  index/
    faiss_index.py          # IndexFlatIP, L2 normalizasyon, top-k arama
  engine/
    extractor.py            # batch'li öznitelik çıkarma, .npy + meta .parquet yazma
    trainer.py              # projeksiyon başlığı eğitimi (InfoNCE), checkpoint, resume
  evaluation/
    closed_set.py           # Recall@K, mAP
    open_set.py             # mutlak eşik / margin skorları, kapsama–kesinlik, AUROC, yanlış eşleşme oranı
    breakdown.py            # occlusion / scale / zoom_in / viewpoint / kategori kırılımları
    efficiency.py           # sorgu süresi, gömme boyutu, bellek
    evaluator.py            # hepsini toplayan üst sınıf → metrics.json
  analysis/
    galleries.py            # başarılı / başarısız sorgu görselleri (sorgu + top-5)
  reporting/
    tables.py               # metrics.json'lardan karşılaştırma tabloları (CSV + LaTeX)
  utils/
    io.py, seed.py, run.py  # dosya G/Ç, tohum sabitleme, run_id + git hash
scripts/
  prepare_data.py           # → data/processed/
  extract_features.py       # --model dinov2_vitb14 --split val
  train_head.py             # --config configs/train/head.yaml
  evaluate.py               # --features <dir> --config configs/eval/default.yaml
  make_tables.py
tests/
  test_closed_set.py, test_open_set.py, test_splits.py
```

## Veri akışı
```
DF2 annos/*.json ──df2_parser──► items.parquet ──cropper──► data/processed/crops/*.jpg
                                        │
                                     splits ──► gallery.csv / queries.csv / openset.csv
                                        │
              FeatureExtractor (registry) ──extractor──► data/features/<model>/<split>/{emb.npy, meta.parquet}
                                        │
                     (isteğe bağlı) trainer ──► projection head ──► yeni öznitelikler
                                        │
                      FaissIndex + Evaluator ──► outputs/results/<run_id>/metrics.json
```

## Arayüzler
```python
class FeatureExtractor(Protocol):
    name: str
    embed_dim: int
    def preprocess(self) -> Callable[[PIL.Image.Image], torch.Tensor]: ...
    def encode(self, batch: torch.Tensor) -> torch.Tensor: ...   # (B, D), L2 normalize edilmiş

class RetrievalIndex(Protocol):
    def add(self, emb: np.ndarray) -> None: ...
    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...  # (scores, indices)

def evaluate(query_emb, query_keys, gallery_emb, gallery_keys, openset_emb, cfg) -> dict: ...
```
`match_key = f"{pair_id}_{style}"`. Değerlendirme sadece bu anahtarlara bakar.

## Adlandırma
- `run_id = {YYYYMMDD}_{model}_{variant}_s{seed}` (ör. `20261020_dinov2_vitb14_zeroshot_s0`).
- `metrics.json` anahtarları: `recall@1`, `recall@5`, `recall@10`, `recall@20`, `map`,
  `openset.auroc_abs`, `openset.auroc_margin`, `openset.precision@cov{50,80,95}`, `openset.false_match_rate@tau`,
  `breakdown.{occlusion,scale,zoom_in,viewpoint,category}.{değer}.recall@1`, `efficiency.{embed_dim,ms_per_query,index_mb}`.
