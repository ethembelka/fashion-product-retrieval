# Değerlendirme Protokolü

## Ayrımlar
| Ad | Kaynak | Kullanım |
|---|---|---|
| train_fit | DF2 train, pair_id'lerin %90'ı | Projeksiyon başlığı eğitimi |
| val_internal | DF2 train, pair_id'lerin %10'u | Erken durdurma, eşik (τ) seçimi |
| test (= DF2 validation) | Resmi validation seti | **Sadece rapor.** Hiçbir seçim bunun üzerinde yapılmaz |

Ayrım `pair_id` düzeyinde ve sabit tohumla (seed 0) yapılır; liste `data/processed/splits/` altında saklanır ve git'e girer (küçük CSV).

## Galeri ve sorgular (her ayrımda aynı kural)
- **Galeri:** `source = shop` ve `style > 0` ürün kırpıntıları.
- **Kapalı küme sorguları:** `source = user`, `style > 0` ve galeride aynı `match_key`'e sahip en az bir ürünü olanlar.
- **Açık küme sorguları:** `source = user` ve `style = 0` (katalogda karşılığı tanımlı değil).
- **Eşleşme:** aynı `pair_id` **ve** aynı `style`. Kategori aynı ama style farklıysa yanlış sayılır.
- Kırpma: kutu + %10 kenar payı, kısa kenar 224'e ölçek. Çok küçük kutular (< 32 px kenar) atılır ve sayıları raporlanır.

## Benzerlik
L2-normalize gömmeler, FAISS `IndexFlatIP` (kosinüs). Tam arama; yaklaşık indeks kullanılmaz.

## Metrikler
| Metrik | Tanım | Not |
|---|---|---|
| Recall@K (K=1,5,10,20) | İlk K sonuçta en az bir doğru eşleşme olan sorgu oranı | DF2 makalesiyle karşılaştırma için ana metrik |
| mAP | Sorgu başına average precision ortalaması (tüm doğru eşleşmeler üzerinden) | Sıralamanın tamamını ölçer |
| Kapsama–kesinlik eğrisi | τ değiştikçe: kabul edilen sorgu oranı (kapsama) vs kabul edilenlerde top-1 doğruluğu | Açık küme |
| AUROC (bilinen vs bilinmeyen) | Skorun kapalı küme ile açık küme sorgularını ayırma gücü | İki skor: mutlak s1, margin s1–s2 |
| Yanlış eşleşme oranı | Açık küme sorgularından τ üstünde kabul edilenlerin oranı | τ, val_internal'da %95 kapsama için seçilir |
| Kırılımlar | Recall@1, occlusion / scale / zoom_in / viewpoint / kategori bazında | Sorgu (user) ürününün özelliklerine göre |
| Verimlilik | Gömme boyutu, sorgu başına ms (GPU çıkarma + arama), indeks boyutu MB | |

## Referans
DeepFashion2 makalesi (Ge vd., CVPR 2019) consumer-to-shop için Match R-CNN'in top-1…top-20 doğruluklarını verir.
**Sayıları rapora koymadan önce makalenin retrieval tablosundan doğrula** ve buraya yaz. Bu sayılar kendi kurulumumuzla
bire bir aynı değildir (kırpıntı kuralı, filtreler); raporda farkı açıkça yaz.

## Rapor biçimi
- Tablo 1: omurga karşılaştırması (eğitimsiz) — R@1/5/10/20, mAP, embed_dim, ms/sorgu.
- Tablo 2: en iyi omurga, eğitimsiz vs projeksiyon başlığı.
- Şekil: kapsama–kesinlik eğrileri (mutlak vs margin), kırılım çubuk grafikleri, başarı/başarısızlık galerileri.
- Tohum tekrarı yapılırsa ortalama ± std.

## Çıktı dosyaları (`outputs/results/<run_id>/`)
`config.yaml`, `git_hash.txt`, `metrics.json`, `per_query.parquet` (sorgu başına sıra, skor, doğru mu),
`coverage_curve.csv`, `figures/*.png`.
