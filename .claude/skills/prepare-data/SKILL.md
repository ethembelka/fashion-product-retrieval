---
name: prepare-data
description: DeepFashion2 JSON etiketlerinden ürün kırpıntılarını, pair_id düzeyinde ayrımları ve galeri/sorgu/açık küme listelerini üretmek için kullan.
---

# Veri hazırlama

1. `configs/paths.yaml` ve `configs/data.yaml` dosyalarını oku. `<deepfashion2_root>/{train,validation}/{image,annos}` var mı kontrol et (test listeleri için `validation/json_for_validation/val_query.json` ve `val_gallery.json` da gerekli); yoksa kullanıcıya hangi klasörün eksik olduğunu söyle ve dur.
2. `src/fashret/data/df2_parser.py`: her JSON'dan `item1, item2, ...` anahtarlarını oku; her ürün için bir satır üret:
   `item_id, split, image_path, source, pair_id, style, category_id, bbox, scale, occlusion, zoom_in, viewpoint`.
   Sonuç: `data/processed/items_{train,validation}.parquet`.
3. `src/fashret/data/cropper.py`: kutu + kenar payı ile kırp, `min_side_px` altını at, kısa kenarı `resize_short`'a getir, `data/processed/crops/{split}/{item_id}.jpg` olarak yaz.
   Kaç ürünün atıldığını `data/processed/crop_report.json`'a yaz. Paralel çalıştır (çok çekirdek), zaten var olan kırpıntıyı atla (yeniden başlatılabilir olsun).
4. `src/fashret/data/splits.py`:
   - train'deki pair_id'leri tohum 0 ile %90 / %10 böl → `train_fit`, `val_internal`.
   - Her ayrım için `gallery.csv`, `queries.csv`, `openset.csv` üret (kurallar `docs/EVALUATION_PROTOCOL.md`).
   - `match_key = f"{pair_id}_{style}"`. Kapalı küme sorgusunun galeride eşi yoksa listeye girmez.
   - Ayrım CSV'leri `data/processed/splits/` altında; bunlar küçük olduğu için git'e girer.
5. `tests/test_splits.py` çalıştır: train_fit ve val_internal pair_id kümeleri kesişmemeli.
6. Özet tablo yazdır: ayrım başına galeri / sorgu / açık küme sayısı, kategori dağılımı. Bu tablo rapordaki veri bölümüne girer → `reports/tables/data_summary.csv`.
7. Colab gerekecekse `data/processed/crops/` klasörünü `crops.zip` olarak paketle (birkaç GB) ve kullanıcıya Drive'a yüklemesini söyle.
