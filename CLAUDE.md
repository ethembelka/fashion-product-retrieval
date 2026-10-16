# fashion-product-retrieval

CMPE 9103 Computer Vision dönem projesi (İstanbul Üniversitesi, 2026 Güz).
Öğrenci: Ethem Belka Şahin · Hoca: Dr. Öğr. Üyesi Abbas Memiş · **Yanıtları Türkçe ver.** Kod, değişken adları ve yorumlar İngilizce.

## Proje tek cümlede
DeepFashion2'de **müşteri fotoğrafından (user) mağaza kataloğundaki (shop) aynı ürünü bulmak** (consumer-to-shop, örnek düzeyinde retrieval)
ve katalogda karşılığı olmayan sorgular için **"katalogda yok" diyebilmek** (açık küme reddi). Açık küme sorguları için
DeepFashion2'deki `style = 0` ürünler doğal "katalog dışı" örnek olarak kullanılır; ek etiketleme gerekmez.

## Kısıtlar (önemli)
- **Eğitim sayısı sabit değil; zamana göre planlanır.** Önce eğitimsiz omurga karşılaştırması (Tablo 1) yapılır,
  sonra eğitimler aşağıdaki öncelik sırasıyla, süresi ölçülerek eklenir.
- Her yeni eğitimden önce: tahmini süreyi hesapla, `docs/EXPERIMENTS.md`'deki plana yaz, kullanıcıya sor.

## Eğitim planı (öncelik sırası)
1. Eğitimsiz: ResNet-50 / CLIP / SigLIP / DINOv2 karşılaştırması (eğitim yok, sadece öznitelik çıkarma)
2. En iyi omurga + projeksiyon başlığı (omurga dondurulmuş, önbellekteki öznitelikler üzerinde — dakikalar)
3. Kayıp karşılaştırması: InfoNCE vs ArcFace (yine önbellek üzerinde, ucuz)
4. Kısmi fine-tune: omurganın son 2–4 bloğu açık (asıl ağır eğitim; yerel 6 GB'ta küçük batch + gradyan biriktirme, ya da Kaggle)
5. Zaman kalırsa: ikinci bir omurgada aynı fine-tune
- Donanım: RTX 3060 Laptop **6 GB VRAM**, 16 GB RAM. Ham veri (≈17 GB) yerelde; ağır iş bir kez yapılır (kırpma + öznitelik).
  Colab gerekirse sadece küçük `crops.zip` ya da `.npy` öznitelikleri Drive'a taşınır (notebooks/colab_train.ipynb).

## Araştırma soruları
1. Hangi hazır omurga (ResNet-50 / CLIP / SigLIP / DINOv2) consumer-to-shop'ta en iyi? (Recall@K, mAP)
2. Küçük bir projeksiyon başlığı ne kadar kazandırıyor?
3. Başarı kapanma (occlusion), ölçek (scale), yakınlaştırma (zoom_in), bakış açısı (viewpoint) ve kategoriye göre nasıl değişiyor?
4. Açık küme: mutlak eşik mi, top1–top2 farkı (margin) mı daha iyi "katalogda yok" kararı veriyor? (kapsama–kesinlik eğrisi, AUROC)

## Belgeler (kod yazmadan önce oku)
- `docs/DECISIONS.md` → alınmış kararlar (**ilk bunu oku**)
- `docs/STATUS.md` → güncel durum, sıradaki adımlar
- `docs/ARCHITECTURE.md` → modüller, sorumluluklar, arayüzler, kod standartları
- `docs/EVALUATION_PROTOCOL.md` → galeri/sorgu tanımı, eşleşme kuralı, metrikler, raporlama
- `docs/EXPERIMENTS.md` → deney kaydı

## Skills (.claude/skills/)
| Skill | Ne zaman |
|---|---|
| `prepare-data` | DeepFashion2 JSON'larından ürün kırpıntıları ve galeri/sorgu listelerini üretirken |
| `extract-features` | Bir omurgayla galeri ve sorgu öznitelikleri çıkarırken |
| `train-embedding` | Projeksiyon başlığını eğitirken |
| `evaluate-retrieval` | Kapalı küme + açık küme değerlendirmesi ve kırılım analizleri |
| `log-experiment` | Her deneyden sonra kaydı güncellerken |

## Çalışma kuralları
- Yol, eşik, hiperparametre **koda gömülmez**; `configs/` altındaki YAML'lardan okunur.
- `scripts/` sadece ince komut satırı girişleridir; iş mantığı `src/fashret/` içindedir.
- Her çalıştırma `outputs/results/<run_id>/` altına `config.yaml`, `metrics.json` ve git commit hash'ini yazar.
- Metrik kodu değişirse `pytest -q` çalıştırılır. Recall@K ve mAP için elle hesaplanmış küçük örnek testleri var.
- **Sızıntı yasak:** aynı `pair_id` hem eğitimde hem değerlendirmede olamaz. İç doğrulama ayrımı `pair_id` düzeyinde yapılır.
- Veri, kırpıntılar, öznitelikler ve ağırlıklar git'e girmez (`.gitignore`).

## Veri yerleşimi
```
data/raw/DeepFashion2/                      # Kaggle sürümü (kontrol edildi 15.10.2026)
  deepfashion2_original_images/             # = configs/paths.yaml → deepfashion2_root
    train/image (191.961)   train/annos (191.961 json)
    validation/image (32.153)   validation/annos (32.153 json)
    validation/json_for_validation/val_query.json (12.377 sorgu), val_gallery.json (36.961 galeri öğesi)
    test/test/image (62.629, etiketsiz — kullanılmaz)
  img_info_dataframes/*.csv                 # pair_id/source/style İÇERMEZ → kullanma; annos JSON'larını kullan
```
Veri başka bir yerdeyse sadece `configs/paths.yaml` içindeki `deepfashion2_root` değiştirilir.

## Veri
DeepFashion2 (Ge vd., CVPR 2019): 13 kategori, ürün başına kutu, `style`, `scale`, `occlusion`, `zoom_in`, `viewpoint`;
görüntü başına `source` (user/shop) ve `pair_id`. Eşleşme: **aynı pair_id + aynı style (>0)**.
Test etiketleri yayınlanmadığı için resmi validation seti **test** gibi kullanılır.
Ham veri: `data/raw/DeepFashion2/` (train/, validation/ altında image/ ve annos/). Kırpıntılar: `data/processed/`. Öznitelikler: `data/features/`.

## Teslimler
Öneri 18 Ekim 2026 · Teslim 13 Aralık 2026 (IEEE rapor, sunum, kod) · Sunum 16 Aralık 2026.
Araştırma makaleleri: `../../03_Research_Paper/` (ILIAS, CVPR 2025 · AMES, ECCV 2024).

## Senkronizasyon (claude.ai sohbeti ↔ Claude Code)
Kullanıcı bu proje hakkında claude.ai'deki "Doktora Dersleri" sohbetinde de konuşuyor; o sohbet bu klasöre doğrudan yazabiliyor.
Ortak hafıza bu depodaki dosyalardır:
- `docs/DECISIONS.md` → alınmış kararlar (hoca görüşmeleri, kapsam değişiklikleri). **Oturum başında oku.** Yeni bir karar alınırsa en üste tarihli satır ekle.
- `docs/STATUS.md` → güncel durum ve sıradaki adımlar. **Her çalışma oturumunun sonunda güncelle.**
- `docs/EXPERIMENTS.md` → deney sonuçları.
- Kurallar: işi bitirince commit at (yarım değişiklik bırakma); oturum başında `git log --oneline -5` ve `git status` ile başka yerden gelen değişikliklere bak.
