# Fashion Product Retrieval (DeepFashion2)

DeepFashion2'de müşteri fotoğrafından mağaza ürününü bulma (consumer-to-shop retrieval) ve "katalogda yok" reddi. (CMPE 9103 dönem projesi) Ayrıntılar için `CLAUDE.md`.

## Kurulum (yerel)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Veri
DeepFashion2'yi `data/raw/` içine koy (ya da bulunduğu yeri `configs/paths.yaml` dosyasında belirt).

## Colab (isteğe bağlı)
Ham DeepFashion2 (≈17 GB) Drive'a taşınmaz. Kırpıntılar yerelde bir kez üretilir (`prepare-data`), `crops.zip` olarak
Drive'a yüklenir ve `notebooks/colab_train.ipynb` onu kullanır. Projeksiyon başlığı önceden çıkarılmış öznitelikler üzerinde
eğitildiği için çoğu durumda Colab'a gerek kalmaz.

## Yapı
```
configs/    YAML ayarları (yol, model, eğitim, değerlendirme) — kodda sabit değer yok
src/fashret/  iş mantığı (veri, model, değerlendirme, raporlama)
scripts/    ince komut satırı girişleri
notebooks/  Colab defteri
docs/       mimari, değerlendirme protokolü, deney kaydı
tests/      metrik testleri (pytest)
outputs/    çalıştırma çıktıları (git'e girmez)
reports/    rapora girecek tablo ve şekiller
.claude/skills/  Claude Code skill'leri
```

## Claude Code ile çalışma
Bu klasörü Claude Code ile aç; `CLAUDE.md` ve `.claude/skills/` otomatik okunur.
Örnek istekler: "veriyi hazırla", "dinov2 özniteliklerini çıkar", "dört omurgayı karşılaştır", "açık küme analizini yap".
