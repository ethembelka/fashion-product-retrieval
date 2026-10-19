"""Build docs/reports/01_veri_hazirlama.docx from the prepare-data outputs.

Usage: python scripts/report_01_data.py
Reads: data/processed/crop_report.json, reports/tables/data_summary.csv, data_categories.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from fashret.reporting.docx_report import Report  # noqa: E402
from fashret.utils.io import read_json  # noqa: E402

CATEGORY_NAMES = {
    1: "short sleeve top", 2: "long sleeve top", 3: "short sleeve outwear", 4: "long sleeve outwear",
    5: "vest", 6: "sling", 7: "shorts", 8: "trousers", 9: "skirt", 10: "short sleeve dress",
    11: "long sleeve dress", 12: "vest dress", 13: "sling dress",
}


def n(df: pd.DataFrame, split: str, lst: str, col: str = "n_items") -> int:
    return int(df[(df.split == split) & (df["list"] == lst)][col].iloc[0])


def main() -> None:
    crop = read_json(ROOT / "data/processed/crop_report.json")
    summ = pd.read_csv(ROOT / "reports/tables/data_summary.csv")
    cats = pd.read_csv(ROOT / "reports/tables/data_categories.csv", index_col=0)
    crops_gb = sum(p.stat().st_size for p in (ROOT / "data/processed/crops").rglob("*.jpg")) / 1e9

    r = Report(
        title="Adım 1 — Veri Hazırlama",
        subtitle="DeepFashion2 ham verisinden ürün kırpıntıları, ayrımlar ve değerlendirme listeleri",
        author="Ethem Belka Şahin",
        course="CMPE 9103 Computer Vision · Dönem Projesi İlerleme Raporu · İstanbul Üniversitesi",
    )

    r.h1("1. Bu rapor ne anlatıyor?")
    r.para(
        "Projenin amacı, bir müşterinin çektiği kıyafet fotoğrafından yola çıkarak mağaza kataloğundaki "
        "aynı ürünü bulmaktır. Bunu yapabilmek için önce ham veri setinin bilgisayarın işleyebileceği "
        "düzenli bir biçime getirilmesi gerekir. Bu rapor, o hazırlık adımını hiç bilmeyen bir okuyucu "
        "için anlatır: nereden başladık, proje önerisinde ne söz verdik, ne yaptık ve ortaya ne çıktı."
    )

    r.h1("2. Başlangıç noktası")
    r.para(
        "Elimizde Kaggle'dan indirilmiş 15 GB'lık bir zip dosyası vardı: DeepFashion2 veri seti. "
        "Bu set, 2019'da yayınlanmış, kıyafet görüntüleri üzerine kurulu büyük bir araştırma veri setidir. "
        "Her fotoğrafın yanında bir etiket dosyası (JSON) bulunur ve bu dosya fotoğraftaki her kıyafet için şunları söyler:"
    )
    r.bullets([
        "Kıyafetin fotoğraftaki yeri (dikdörtgen bir kutu: sol-üst ve sağ-alt köşe koordinatları).",
        "Kategorisi (13 sınıf: kısa kollu üst, pantolon, etek, elbise, ...).",
        "Fotoğrafın kaynağı: müşteri çekimi (user) mi, mağaza çekimi (shop) mu.",
        "Ürün grubu numarası (pair_id): aynı ürünün müşteri ve mağaza fotoğrafları aynı numarayı taşır.",
        "Stil numarası (style): aynı ürün grubunda farklı renk/desen varyantlarını ayırır. style = 0 ise o kıyafetin katalogda tanımlı bir karşılığı yoktur.",
        "Çekim koşulları: kapanma (occlusion), ölçek (scale), yakınlaştırma (zoom_in), bakış açısı (viewpoint).",
    ])
    r.table(
        ["Bölüm", "Fotoğraf", "Etiket (JSON)", "Kullanım"],
        [
            ["train", 191961, 191961, "Eğitim ve iç doğrulama"],
            ["validation", 32153, 32153, "Test (resmi retrieval listeleriyle)"],
            ["test", 62629, "yok", "Kullanılmıyor (etiketsiz)"],
        ],
        caption="Tablo 1. Ham veri setinin içeriği",
    )
    r.note(
        "Neden resmi test bölümü kullanılmıyor? DeepFashion2'nin test etiketleri yayınlanmamıştır; doğru cevabı "
        "bilmeden başarı ölçülemez. Bu yüzden araştırma camiasının yaptığı gibi 'validation' bölümünü test gibi kullanıyoruz."
    )

    r.h1("3. Proje önerisinde ne söz verdik?")
    r.para("18 Ekim 2026'da teslim edilen öneride veri hazırlığıyla ilgili şu taahhütler yer aldı:")
    r.bullets([
        "Her kıyafet, etiketteki gerçek kutusundan kesilecek (nesne tespiti projenin konusu değil).",
        "Resmi değerlendirme listeleri (12.377 müşteri sorgusu, 36.961 mağaza galerisi öğesi) test seti olarak kullanılacak.",
        "Eğitim verisindeki ürün gruplarının (pair_id) %10'u model seçimi için ayrılacak; aynı ürün hem eğitimde hem değerlendirmede bulunmayacak.",
        "style = 0 olan müşteri ürünleri, ek etiketleme yapılmadan 'katalogda yok' sorguları olarak kullanılacak.",
    ])

    r.h1("4. Ne yaptık?")
    r.h2("4.1 Etiketleri tabloya çevirdik")
    r.para(
        "224.114 JSON dosyası tek tek okundu ve her kıyafet için bir satır içeren iki tablo üretildi "
        "(items_train.parquet, items_validation.parquet). Böylece 'hangi fotoğrafta, hangi kutuda, hangi ürün var' "
        "sorusu tek bir tablodan cevaplanabiliyor."
    )
    r.h2("4.2 Kıyafetleri kestik (kırpma)")
    r.para(
        "Model, fotoğrafın tamamına değil yalnızca kıyafete bakmalı. Bu yüzden her kutu, çevresine %10 pay eklenerek "
        "kesildi ve kısa kenarı 256 piksel olacak şekilde küçültüldü. Sonuç, her ürün için küçük bir JPEG dosyasıdır."
    )
    r.table(
        ["Bölüm", "Toplam ürün", "Kesilen", "Çok küçük (atıldı)", "Hata"],
        [[s, crop[s]["total"], crop[s]["cropped"], crop[s]["dropped_too_small"], crop[s]["errors"]] for s in ("train", "validation")],
        caption="Tablo 2. Kırpma raporu",
    )
    r.note(
        f"Kenarı 32 pikselden küçük kutular atıldı; bu kadar küçük bir görüntüden anlamlı bir öznitelik çıkmaz. "
        f"Kırpıntıların toplam boyutu yaklaşık {crops_gb:.1f} GB; ham veri 15 GB idi."
    )
    r.h2("4.3 Eğitim verisini ikiye böldük")
    r.para(
        "Eğitim bölümündeki ürün grupları (pair_id) sabit bir rastgele tohumla karıştırıldı; %90'ı train_fit "
        "(model eğitimi), %10'u val_internal (eşik ve model seçimi) oldu. Bölme fotoğraf düzeyinde değil ürün grubu "
        "düzeyinde yapıldı: bir ürünün müşteri fotoğrafı bir tarafta, mağaza fotoğrafı diğer tarafta kalamaz."
    )
    r.note(
        "Bu neden önemli? Model, eğitimde gördüğü bir ürünü değerlendirmede tekrar görürse başarı yapay olarak yükselir "
        "(veri sızıntısı). Otomatik bir test, iki kümenin kesişmediğini her çalıştırmada doğrular."
    )
    r.h2("4.4 Galeri, sorgu ve 'katalogda yok' listelerini oluşturduk")
    r.bullets([
        "Galeri (katalog): mağaza fotoğrafları, style > 0.",
        "Sorgular: müşteri fotoğrafları, style > 0 ve galeride aynı ürün (aynı pair_id ve style) bulunanlar.",
        "Açık küme sorguları: müşteri fotoğrafları, style = 0 — katalogda karşılığı tanımlı değil.",
        "Test için resmi val_query.json ve val_gallery.json listeleri birebir kullanıldı; kendi kurallarımız yalnızca "
        "train_fit, val_internal ve açık küme için geçerli.",
    ])
    rows = []
    for split in ("train_fit", "val_internal", "test"):
        for lst in ("gallery", "queries", "openset"):
            rows.append([split, lst, n(summ, split, lst), n(summ, split, lst, "n_products")])
    r.table(["Ayrım", "Liste", "Görüntü sayısı", "Farklı ürün sayısı"], rows, caption="Tablo 3. Üretilen listeler")
    miss = crop.get("test_official_missing", {})
    r.para(
        f"Resmi test listelerinden eşleştirilemeyen kayıt: sorgu {miss.get('queries', 0)}, galeri {miss.get('gallery', 0)}. "
        "Bu sayılar raporda belirtilecek; sonuçlar DeepFashion2 makalesiyle karşılaştırılırken dikkate alınacaktır."
    )

    r.h1("5. Verinin kategori dağılımı")
    r.para("Test sorgularının kategorilere dağılımı aşağıdadır. Bazı kategoriler çok az örnek içerir; kırılım analizinde bu gruplar işaretlenecektir.")
    col = "test/queries"
    cat_rows = [[cid, CATEGORY_NAMES.get(cid, "?"), int(cats.loc[cid, col])] for cid in cats.index if col in cats]
    r.table(["Kategori no", "Kategori", "Test sorgusu"], cat_rows, caption="Tablo 4. Test sorgularının kategori dağılımı")

    r.h1("6. Ortaya çıkan dosyalar")
    r.bullets([
        "data/processed/items_train.parquet, items_validation.parquet — ürün tabloları",
        "data/processed/crops/{train,validation}/*.jpg — kırpıntılar (git'e girmez, Drive'a crops.zip olarak yüklenir)",
        "data/processed/splits/{train_fit,val_internal,test}/{gallery,queries,openset}.csv — listeler (git'te)",
        "data/processed/crop_report.json, reports/tables/data_summary.csv, data_categories.csv — özetler",
    ])

    r.h1("7. Sıradaki adım")
    r.para(
        "Kırpıntılar crops.zip olarak Google Drive'a yüklenecek. Adım 2'de dört hazır model (ResNet-50, CLIP, SigLIP, DINOv2) "
        "Colab'da bu kırpıntıları sayı vektörlerine çevirecek ve ilk başarı tablosu (Tablo 1: eğitimsiz omurga karşılaştırması) çıkacaktır."
    )

    out = r.save(ROOT / "docs/reports/01_veri_hazirlama.docx")
    print("written", out)


if __name__ == "__main__":
    main()
