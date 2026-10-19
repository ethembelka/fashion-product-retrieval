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

from fashret.reporting import figures as fig  # noqa: E402
from fashret.reporting.docx_report import Report  # noqa: E402
from fashret.utils.io import read_json, read_table  # noqa: E402

FIG_DIR = ROOT / "reports/figures/01_data"
PROCESSED = ROOT / "data/processed"

CATEGORY_NAMES = {
    1: "short sleeve top", 2: "long sleeve top", 3: "short sleeve outwear", 4: "long sleeve outwear",
    5: "vest", 6: "sling", 7: "shorts", 8: "trousers", 9: "skirt", 10: "short sleeve dress",
    11: "long sleeve dress", 12: "vest dress", 13: "sling dress",
}


def n(df: pd.DataFrame, split: str, lst: str, col: str = "n_items") -> int:
    return int(df[(df.split == split) & (df["list"] == lst)][col].iloc[0])


def make_example_figures(items: pd.DataFrame) -> tuple[Path, Path]:
    """(a) one validation photo with its boxes and crops; (b) user vs shop photos of the same product."""
    ok = items[items["cropped"]]
    # (a) a photo with exactly two garments, both reasonably large
    counts = ok.groupby("image_id").size()
    two = ok[ok["image_id"].isin(counts[counts == 2].index)]
    two = two[(two.x2 - two.x1) > 150]
    img_id = two["image_id"].iloc[0]
    rows = ok[ok["image_id"] == img_id]
    boxes = [(r.x1, r.y1, r.x2, r.y2, CATEGORY_NAMES.get(r.category_id, "")) for r in rows.itertuples()]
    fa = fig.box_and_crops(rows["image_path"].iloc[0], boxes, list(rows["crop_path"]),
                           FIG_DIR / "example_crop.png", "Bir fotoğraftan kırpıntılara")
    # (b) same match_key, one user + one shop crop, large boxes
    big = ok[(ok.x2 - ok.x1) > 120]
    pair = None
    for mk, g in big[big["style"] > 0].groupby("match_key"):
        if {"user", "shop"} <= set(g["source"]):
            pair = g
            break
    u = pair[pair["source"] == "user"].iloc[0]
    s = pair[pair["source"] == "shop"].iloc[0]
    fb = fig.box_and_crops(u["image_path"], [(u.x1, u.y1, u.x2, u.y2, "müşteri (user)")],
                           [u["crop_path"], s["crop_path"]], FIG_DIR / "example_pair.png",
                           f"Aynı ürün: müşteri fotoğrafı (sol) ve mağaza fotoğrafı (sağ) — pair_id {u.pair_id}, style {u.style}")
    return fa, fb


def main() -> None:
    crop = read_json(ROOT / "data/processed/crop_report.json")
    items_val = read_table(PROCESSED / "items_validation_cropped.parquet")
    fig_example, fig_pair = make_example_figures(items_val)
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
    r.figure(fig_pair, "Şekil 1. Projenin çözdüğü problem: soldaki müşteri fotoğrafından, sağdaki mağaza fotoğrafını bulmak. "
             "İki fotoğraf aynı ürünü gösterir ama ışık, poz ve çekim açısı farklıdır.", width_in=5.5)
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
    flow = fig.flow_diagram(
        ["Ham fotoğraf\n+ JSON etiket", "Ürün tablosu\n(parquet)", "Kırpıntılar\n(JPEG)", "pair_id\nayrımı", "Galeri / sorgu /\naçık küme listeleri"],
        FIG_DIR / "pipeline.png", "Veri hazırlama akışı",
    )
    r.figure(flow, "Şekil 2. Ham veriden değerlendirme listelerine giden beş adım.", width_in=6.3)
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
    r.figure(fig_example, "Şekil 3. Solda etiketlerdeki kutular (renkli dikdörtgenler), sağda bu kutulardan üretilen kırpıntılar. "
             "Model yalnızca sağdaki görüntüleri görür.", width_in=6.0)
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
    splits_fig = fig.grouped_hbar(
        ["train_fit", "val_internal", "test"],
        {
            "Galeri (mağaza)": [n(summ, s, "gallery") for s in ("train_fit", "val_internal", "test")],
            "Sorgu (müşteri)": [n(summ, s, "queries") for s in ("train_fit", "val_internal", "test")],
            "Katalogda yok": [n(summ, s, "openset") for s in ("train_fit", "val_internal", "test")],
        },
        FIG_DIR / "splits.png", "Ayrım başına liste büyüklükleri (görüntü sayısı)",
    )
    r.figure(splits_fig, "Şekil 4. Üç ayrımın büyüklükleri. train_fit yalnızca eğitimde, val_internal yalnızca seçimde, "
             "test yalnızca raporlamada kullanılır.", width_in=6.0)
    miss = crop.get("test_official_missing", {})
    r.para(
        f"Resmi test listelerinden eşleştirilemeyen kayıt: sorgu {miss.get('queries', 0)}, galeri {miss.get('gallery', 0)}. "
        "Bu sayılar raporda belirtilecek; sonuçlar DeepFashion2 makalesiyle karşılaştırılırken dikkate alınacaktır."
    )

    r.h1("5. Verinin kategori dağılımı")
    r.para("Test sorgularının kategorilere dağılımı aşağıdadır. Bazı kategoriler çok az örnek içerir; kırılım analizinde bu gruplar işaretlenecektir.")
    col = "test/queries"
    cat_rows = [[cid, CATEGORY_NAMES.get(cid, "?"), int(cats.loc[cid, col])] for cid in cats.index if col in cats]
    cat_rows.sort(key=lambda x: -x[2])
    cat_fig = fig.hbar([f"{c[1]} ({c[0]})" for c in cat_rows], [c[2] for c in cat_rows],
                       FIG_DIR / "test_query_categories.png", "Test sorgularının kategori dağılımı", "sorgu sayısı")
    r.figure(cat_fig, "Şekil 5. 13 kategorinin test sorgularındaki payı. Üstler ve etekler çoğunlukta; "
             "'short sleeve outwear' ve 'sling' gibi kategoriler çok az örnek içerir.", width_in=5.8)
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
