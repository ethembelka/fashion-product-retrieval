"""Build docs/reports/01_veri_hazirlama.docx — the data set and the data-preparation step, explained for a reader
who is learning the subject from these reports.

Usage: python scripts/report_01_data.py
Reads: data/raw (one JSON + a few images for figures), data/processed/*.parquet, crop_report.json,
       reports/tables/data_summary.csv, data_categories.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from fashret.config import DataCfg  # noqa: E402
from fashret.data.cropper import expand_box  # noqa: E402
from fashret.evaluation.breakdown import LABELS  # noqa: E402
from fashret.reporting import figures as fig  # noqa: E402
from fashret.reporting.docx_report import Report  # noqa: E402
from fashret.utils.io import read_json, read_table  # noqa: E402

FIG_DIR = ROOT / "reports/figures/01_data"
PROCESSED = ROOT / "data/processed"
CATS = LABELS["category_id"]
ATTR_TR = {"scale": "Ölçek (scale)", "occlusion": "Kapanma (occlusion)", "zoom_in": "Yakınlaştırma (zoom_in)", "viewpoint": "Bakış açısı (viewpoint)"}
LEVEL_TR = {
    "scale": {1: "1 küçük", 2: "2 orta", 3: "3 büyük"},
    "occlusion": {1: "1 hafif", 2: "2 orta", 3: "3 ağır"},
    "zoom_in": {1: "1 yok", 2: "2 orta", 3: "3 çok"},
    "viewpoint": {1: "1 giyilmemiş", 2: "2 önden", 3: "3 yandan/arkadan"},
}


def n(df: pd.DataFrame, split: str, lst: str, col: str = "n_items") -> int:
    return int(df[(df.split == split) & (df["list"] == lst)][col].iloc[0])


# ------------------------------------------------------------------ figures
def fig_pair_group(items: pd.DataFrame, pair_id: int) -> Path:
    """Every cropped garment of one pair_id, user row vs shop row, captioned with style + category."""
    g = items[items.pair_id == pair_id].sort_values(["source", "style"])
    rows, labels = [], []
    for src, lab in (("user", "müşteri (user)"), ("shop", "mağaza (shop)")):
        sub = g[g.source == src]
        rows.append([(PROCESSED / r.crop_path, f"style {r.style}\n{r.category_name}") for r in sub.itertuples()])
        labels.append(lab)
    frame = [[("#1baf7a" if int(c[1].split()[1]) == 0 else fig.SERIES[int(c[1].split()[1]) % 6]) for c in row] for row in rows]
    return fig.image_grid(rows, FIG_DIR / "pair_group.png", f"Bir ürün grubu: pair_id = {pair_id} — aynı fotoğraf çekimine ait tüm kıyafetler",
                          row_labels=labels, cell=1.45, frame=frame)


def fig_attribute_levels(items: pd.DataFrame) -> Path:
    """One example crop per level for each of the four condition labels (user photos, large boxes)."""
    rng = np.random.default_rng(3)
    big = items[(items.source == "user") & ((items.x2 - items.x1) > 140) & ((items.y2 - items.y1) > 140)]
    rows, labels = [], []
    for attr in ("scale", "occlusion", "zoom_in", "viewpoint"):
        row = []
        for lvl in (1, 2, 3):
            pool = big if attr != "scale" else items[items.source == "user"]
            c = pool[pool[attr] == lvl]
            r = c.iloc[rng.integers(len(c))]
            row.append((PROCESSED / r.crop_path, LEVEL_TR[attr][lvl]))
        rows.append(row)
        labels.append(ATTR_TR[attr])
    return fig.image_grid(rows, FIG_DIR / "attribute_levels.png", "Dört çekim koşulu etiketi ve her düzeyine bir örnek (müşteri fotoğrafları)", row_labels=labels, cell=1.5)


def fig_categories(items: pd.DataFrame) -> Path:
    rng = np.random.default_rng(5)
    shop = items[(items.source == "shop") & ((items.x2 - items.x1) > 150)]
    cells = []
    for cid in range(1, 14):
        c = shop[shop.category_id == cid]
        r = c.iloc[rng.integers(len(c))]
        cells.append((PROCESSED / r.crop_path, f"{cid} {CATS[cid]}"))
    return fig.image_grid([cells[:7], cells[7:]], FIG_DIR / "categories.png", "13 kıyafet kategorisi (mağaza fotoğraflarından birer örnek)", cell=1.35)


def fig_crop_arith(items: pd.DataFrame, cfg: DataCfg) -> tuple[Path, dict]:
    """The worked example used in the text: one box, its margin, the clipped box, the resize."""
    r = items[items.item_id == "validation_000001_item2"].iloc[0]
    with Image.open(r.image_path) as im:
        W, H = im.size
    box = expand_box(r.x1, r.y1, r.x2, r.y2, cfg.crop.margin, W, H)
    bw, bh = r.x2 - r.x1, r.y2 - r.y1
    cw, ch = box[2] - box[0], box[3] - box[1]
    s = cfg.crop.resize_short / min(cw, ch)
    out = (round(cw * s), round(ch * s)) if min(cw, ch) > cfg.crop.resize_short else (cw, ch)
    with Image.open(PROCESSED / r.crop_path) as c:
        real = c.size
    facts = dict(W=W, H=H, x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2, bw=bw, bh=bh, dx=round(bw * cfg.crop.margin), dy=round(bh * cfg.crop.margin),
                 box=box, cw=cw, ch=ch, out=out, real=real, margin=cfg.crop.margin, short=cfg.crop.resize_short)
    p = fig.box_and_crops(r.image_path, [(r.x1, r.y1, r.x2, r.y2, "etiket kutusu"), (box[0], box[1], box[2], box[3], "+%10 pay")],
                          [r.crop_path and PROCESSED / r.crop_path], FIG_DIR / "crop_arith.png", "Kırpma: etiket kutusu, payla büyütülmüş kutu ve sonuç")
    return p, facts


def fig_example_pair(items: pd.DataFrame) -> Path:
    big = items[(items.x2 - items.x1) > 120]
    for mk, g in big[big["style"] > 0].groupby("match_key"):
        if {"user", "shop"} <= set(g["source"]):
            u, s = g[g.source == "user"].iloc[0], g[g.source == "shop"].iloc[0]
            break
    return fig.box_and_crops(u.image_path, [(u.x1, u.y1, u.x2, u.y2, "müşteri")], [PROCESSED / u.crop_path, PROCESSED / s.crop_path],
                             FIG_DIR / "example_pair.png", f"Aynı ürün: müşteri fotoğrafı (sol, kutu içinde) ve mağaza fotoğrafı (sağ) — pair_id {u.pair_id}, style {u.style}")


# ------------------------------------------------------------------ document
def main() -> None:
    cfg = DataCfg.from_yaml(ROOT / "configs/data.yaml", root=ROOT)
    crop = read_json(PROCESSED / "crop_report.json")
    summ = pd.read_csv(ROOT / "reports/tables/data_summary.csv")
    cats = pd.read_csv(ROOT / "reports/tables/data_categories.csv", index_col=0)
    items = read_table(PROCESSED / "items_validation_cropped.parquet")
    items_ok = items[items.cropped]
    crops_gb = sum(p.stat().st_size for p in (PROCESSED / "crops").rglob("*.jpg")) / 1e9
    anno = json.load(open(cfg.paths.deepfashion2_root / "validation/annos/000001.json", encoding="utf-8"))
    anno_short = {k: ({kk: vv for kk, vv in v.items() if kk not in ("segmentation", "landmarks")} if isinstance(v, dict) else v) for k, v in anno.items()}

    f_pair = fig_example_pair(items_ok)
    f_group = fig_pair_group(items_ok, 150)
    f_attr = fig_attribute_levels(items_ok)
    f_cat = fig_categories(items_ok)
    f_crop, A = fig_crop_arith(items, cfg)
    f_flow = fig.flow_diagram(["Ham fotoğraf\n+ JSON etiket", "Ürün tablosu\n(parquet)", "Kırpıntılar\n(JPEG)", "pair_id\nayrımı", "Galeri / sorgu /\naçık küme listeleri"],
                              FIG_DIR / "pipeline.png", "Veri hazırlama akışı")
    cat_rows = sorted([[cid, CATS[cid], int(cats.loc[cid, "test/queries"]), int(cats.loc[cid, "test/gallery"])] for cid in cats.index], key=lambda x: -x[2])
    f_catbar = fig.hbar([f"{c[1]} ({c[0]})" for c in cat_rows], [c[2] for c in cat_rows], FIG_DIR / "test_query_categories.png", "Test sorgularının kategori dağılımı", "sorgu sayısı")

    r = Report(
        title="Adım 1 — Veri Seti ve Veri Hazırlama",
        subtitle="DeepFashion2 nedir, içinde ne var, ve onu bu projede kullanılabilir hâle getirmek için ne yapıldı",
        author="Ethem Belka Şahin",
        course="CMPE 9103 Computer Vision · Dönem Projesi İlerleme Raporu · İstanbul Üniversitesi",
    )

    # ------------------------------------------------------------------ 1
    r.h1("1. Problem: neyi arıyoruz?")
    r.para(
        "Bir müşteri, beğendiği bir kıyafetin fotoğrafını telefonuyla çekiyor. Sistem bu fotoğrafa bakıp bir mağaza kataloğundaki "
        "binlerce ürün fotoğrafı arasından aynı ürünü bulmalı. Buna literatürde consumer-to-shop retrieval (müşteriden mağazaya erişim) denir. "
        "'Aynı ürün' demek, aynı modelin aynı renk/desen varyantı demek; aynı türden bir başka tişört doğru cevap sayılmaz."
    )
    r.figure(f_pair, "Şekil 1. Soldaki müşteri fotoğrafındaki kıyafet (mavi kutu) ile sağdaki mağaza fotoğrafındaki kıyafet aynı üründür. "
             "Işık, poz, arka plan ve çekim açısı tamamen farklı; sistemin aşması gereken zorluk budur.", width_in=5.5)
    r.para(
        "Projenin ikinci sorusu: ürün katalogda hiç yoksa sistem bunu fark edebilir mi? Normal karşılaştırma protokollerinde her sorgunun "
        "bir cevabı olduğu varsayılır; gerçek hayatta öyle değildir. Bu rapor henüz bu soruya girmiyor, ama veri hazırlığı onu da düşünerek yapıldı."
    )

    # ------------------------------------------------------------------ 2
    r.h1("2. Veri seti: DeepFashion2")
    r.para(
        "DeepFashion2 (Ge vd., CVPR 2019), Çin'deki e-ticaret sitelerinden toplanmış 491 bin kıyafet fotoğrafından oluşan, "
        "herkese açık bir araştırma veri setidir. Her fotoğraf insan eliyle etiketlenmiştir. Bu projede Kaggle'da yayınlanan 15 GB'lık kopyası kullanıldı."
    )
    r.h2("2.1 Fotoğraflar iki kaynaktan geliyor: müşteri ve mağaza")
    r.para(
        "Veri setindeki her fotoğrafın bir kaynağı (source) vardır: 'user' = müşterinin kendi çektiği fotoğraf (ayna selfie'si, oda, sokak), "
        "'shop' = satıcının ürün sayfasına koyduğu fotoğraf (stüdyo, manken, düz zemin). Aynı ürünün hem müşteri hem mağaza fotoğrafları vardır; "
        "projenin bütün fikri bu ikisini eşleştirmektir."
    )
    r.h2("2.2 Bir fotoğrafın etiket dosyası")
    r.para(
        "Her fotoğrafın yanında aynı adı taşıyan bir JSON dosyası bulunur. JSON, metin biçiminde bir sözlüktür: anahtar–değer çiftleri. "
        "Aşağıda validation bölümündeki 000001.jpg fotoğrafının etiket dosyası, bu projede kullanılmayan iki alan (piksel maskesi ve vücut noktaları) çıkarılmış hâlde:"
    )
    r.mono(json.dumps(anno_short, indent=1, ensure_ascii=False))
    r.para("Bu dosyadaki alanların anlamı:")
    r.definition("source", "fotoğrafı kimin çektiği: 'user' ya da 'shop'. Fotoğraf başına tek değer.")
    r.definition("pair_id", "ürün grubu numarası. Aynı ürün sayfasına ait tüm fotoğraflar (müşteri çekimleri + mağaza çekimleri) aynı pair_id'yi taşır. "
                 "Yukarıdaki fotoğrafın pair_id'si 1; pair_id'si 1 olan başka fotoğraflar da aynı ürün sayfasından gelir.")
    r.definition("item1, item2, …", "fotoğraftaki her bir kıyafet. Bir fotoğrafta birden fazla kıyafet olabilir (üst + alt gibi); her biri ayrı bir item'dır.")
    r.definition("bounding_box", "kıyafetin fotoğraftaki yeri: [x1, y1, x2, y2] = sol-üst köşenin ve sağ-alt köşenin piksel koordinatları. "
                 f"Yukarıdaki item2 için [204, 189, 293, 414]: kutu x=204'ten 293'e (genişlik {A['bw']} piksel), y=189'dan 414'e (yükseklik {A['bh']} piksel).")
    r.definition("category_id / category_name", "13 kategoriden biri (Şekil 4).")
    r.definition("style", "bu kıyafetin ürün grubu içindeki varyant numarası — aşağıda ayrıca açıklanıyor.")
    r.definition("scale, occlusion, zoom_in, viewpoint", "çekim koşulu etiketleri; her biri 1, 2 ya da 3 (Şekil 3).")

    r.h2("2.3 style ne demek? pair_id ile birlikte ürünün kimliği")
    r.para(
        "Bir ürün sayfasında (bir pair_id'de) genellikle birden fazla kıyafet görünür: satılan tişört, mankenin giydiği şort, aynı tişörtün başka "
        "renkte bir varyantı. style, bunları birbirinden ayıran numaradır:"
    )
    r.bullets([
        "style = 1, 2, 3 … : satılan ürünün varyantları. Aynı pair_id içinde style 1 (örneğin beyaz) ile style 2 (siyah) farklı ürünlerdir.",
        "style = 0 : bu kıyafet o ürün sayfasında satılan şey değil (mankenin üzerindeki diğer parça, arka plandaki başka bir kıyafet). "
        "Dolayısıyla style = 0 bir müşteri kıyafetinin katalogda tanımlı bir karşılığı yoktur.",
    ])
    r.para(
        "Bu projede bir ürünün kimliği (pair_id, style) ikilisidir; kodda bu ikili 'match_key' adıyla \"150_2\" gibi tek bir metne çevrilir. "
        "Bir müşteri fotoğrafı ile bir mağaza fotoğrafı ancak match_key'leri aynıysa 'aynı ürün' sayılır."
    )
    r.figure(f_group, "Şekil 2. pair_id = 150 ürün grubundaki bütün kıyafetler. Üst sıra müşteri, alt sıra mağaza fotoğraflarından kesilmiş. "
             "Satılan ürün bir tişört; mağazada hem style 1 (sarı çerçeve) hem style 2 (turuncu) varyantı var, müşteri style 2'yi çekmiş. "
             "Mankenin giydiği şortlar style 0 (yeşil): ürün sayfasının konusu değil, katalogda karşılığı yok.", width_in=6.3)
    r.note(
        "Neden önemli? Açık küme (katalogda yok) deneyi için ek etiketleme gerekmiyor: style = 0 olan müşteri kıyafetleri, tanımı gereği "
        "katalogda eşi olmayan sorgulardır. Yalnız bir kısıt var: aynı ürün grubunun mağaza fotoğraflarında bu kıyafet de görünür "
        "(örneğin şortlar). Bu kısıt Adım 3 raporunda tartışılacak."
    )

    r.h2("2.4 Çekim koşulu etiketleri")
    r.para("Her kıyafet için dört koşul etiketi vardır; başarının hangi koşulda düştüğünü ölçmek için (Araştırma Sorusu 3) kullanılacaklar:")
    r.bullets([
        "scale (ölçek): kıyafetin fotoğraf içinde kapladığı alan — 1 küçük (fotoğrafın <%10'u), 2 orta, 3 büyük (>%40).",
        "occlusion (kapanma): kıyafetin ne kadarı görünmüyor (kol, çanta, başka giysi, fotoğraf kenarı) — 1 hafif, 2 orta, 3 ağır (yarıdan fazlası kapalı).",
        "zoom_in (yakınlaştırma): fotoğraf kıyafete ne kadar yakından çekilmiş — 1 yok (tüm beden görünüyor), 2 orta, 3 çok (yalnızca bir parça görünüyor).",
        "viewpoint (bakış açısı): 1 giyilmemiş (askıda ya da düz serilmiş), 2 önden giyilmiş, 3 yandan ya da arkadan giyilmiş.",
    ])
    r.figure(f_attr, "Şekil 3. Dört koşul etiketinin üç düzeyine birer örnek. Kırpıntılar müşteri fotoğraflarından.", width_in=6.0)

    r.h2("2.5 Kategoriler")
    r.figure(f_cat, "Şekil 4. 13 kategori. Üstler (1–6), altlar (7–9) ve elbiseler (10–13).", width_in=6.3)

    r.h2("2.6 Veri setinin bölümleri")
    r.table(
        ["Bölüm", "Fotoğraf", "Kıyafet (item)", "Etiket", "Bu projede"],
        [["train", 191961, crop["train"]["total"], "var", "Eğitim + iç doğrulama"],
         ["validation", 32153, crop["validation"]["total"], "var", "Test (resmi retrieval listeleriyle)"],
         ["test", 62629, "—", "YOK", "Kullanılmıyor"]],
        caption="Tablo 1. DeepFashion2'nin bölümleri")
    r.para(
        "Veri setinin yazarları 'test' bölümünün etiketlerini yayınlamamıştır (yarışma amaçlı). Etiket olmadan doğru cevap bilinemez, "
        "başarı ölçülemez. Bu yüzden, literatürdeki her çalışma gibi, 'validation' bölümü bu projede test seti olarak kullanılır. "
        "Yazarlar bunun için resmi iki liste de yayınlamıştır: val_query.json (12.377 müşteri kıyafeti = sorgular) ve val_gallery.json "
        "(36.961 mağaza kıyafeti = katalog). Sonuçlar bu listelerle hesaplanınca makaledekiyle karşılaştırılabilir olur."
    )

    # ------------------------------------------------------------------ 3
    r.h1("3. Proje önerisinde ne söz verdik?")
    r.bullets([
        "Her kıyafet, etiketteki kutusundan kesilecek; kıyafeti fotoğrafta bulma (tespit) projenin konusu değil.",
        "Resmi val_query / val_gallery listeleri test seti olacak.",
        "Eğitim verisindeki ürün gruplarının %10'u model seçimi için ayrılacak; aynı ürün hem eğitimde hem değerlendirmede olmayacak.",
        "style = 0 müşteri kıyafetleri, ek etiket olmadan 'katalogda yok' sorguları olacak.",
    ])

    # ------------------------------------------------------------------ 4
    r.h1("4. Ne yapıldı? — beş işlem")
    r.figure(f_flow, "Şekil 5. Ham veriden değerlendirme listelerine giden beş işlem. Her biri aşağıda ayrı anlatılıyor.", width_in=6.3)

    r.h2("4.1 Etiket dosyalarını tek bir tabloya çevirmek")
    r.para("Amaç: 224.114 ayrı JSON dosyası yerine, her satırı bir kıyafet olan tek bir tablo üzerinde çalışmak; 'style = 0 olan müşteri kıyafetleri' gibi seçimler böylece tek satırlık süzgeçler olur.", "")
    r.para(
        "Yapılan: Her JSON dosyası okundu; içindeki her item için bir satır üretildi. Satır, fotoğraf düzeyindeki alanları (source, pair_id) "
        "ve kıyafet düzeyindeki alanları (kutu, kategori, style, dört koşul) yan yana taşır. Kutu [x1, y1, x2, y2] dört ayrı sütuna açıldı. "
        "8 işlemci çekirdeği paralel çalıştı. Sonuç iki tablo: items_train.parquet (312.186 satır) ve items_validation.parquet (52.490 satır)."
    )
    sample = items_ok.head(3)[["item_id", "image_id", "source", "pair_id", "style", "category_id", "x1", "y1", "x2", "y2", "scale", "occlusion", "zoom_in", "viewpoint", "match_key"]]
    r.mono(sample.to_string(index=False))
    r.definition("parquet", "tabloları sütun sütun, sıkıştırılmış ve tip bilgisiyle saklayan bir dosya biçimi; CSV'den küçük ve okuması hızlı. İçerik yine satır–sütun tablosudur.")
    r.definition("item_id", "bu projede kıyafete verilen benzersiz ad: bölüm + fotoğraf numarası + item anahtarı. Kırpıntı dosyasının adı da budur.")

    r.h2("4.2 Kıyafetleri fotoğraftan kesmek (kırpma)")
    r.para(
        "Amaç: Model fotoğrafın tamamını değil yalnızca kıyafeti görmeli. Müşteri fotoğrafında oda, ayna, yüz, diğer giysiler var; "
        "bunlar karşılaştırmaya girerse 'aynı oda' ile 'aynı tişört' birbirine karışır."
    )
    r.para(
        "Bir dijital fotoğraf, yükseklik × genişlik × 3 boyutlu bir sayı matrisidir: her pikselin kırmızı, yeşil, mavi değeri (0–255). "
        f"Örnek fotoğraf 000001.jpg {A['W']}×{A['H']} piksel. Kırpma, bu matristen bir dikdörtgen alt-matris almaktır. Adımlar, yukarıdaki item2 (kutu [{A['x1']}, {A['y1']}, {A['x2']}, {A['y2']}]) üzerinde:"
    )
    r.numbered([
        f"Kutuyu büyüt: kutu genişliği {A['bw']}, yüksekliği {A['bh']} piksel. Her kenara kendi boyunun %{int(A['margin']*100)}'u kadar pay eklendi: "
        f"yatayda {A['dx']}, dikeyde {A['dy']} piksel. Yeni kutu [{A['box'][0]}, {A['box'][1]}, {A['box'][2]}, {A['box'][3]}] (fotoğraf kenarını aşan kısım kesilir). "
        "Neden: etiket kutuları bazen kıyafetin kenarını tam içine almaz; küçük bir pay yaka ve etek ucunu güvenceye alır.",
        f"Alt-matrisi al: satır {A['box'][1]}–{A['box'][3]}, sütun {A['box'][0]}–{A['box'][2]} → {A['cw']}×{A['ch']} piksellik yeni bir görüntü.",
        f"Küçült: kısa kenar {A['short']} piksel olacak şekilde yeniden örnekle (bicubic enterpolasyon: her yeni pikselin değeri, eski görüntüde "
        f"karşılık gelen konumun çevresindeki 4×4 pikselin ağırlıklı ortalaması). Örnekte {A['cw']}×{A['ch']} → {A['real'][0]}×{A['real'][1]}. "
        "Neden: modellerin hepsi 224 piksellik giriş kullanıyor; daha büyük saklamak disk ve zaman israfı. Kısa kenar zaten 256'dan küçükse büyütülmez (bilgi uydurmanın anlamı yok).",
        f"JPEG olarak kaydet (kalite {cfg.crop.jpeg_quality}): data/processed/crops/validation/validation_000001_item2.jpg.",
    ])
    r.figure(f_crop, "Şekil 6. Aynı örnek: mavi = etiketteki kutu, turuncu = %10 payla büyütülmüş kutu, sağda ortaya çıkan kırpıntı.", width_in=5.5)
    r.para("Dikkat edilenler:", "")
    r.bullets([
        f"Kenarı {cfg.crop.min_side_px} pikselden küçük kutular atlandı (train {crop['train']['dropped_too_small']}, validation {crop['validation']['dropped_too_small']} adet). "
        "Bu kadar küçük bir yamada kıyafet ayırt edilemez; sayı o kadar küçük ki sonuçları etkilemez ama raporlanır.",
        "Zaten kesilmiş bir kıyafet yeniden kesilmez; iş yarıda kesilirse kaldığı yerden sürer. Toplam 1 saat sürdü (8 çekirdek).",
        f"Bozuk/okunamayan fotoğraf: 0. Toplam {crop['train']['cropped'] + crop['validation']['cropped']:,} kırpıntı, {crops_gb:.1f} GB (ham veri 15 GB).".replace(",", "."),
    ])
    r.table(["Bölüm", "Toplam kıyafet", "Kesilen", "Çok küçük (atlandı)", "Hata"],
            [[s, crop[s]["total"], crop[s]["cropped"], crop[s]["dropped_too_small"], crop[s]["errors"]] for s in ("train", "validation")],
            caption="Tablo 2. Kırpma sonucu")

    r.h2("4.3 Eğitim verisini ikiye bölmek: train_fit ve val_internal")
    r.para(
        "Amaç: İleride bir model eğitildiğinde (Adım 4) hangi ayarın iyi olduğuna karar vermek için 'görülmemiş' bir veri gerekir. "
        "Bu karar test setinde verilemez; test yalnızca en sonda, bir kez, raporlamak için kullanılır. Aksi hâlde test sonucu iyimser olur."
    )
    r.para(
        f"Yapılan: train bölümündeki {len(pd.read_csv(PROCESSED/'splits/train_fit_pair_ids.csv')) + len(pd.read_csv(PROCESSED/'splits/val_internal_pair_ids.csv')):,} farklı pair_id numarası alındı, sabit bir tohumla (seed = {cfg.splits.seed}) karıştırıldı, "
        f"%{int(cfg.splits.internal_val_ratio*100)}'u val_internal'a, kalanı train_fit'e verildi. Bir pair_id'nin bütün fotoğrafları (müşteri + mağaza) aynı tarafa düşer.".replace(",", ".")
    )
    r.para("Dikkat edilen:", "")
    r.bullets([
        "Bölme fotoğraf düzeyinde değil ürün grubu düzeyinde yapıldı. Fotoğraf düzeyinde bölünseydi aynı ürünün müşteri fotoğrafı eğitimde, "
        "mağaza fotoğrafı doğrulamada kalabilirdi; model o ürünü 'ezberleyip' doğrulamada hak etmediği puan alırdı (veri sızıntısı).",
        "Sabit tohum: aynı komut her çalıştırıldığında aynı bölme çıkar; deneyler tekrarlanabilir.",
        "İki kümenin pair_id listeleri git'e konuldu ve bir otomatik test kesişimin boş olduğunu her çalıştırmada doğrular.",
    ])

    r.h2("4.4 Galeri, sorgu ve 'katalogda yok' listelerini kurmak")
    r.para("Amaç: Her ayrım için 'katalog hangi fotoğraflardan oluşuyor, sorgular hangileri, katalogda olmayan sorgular hangileri' sorularını kesin kurallarla cevaplamak.")
    r.table(["Liste", "Kural", "Neden"],
            [["Galeri (katalog)", "source = shop ve style > 0", "Katalog mağaza fotoğraflarıdır; style = 0 mağaza kıyafetleri satılan ürün olmadığı için katalogda yer almaz"],
             ["Sorgular (kapalı küme)", "source = user, style > 0 ve galeride aynı match_key'e sahip en az bir kıyafet var", "Cevabı katalogda olmayan bir sorguyla kapalı küme başarısı ölçülemez"],
             ["Açık küme sorguları", "source = user ve style = 0", "Tanımı gereği katalogda karşılığı yok"]],
            caption="Tablo 3. Liste kuralları (train_fit, val_internal ve test'in açık küme listesi için)")
    r.para(
        "Test ayrımı için galeri ve sorgu listeleri bu kurallarla değil, veri setinin resmi val_gallery.json ve val_query.json dosyalarıyla kuruldu; "
        "böylece sayılar makaleyle karşılaştırılabilir. Resmi listedeki her kayıt (fotoğraf numarası + kutu) bizim tablomuzdaki satırla eşlendi; "
        f"kenarı küçük olduğu için kesilmemiş {crop['test_official_missing']['queries']} sorgu ve {crop['test_official_missing']['gallery']} galeri kaydı listeye giremedi."
    )
    r.note(
        "Fark edilen bir nokta: resmi test galerisi style = 0 mağaza kıyafetlerini de içeriyor (36.961'in 13.088'i). Yani test kataloğunda, hiçbir "
        "sorguyla eşleşmeyen 13 bin 'dikkat dağıtıcı' var; val_internal galerisinde ise yok. Bu, test'i biraz daha zor yapar; raporda belirtilecek."
    )
    rows = [[split, lst, n(summ, split, lst), n(summ, split, lst, "n_products")] for split in ("train_fit", "val_internal", "test") for lst in ("gallery", "queries", "openset")]
    r.table(["Ayrım", "Liste", "Kıyafet sayısı", "Farklı ürün (match_key) sayısı"], rows, caption="Tablo 4. Üretilen listeler")
    r.figure(f_catbar, "Şekil 7. Test sorgularının kategorilere dağılımı. Üstler ve etekler çoğunlukta; 'short sleeve outwear' (32) ve 'sling' (139) çok az.", width_in=5.8)

    r.h2("4.5 Dosyaları Colab'a taşınabilir yapmak")
    r.para(
        "Ağır GPU işleri Google Colab'da yapılacağı için kırpıntılar ve listeler tek bir arşivde toplandı (crops.zip, 7,7 GB, 364.592 dosya). "
        "Listelerdeki dosya yolları bilgisayara özgü tam yol yerine göreli yazıldı (crops/validation/….jpg), böylece aynı listeler Colab'da da çalışır. "
        "Ham 15 GB'lık veri Drive'a yüklenmedi; Colab'da hiçbir adım ona ihtiyaç duymuyor."
    )

    # ------------------------------------------------------------------ 5
    r.h1("5. Bu adımın sonunda elimizde ne var?")
    r.bullets([
        f"{crop['train']['cropped'] + crop['validation']['cropped']:,} kıyafet kırpıntısı (JPEG), her biri kısa kenarı ≤256 piksel.".replace(",", "."),
        "Her kırpıntı için kimlik (match_key), kategori ve dört koşul etiketini taşıyan tablolar.",
        "Üç ayrım: train_fit (eğitim), val_internal (seçim), test (rapor); test resmi listelerle.",
        f"Her ayrımda galeri / sorgu / açık küme listeleri: test'te {n(summ,'test','gallery'):,} galeri, {n(summ,'test','queries'):,} sorgu, {n(summ,'test','openset'):,} açık küme sorgusu.".replace(",", "."),
        "Hiçbir fotoğraf değiştirilmedi; yalnızca kesildi ve küçültüldü. Etiketlere elle müdahale yok.",
    ])
    r.h1("6. Sıradaki adım")
    r.para(
        "Adım 2'de dört hazır model bu kırpıntıları sayı dizilerine (vektörlere) çevirecek ve her sorgu için katalogdaki en benzer vektörler "
        "bulunacak. Vektör, benzerlik ve bu modellerin ne olduğu Adım 2 raporunda anlatılıyor."
    )
    out = r.save(ROOT / "docs/reports/01_veri_hazirlama.docx")
    print("written", out)


if __name__ == "__main__":
    main()
