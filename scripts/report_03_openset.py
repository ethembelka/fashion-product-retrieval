"""Build docs/reports/03_acik_kume_ve_kirilimlar.docx — the open-set problem, the two scores, AUROC, the
coverage–precision curve and the condition breakdowns, explained with worked examples from real queries.

Usage: python scripts/report_03_openset.py [--model siglip_vitb16]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fashret.evaluation.breakdown import ATTRIBUTES  # noqa: E402
from fashret.reporting import figures as fig  # noqa: E402
from fashret.reporting.docx_report import Report  # noqa: E402
from fashret.utils.io import read_json, read_table  # noqa: E402

FIG_DIR = ROOT / "reports/figures/03_openset"
PROCESSED = ROOT / "data/processed"
MODEL_LABEL = {"resnet50": "ResNet-50", "clip_vitb16": "CLIP ViT-B/16", "siglip_vitb16": "SigLIP ViT-B/16", "dinov2_vitb14": "DINOv2 ViT-B/14"}
ATTR_TR = {"occlusion": "Kapanma (occlusion)", "scale": "Ölçek (scale)", "zoom_in": "Yakınlaştırma (zoom_in)", "viewpoint": "Bakış açısı (viewpoint)", "category_id": "Kategori"}
LEVEL_TR = {1: "1 (küçük / hafif / yok / giyilmemiş)", 2: "2 (orta / önden)", 3: "3 (büyük / ağır / çok / yandan-arkadan)"}


def run_dir(model: str, split: str) -> Path:
    return sorted((ROOT / "outputs/results").glob(f"*_{model}_zeroshot_{split}_s0"))[-1]


def topk_rows(emb: np.ndarray, g_emb: np.ndarray, g_meta: pd.DataFrame, k: int = 5) -> list[list[object]]:
    sims = g_emb @ emb
    top = np.argsort(-sims)[:k]
    return [[i + 1, g_meta.match_key[j], g_meta.category_id[j], round(float(sims[j]), 3)] for i, j in enumerate(top)]


def worked_examples(model: str) -> tuple[dict, dict]:
    """One known query with a correct, confident top-1 and one open-set query whose top-1 is another product."""
    d = run_dir(model, "test")
    pq, po = read_table(d / "per_query.parquet"), read_table(d / "per_openset.parquet")
    g_meta = read_table(ROOT / f"data/features/{model}/test/gallery_meta.parquet")
    g_emb = np.load(ROOT / f"data/features/{model}/test/gallery_emb.npy")
    q_emb = np.load(ROOT / f"data/features/{model}/test/queries_emb.npy")
    o_emb = np.load(ROOT / f"data/features/{model}/test/openset_emb.npy")
    i = pq.index[(pq.first_hit_rank == 1) & (pq.s1 - pq.s2 > 0.05)][3]
    po["top1_pair"] = po.top1_key.str.split("_").str[0].astype(int)
    k = po.index[(po.top1_pair != po.pair_id) & (po.s1 > 0.85)][2]
    known = dict(item=pq.item_id[i], key=pq.match_key[i], s1=float(pq.s1[i]), s2=float(pq.s2[i]), rows=topk_rows(q_emb[i], g_emb, g_meta), crop=PROCESSED / pq.crop_path[i])
    unknown = dict(item=po.item_id[k], key=po.match_key[k], s1=float(po.s1[k]), s2=float(po.s2[k]), rows=topk_rows(o_emb[k], g_emb, g_meta), crop=PROCESSED / po.crop_path[k])
    return known, unknown


def openset_examples(model: str, n: int = 3) -> Path:
    d = run_dir(model, "test")
    po = read_table(d / "per_openset.parquet")
    g_meta = read_table(ROOT / f"data/features/{model}/test/gallery_meta.parquet")
    g_emb = np.load(ROOT / f"data/features/{model}/test/gallery_emb.npy")
    o_emb = np.load(ROOT / f"data/features/{model}/test/openset_emb.npy")
    rng = np.random.default_rng(1)
    sel = rng.choice(po.index[po.s1 > po.s1.quantile(0.8)].to_numpy(), n, replace=False)
    rows = []
    for i in sel:
        sims = g_emb @ o_emb[i]
        top = np.argsort(-sims)[:5]
        rows.append((PROCESSED / po.crop_path[i], [(PROCESSED / g_meta.crop_path[j], False, float(sims[j])) for j in top]))
    return fig.retrieval_grid(rows, FIG_DIR / f"{model}_openset_examples.png", "Katalogda karşılığı olmayan sorgular (style = 0) ve sistemin getirdiği ilk 5")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="siglip_vitb16")
    model = ap.parse_args().model
    d_test, d_val = run_dir(model, "test"), run_dir(model, "val_internal")
    m, mv = read_json(d_test / "metrics.json"), read_json(d_val / "metrics.json")
    o, ov = m["openset"], mv["openset"]
    pq, po = read_table(d_test / "per_query.parquet"), read_table(d_test / "per_openset.parquet")
    curves = {n: pd.read_csv(d_test / f"coverage_curve_{n}.csv") for n in ("abs", "margin")}
    bt = pd.read_csv(d_test / "breakdown.csv")
    all_m = {k: read_json(run_dir(k, "test") / "metrics.json")["openset"] for k in MODEL_LABEL}
    K, U = worked_examples(model)
    n_known, n_unknown = len(pq), len(po)

    # ---------- figures
    f_hist_abs = fig.hist_overlay({"bilinen, ilk sonuç doğru": pq.s1[pq.top1_correct].to_numpy(), "bilinen, ilk sonuç yanlış": pq.s1[~pq.top1_correct].to_numpy(), "katalogda yok": po.s1.to_numpy()},
                                  FIG_DIR / "hist_abs.png", "Mutlak skor s1 dağılımları", "s1 (en benzer katalog öğesinin kosinüsü)")
    f_hist_margin = fig.hist_overlay({"bilinen, ilk sonuç doğru": (pq.s1 - pq.s2)[pq.top1_correct].to_numpy(), "bilinen, ilk sonuç yanlış": (pq.s1 - pq.s2)[~pq.top1_correct].to_numpy(), "katalogda yok": (po.s1 - po.s2).to_numpy()},
                                     FIG_DIR / "hist_margin.png", "Margin skoru s1 − s2 dağılımları", "s1 − s2")
    f_cov = fig.lines({"mutlak (s1)": (curves["abs"].coverage, curves["abs"].precision), "margin (s1 − s2)": (curves["margin"].coverage, curves["margin"].precision)},
                      FIG_DIR / "coverage_precision.png", "Kapsama–kesinlik eğrisi (test)", "kapsama (cevap verilen sorgu oranı)", "kesinlik (cevap verilenlerde ilk sonuç doğru oranı)",
                      xlim=(0, 1), marker_at={"mutlak (s1)": (o["at_tau_abs"]["coverage"], o["at_tau_abs"]["precision"]), "margin (s1 − s2)": (o["at_tau_margin"]["coverage"], o["at_tau_margin"]["precision"])})
    f_fmr = fig.lines({"mutlak (s1)": (curves["abs"].coverage, curves["abs"].false_match_rate), "margin (s1 − s2)": (curves["margin"].coverage, curves["margin"].false_match_rate)},
                      FIG_DIR / "coverage_fmr.png", "Kapsama – yanlış eşleşme oranı", "kapsama", "katalogda olmayanların eşiği geçme oranı", xlim=(0, 1), ylim=(0, 1))
    cond = bt[bt.attribute != "category_id"]
    f_break = fig.grouped_hbar([ATTR_TR[a] for a in ATTRIBUTES[:-1]],
                               {LEVEL_TR[v]: [float(cond[(cond.attribute == a) & (cond.value == v)]["recall@1"].iloc[0]) for a in ATTRIBUTES[:-1]] for v in (1, 2, 3)},
                               FIG_DIR / "breakdown_conditions.png", "Recall@1 — çekim koşuluna göre (test)", fmt="{:.3f}")
    cat = bt[bt.attribute == "category_id"].sort_values("recall@1", ascending=False)
    f_cat = fig.hbar([f"{r.label} (n={r.n})" for r in cat.itertuples()], list(cat["recall@1"]), FIG_DIR / "breakdown_category.png", "Recall@1 — kategoriye göre (test)", "Recall@1", fmt="{:.3f}")
    f_ex = openset_examples(model)

    r = Report(
        title="Adım 3 — Açık Küme Analizi ve Kırılımlar",
        subtitle=f"{MODEL_LABEL[model]} 'katalogda yok' diyebilir mi? Başarı hangi çekim koşullarında düşüyor?",
        author="Ethem Belka Şahin",
        course="CMPE 9103 Computer Vision · Dönem Projesi İlerleme Raporu · İstanbul Üniversitesi",
    )

    # ------------------------------------------------------------------ 1
    r.h1("1. Problem: kapalı küme ve açık küme")
    r.para(
        "Adım 2'deki ölçüm kapalı kümeydi: her sorgunun katalogda en az bir doğru eşi vardı; soru yalnızca 'kaçıncı sırada?' idi. "
        "Gerçek bir uygulamada müşteri, mağazanın satmadığı bir kıyafetin fotoğrafını da gönderir. Sistem o zaman ne yapar? Arama her zaman bir "
        "sonuç döndürür — en benzer katalog öğesi her zaman vardır — ama bu sonuç yanlıştır. Açık küme problemi, sistemin 'bu ürün katalogda yok' diyebilmesidir (Araştırma Sorusu 4)."
    )
    r.para(
        f"Veri: 'bilinen' sorgular = Adım 2'deki {n_known:,} test sorgusu (hepsinin katalogda eşi var). 'Katalogda yok' sorgular = style = 0 olan {n_unknown:,} müşteri kıyafeti "
        "(Adım 1 raporu, Bölüm 2.3: ürün sayfasının konusu olmayan kıyafetler). Katalog aynı 36.958 öğelik test galerisi.".replace(",", ".")
    )

    # ------------------------------------------------------------------ 2
    r.h1("2. Karar nasıl verilir? İki güven skoru")
    r.para(
        "Sistem her sorgu için kataloğu sıralar ve bir güven skoru üretir; skor bir eşiğin (τ) üstündeyse ilk sonucu cevap olarak verir, altındaysa 'katalogda yok' der. "
        "Soru, hangi skorun iyi çalıştığı. Önerideki iki aday:"
    )
    r.definition("mutlak skor s1", "ilk sıradaki katalog öğesinin kosinüs benzerliği. Mantık: aynı ürün çok yüksek benzerlik verir, katalogda olmayan ürün vermez.")
    r.definition("margin s1 − s2", "s2 = ilk sonuçtan FARKLI bir ürüne (farklı match_key) ait ilk öğenin benzerliği. Mantık: ürün katalogdaysa onun fotoğrafları "
                 "diğer ürünlerden açık ara önde olur; katalogda yoksa birkaç benzer ürün yakın skorlarla yarışır, fark küçük kalır.")
    r.para("İki gerçek sorgu üzerinde:")
    r.table(["Sıra", "match_key", "kategori", "benzerlik"], K["rows"], caption=f"Tablo 1. Bilinen sorgu {K['item']} (ürün {K['key']}): sıralamanın ilk 5'i")
    r.para(
        f"İlk 5 sonucun hepsi doğru ürünün ({K['key']}) farklı fotoğrafları. s1 = {K['s1']:.3f}. Farklı bir ürüne ait ilk sonuç daha aşağıda, benzerliği s2 = {K['s2']:.3f}. "
        f"Margin = {K['s1'] - K['s2']:.3f}: doğru ürün diğerlerinden açıkça ayrılmış."
    )
    r.table(["Sıra", "match_key", "kategori", "benzerlik"], U["rows"], caption=f"Tablo 2. Katalogda olmayan sorgu {U['item']} (style = 0): sıralamanın ilk 5'i")
    r.para(
        f"Bu kıyafetin katalogda eşi yok, ama sistem s1 = {U['s1']:.3f} ile bir ürün ({U['rows'][0][1]}) buluyor — bilinen sorgudakinden yüksek bir skor. "
        f"s2 = {U['s2']:.3f}, margin = {U['s1'] - U['s2']:.3f}: ilk sonuç ile ikinci farklı ürün neredeyse eşit. Mutlak skor 'katalogda var' derdi; margin 'emin değilim' diyor. "
        "Bu iki örnek bütün resmi anlatmaz; aşağıda 15 bin sorgunun toplu sonucu var."
    )

    # ------------------------------------------------------------------ 3
    r.h1("3. Ölçüler")
    r.h2("3.1 AUROC: bir skor iki grubu ne kadar ayırıyor?")
    r.para(
        "Elimizde iki grup skor var: bilinen sorguların skorları ve katalogda olmayanların skorları. Skor işe yarıyorsa bilinenler yüksek, olmayanlar düşük olmalı. "
        "AUROC şöyle hesaplanır: bir gruptan bir skor, diğer gruptan bir skor rastgele seçildiğinde bilinenin daha yüksek çıkma olasılığı. "
        "Bütün ikililer sayılır: 12.375 × 3.137 çift; bilinen > olmayan olan çiftlerin oranı. 1,0 = kusursuz ayrım, 0,5 = yazı-tura (skorun bilgisi yok), 0,5'in altı = ters yönde bilgi."
    )
    r.para("Küçük örnek: bilinen skorlar {0,7, 0,3}, katalogda yok skorları {0,5, 0,1}. Dört çift: (0,7 > 0,5) evet, (0,7 > 0,1) evet, (0,3 > 0,5) hayır, (0,3 > 0,1) evet → AUROC = 3/4 = 0,75.")
    r.para(
        "Aynı ölçü ikinci bir soru için de kullanıldı: bilinen sorgular içinde 'ilk sonucu doğru olanlar' ile 'yanlış olanlar' skorla ayrılıyor mu? "
        "Bu, 'katalogda var mı' sorusundan farklı: 'verdiğim cevaba güvenebilir miyim' sorusu."
    )
    r.h2("3.2 Kapsama–kesinlik eğrisi")
    r.para(
        "Eşik τ'yu yükselttikçe sistem daha az sorguya cevap verir ama cevapları daha güvenilir olur. Her olası τ için iki sayı: "
        "kapsama = bilinen sorguların kaçta kaçının skoru τ'yu geçti (cevap verildi); kesinlik = cevap verilenlerde ilk sonucun doğru olma oranı. "
        "Bir de yanlış eşleşme oranı: katalogda olmayan sorguların kaçta kaçı τ'yu geçti (yani yanlış yere cevap aldı). "
        "τ, bilinen skorların yüzdelik dilimleri üzerinde 200 adımda taranır; her adım eğride bir nokta."
    )
    r.para("Küçük örnek, 10 bilinen sorgu, skorlar 0,1'den 1,0'a, ilk sonucu doğru olanlar 0,4, 0,6, 0,7, 0,8, 0,9, 1,0 skorlular. τ = 0,28 seçilirse: 0,3 ve üstü 8 sorgu cevap alır → kapsama 8/10 = 0,8; "
           "bunların 6'sı doğru → kesinlik 6/8 = 0,75. τ = 0,55: kapsama 5/10, kesinlik 5/5 = 1,0. (Bu örnek kodun otomatik testlerinden biridir.)")
    r.h2("3.3 τ nasıl seçildi?")
    r.para(
        f"Kural (öneride verilen söz): τ, test'te değil val_internal'da seçilir. Hedef kapsama %95: val_internal'daki bilinen sorguların skorları sıralanır, "
        f"%5'lik dilimin sınırı τ olur (mutlak: {ov['tau']['abs']:.3f}, margin: {ov['tau']['margin']:.4f}). Bu τ test'e olduğu gibi uygulandı. "
        "Test'te seçilseydi sonuç iyimser olurdu: eşik, ölçüldüğü veriye uydurulmuş olurdu."
    )

    # ------------------------------------------------------------------ 4
    r.h1("4. Sonuç 1: skorlar 'katalogda yok' demeyi bilmiyor")
    r.table(["Model", "AUROC var↔yok (s1)", "AUROC var↔yok (margin)", "AUROC doğru↔yanlış (s1)", "AUROC doğru↔yanlış (margin)"],
            [[MODEL_LABEL[k], v["auroc_abs"], v["auroc_margin"], v["auroc_correct_abs"], v["auroc_correct_margin"]] for k, v in all_m.items()],
            caption="Tablo 3. Dört modelde iki skorun ayırma gücü (test). İlk iki sütun 'katalogda var mı', son iki sütun 'ilk sonuç doğru mu' sorusu")
    r.para(
        f"Bilinen ile katalogda olmayan sorguları ayırmada her iki skor da yazı-tura düzeyinde: SigLIP için s1 {o['auroc_abs']:.2f}, margin {o['auroc_margin']:.2f}. "
        "Diğer üç modelde de aynı. CLIP'in s1 değeri 0,29: katalogda olmayan sorgular bilinenlerden bile yüksek skor alıyor."
    )
    r.figure(f_hist_abs, "Şekil 1. Mutlak skor dağılımları. Üç grup üst üste; yatay eksende hiçbir noktaya konan eşik katalogda olmayanları ayıramaz.", width_in=6.0)
    r.para(
        f"Neden? Katalogda olmayan kıyafet için de sistem neredeyse aynı skorla (ortanca s1 ≈ {po.s1.median():.2f}, bilinenlerde {pq.s1.median():.2f}) benzer görünen bir ürün buluyor. "
        "Model, Adım 2 raporunda anlatıldığı gibi görsel/anlamsal benzerlik öğrenmiş: 'siyah kot şort' ile 'bir başka siyah kot şort' arasında fark görmüyor. "
        "Ürün kimliği (bu özel şort) onun eğitim hedefinde yoktu."
    )
    r.note(
        f"Kontrol edilen bir kısıt: resmi test galerisi style = 0 mağaza kıyafetlerini içerdiği için, katalogda olmayan bir sorgu bazen kendi ürün sayfasındaki mağaza fotoğrafında "
        f"görünen aynı kıyafeti bulabilir (ilk sonuç aynı pair_id'den: %{o['openset_top1_same_pair_id']*100:.1f}). Bu durumda 'yok' etiketi tartışmalıdır. Aynı pair_id'li galeri öğeleri "
        f"dışlanarak ölçüm tekrarlandı ('strict'): AUROC s1 {o['auroc_abs_strict']:.2f}, margin {o['auroc_margin_strict']:.2f}. Sonuç değişmiyor."
    )
    r.para(
        f"Buna karşılık ikinci soruda — 'verdiğim ilk sonuç doğru mu?' — margin açık bir sinyal: AUROC {o['auroc_correct_margin']:.2f} (mutlak skorla {o['auroc_correct_abs']:.2f}). "
        f"Ortanca margin: doğru cevaplarda {(pq.s1-pq.s2)[pq.top1_correct].median():.3f}, yanlış cevaplarda {(pq.s1-pq.s2)[~pq.top1_correct].median():.3f}, katalogda olmayanlarda {(po.s1-po.s2).median():.3f}. "
        "Doğru cevapta ilk sonuç rakiplerinden kopuyor; yanlış cevapta ve katalogda olmayanda birkaç benzer ürün başa baş gidiyor."
    )
    r.figure(f_hist_margin, "Şekil 2. Margin dağılımları. 'İlk sonuç doğru' grubu sağa yayılmış; 'yanlış' ve 'katalogda yok' grupları sıfıra yığılmış ve birbirinden ayırt edilemiyor.", width_in=6.0)

    # ------------------------------------------------------------------ 5
    r.h1("5. Sonuç 2: eşik, yanlış cevapları azaltır ama 'yok' diyemez")
    r.figure(f_cov, "Şekil 3. Kapsama–kesinlik. Margin eğrisi her kapsamada mutlak skorun üstünde. Noktalar: val_internal'da seçilen τ'nun test'te düştüğü yer.", width_in=5.8)
    r.table(["Skor", "Kesinlik @%50 kapsama", "@%80", "@%95", "τ (val_internal)", "Test'te kapsama", "Test'te kesinlik", "Yanlış eşleşme oranı"],
            [[n_tr, o[f"precision@cov50_{n}"], o[f"precision@cov80_{n}"], o[f"precision@cov95_{n}"], o["tau"][n], o[f"at_tau_{n}"]["coverage"], o[f"at_tau_{n}"]["precision"], o[f"at_tau_{n}"]["false_match_rate"]]
             for n, n_tr in (("abs", "mutlak s1"), ("margin", "margin s1−s2"))],
            caption="Tablo 4. SigLIP, test: farklı kapsamalarda kesinlik; seçilen τ'nun sonucu")
    r.bullets([
        f"Bütün sorgulara cevap verilirse kesinlik = Recall@1 = {m['recall@1']:.2f}. Margin eşiğiyle yalnızca en emin olunan yarısına cevap verilirse kesinlik {o['precision@cov50_margin']:.2f}'e çıkıyor; mutlak skorla ancak {o['precision@cov50_abs']:.2f}.",
        f"%95 kapsama için seçilen margin τ'su test'te %{o['at_tau_margin']['coverage']*100:.1f} kapsama verdi (hedef %95'e yakın: val_internal'da seçilen eşik test'e taşınabiliyor).",
        f"Ama aynı eşikte katalogda olmayan sorguların %{o['at_tau_margin']['false_match_rate']*100:.0f}'i de eşiği geçiyor. Eşik, 'emin olmadığım cevapları vermeyeyim' için işe yarar; 'bu ürün katalogda yok' için yaramaz.",
    ])
    r.figure(f_fmr, "Şekil 4. Kapsama düştükçe (eşik yükseldikçe) katalogda olmayanların kabul oranı da düşüyor; ama bilinenlerle aynı hızda, yani ayrışma yok.", width_in=5.8)
    r.figure(f_ex, "Şekil 5. Katalogda olmayan üç sorgu ve getirilen ilk 5. Hepsi yanlış ama tür, renk ve kesim olarak çok benzer.", width_in=6.3)
    r.note(
        f"'Benzer alternatif' açısından bu bir başarı: katalogda olmayan sorgularda ilk 5'in %{m['openset_category_consistency@5']*100:.0f}'i sorguyla aynı kategoride "
        f"(bilinenlerde %{m['category_consistency@5']*100:.0f}). Sistem 'aynısı yok ama şunlara bak' diyebilir; yalnızca 'aynısı yok' kısmını kendi başına söyleyemez."
    )

    # ------------------------------------------------------------------ 6
    r.h1("6. Sonuç 3: başarı hangi koşullarda düşüyor?")
    r.para(
        "Hesap: test sorguları, Adım 1'de anlatılan dört koşul etiketine ve kategoriye göre gruplandı; her grupta Recall@1 ayrı hesaplandı. "
        "Etiketler sorgu (müşteri) fotoğrafına aittir. 30'dan az örnekli gruplar işaretlendi (sayıları güvenilir değil)."
    )
    r.figure(f_break, "Şekil 6. Recall@1, sorgu fotoğrafının çekim koşuluna göre.", width_in=6.2)
    rows = [[ATTR_TR[a], rr["label"], int(rr["n"]), float(rr["recall@1"]), float(rr["recall@5"]), "✓" if rr["small_group"] else ""]
            for a in ATTRIBUTES[:-1] for rr in bt[bt.attribute == a].to_dict("records")]
    r.table(["Koşul", "Değer", "n", "R@1", "R@5", "<30 örnek"], rows, caption="Tablo 5. Çekim koşullarına göre kırılım (test, SigLIP)")
    r.figure(f_cat, "Şekil 7. Recall@1 kategoriye göre.", width_in=5.8)
    g = lambda a, v: float(bt[(bt.attribute == a) & (bt.value == v)]["recall@1"].iloc[0])  # noqa: E731
    r.bullets([
        f"Kapanma: hafif {g('occlusion',1):.2f} → orta {g('occlusion',2):.2f} → ağır {g('occlusion',3):.2f}. Kıyafetin yarısı görünmüyorsa vektör, görünen yarıyı ve onu kapatan şeyi (kol, çanta) temsil eder; katalogdaki tam görünümle eşleşmez.",
        f"Ölçek: küçük {g('scale',1):.2f}, orta {g('scale',2):.2f}, büyük {g('scale',3):.2f}. Küçük kutu az piksel taşır; 224'e büyütülünce bulanıklaşır, desen ve doku kaybolur.",
        f"Bakış açısı: giyilmemiş {g('viewpoint',1):.2f}, önden {g('viewpoint',2):.2f}, yandan/arkadan {g('viewpoint',3):.2f}. Katalog fotoğrafları çoğunlukla önden; arkadan çekilen bir tişörtün baskısı görünmez.",
        f"Yakınlaştırma: yok {g('zoom_in',1):.2f}, orta {g('zoom_in',2):.2f}, çok {g('zoom_in',3):.2f}. Belirgin etki yok; orta yakınlaştırma (kıyafet kadrajı doldurur, kesilmez) en iyi.",
        f"Kategori: en kolay {cat.iloc[0]['label']} ({cat.iloc[0]['recall@1']:.2f}), en zor {cat.iloc[-1]['label']} ({cat.iloc[-1]['recall@1']:.2f}). Baskılı/desenli üstler ayırt edici detay taşır; "
        "tek renk pantolon ve askılı elbiseler birbirine çok benzer. 'short sleeve outwear' (n=32) küçük grup.",
        "Bu eğilimler DeepFashion2 makalesindeki Match R-CNN kırılımlarıyla aynı yönde (en zor: küçük ölçek, ağır kapanma, yandan/arkadan).",
    ])

    # ------------------------------------------------------------------ 7
    r.h1("7. Ne öğrendik, sırada ne var?")
    r.bullets([
        "Eğitimsiz SigLIP ürünü iyi buluyor ama 'yok' diyemiyor; çünkü öğrendiği şey görsel benzerlik, ürün kimliği değil. Bu, Adım 4'ün gerekçesi.",
        "Margin, cevabın güvenilirliği için kullanılabilir ve mutlak skordan açıkça iyi. Uygulamada 'düşük marginli cevapları emin değilim diye işaretle' kuralı makul.",
        "Adım 4'te SigLIP vektörlerinin üstüne, DeepFashion2'nin eğitim verisiyle 'aynı ürünü yaklaştır, benzer-ama-farklı ürünü uzaklaştır' hedefiyle küçük bir katman eğitilecek. "
        "Başarılıysa hem Recall@1 hem de açık küme AUROC'u yükselmeli; ikisi de aynı kodla ölçülecek.",
        "Kısıt: val_internal galerisi style = 0 mağaza kıyafetlerini içermiyor, test galerisi içeriyor; τ'nun test'te hedef kapsamadan biraz sapmasının nedeni bu. Nihai raporda belirtilecek.",
    ])
    r.h1("8. Dosyalar")
    r.bullets([
        "outputs/results/<run_id>/{metrics.json, per_query.parquet, per_openset.parquet, coverage_curve_{abs,margin}.csv, breakdown.csv}",
        "reports/figures/03_openset/*.png; src/fashret/evaluation/{open_set.py, breakdown.py}; tests/test_open_set.py (elle hesaplanmış örnekler)",
    ])
    out = r.save(ROOT / "docs/reports/03_acik_kume_ve_kirilimlar.docx")
    print("written", out)


if __name__ == "__main__":
    main()
