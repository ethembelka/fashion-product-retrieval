"""Build docs/reports/03_acik_kume_ve_kirilimlar.docx from outputs/results (SigLIP focus, 4-model AUROC table).

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

from fashret.evaluation.breakdown import ATTRIBUTES, LABELS  # noqa: E402
from fashret.reporting import figures as fig  # noqa: E402
from fashret.reporting.docx_report import Report  # noqa: E402
from fashret.utils.io import read_json, read_table  # noqa: E402

FIG_DIR = ROOT / "reports/figures/03_openset"
PROCESSED = ROOT / "data/processed"
MODEL_LABEL = {"resnet50": "ResNet-50", "clip_vitb16": "CLIP ViT-B/16", "siglip_vitb16": "SigLIP ViT-B/16", "dinov2_vitb14": "DINOv2 ViT-B/14"}
ATTR_TR = {"occlusion": "Kapanma (occlusion)", "scale": "Ölçek (scale)", "zoom_in": "Yakınlaştırma (zoom_in)", "viewpoint": "Bakış açısı (viewpoint)", "category_id": "Kategori"}


def run_dir(model: str, split: str) -> Path:
    return sorted((ROOT / "outputs/results").glob(f"*_{model}_zeroshot_{split}_s0"))[-1]


def openset_examples(model: str, n: int = 3) -> Path:
    d = run_dir(model, "test")
    po = read_table(d / "per_openset.parquet")
    g_meta = read_table(ROOT / f"data/features/{model}/test/gallery_meta.parquet")
    g_emb = np.load(ROOT / f"data/features/{model}/test/gallery_emb.npy")
    o_emb = np.load(ROOT / f"data/features/{model}/test/openset_emb.npy")
    rng = np.random.default_rng(1)
    sel = rng.choice(po.index[po.s1 > po.s1.quantile(0.8)].to_numpy(), n, replace=False)  # confident-looking false matches
    rows = []
    for i in sel:
        sims = g_emb @ o_emb[i]
        top = np.argsort(-sims)[:5]
        rows.append((PROCESSED / po.crop_path[i], [(PROCESSED / g_meta.crop_path[j], False, float(sims[j])) for j in top]))
    return fig.retrieval_grid(rows, FIG_DIR / f"{model}_openset_examples.png", "Katalogda karşılığı olmayan sorgular (style = 0) ve sistemin 'bulduğu' ilk 5")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="siglip_vitb16")
    args = ap.parse_args()
    model = args.model
    d_test, d_val = run_dir(model, "test"), run_dir(model, "val_internal")
    m = read_json(d_test / "metrics.json")
    mv = read_json(d_val / "metrics.json")
    o, ov = m["openset"], mv["openset"]
    pq = read_table(d_test / "per_query.parquet")
    po = read_table(d_test / "per_openset.parquet")
    curves = {n: pd.read_csv(d_test / f"coverage_curve_{n}.csv") for n in ("abs", "margin")}
    bt = pd.read_csv(d_test / "breakdown.csv")
    all_m = {k: read_json(run_dir(k, "test") / "metrics.json")["openset"] for k in MODEL_LABEL}

    # ---------- figures
    f_hist_abs = fig.hist_overlay(
        {"bilinen, top-1 doğru": pq.s1[pq.top1_correct].to_numpy(), "bilinen, top-1 yanlış": pq.s1[~pq.top1_correct].to_numpy(), "katalogda yok": po.s1.to_numpy()},
        FIG_DIR / "hist_abs.png", "Mutlak skor s1 (en benzer katalog öğesinin kosinüsü)", "s1")
    f_hist_margin = fig.hist_overlay(
        {"bilinen, top-1 doğru": (pq.s1 - pq.s2)[pq.top1_correct].to_numpy(), "bilinen, top-1 yanlış": (pq.s1 - pq.s2)[~pq.top1_correct].to_numpy(), "katalogda yok": (po.s1 - po.s2).to_numpy()},
        FIG_DIR / "hist_margin.png", "Margin skoru s1 − s2 (ilk sonuç ile farklı üründen ilk sonucun farkı)", "s1 − s2")
    f_cov = fig.lines(
        {"mutlak (s1)": (curves["abs"].coverage, curves["abs"].precision), "margin (s1 − s2)": (curves["margin"].coverage, curves["margin"].precision)},
        FIG_DIR / "coverage_precision.png", "Kapsama–kesinlik eğrisi (test)", "kapsama (kabul edilen sorgu oranı)", "kesinlik (kabul edilenlerde top-1 doğruluğu)",
        xlim=(0, 1), marker_at={"mutlak (s1)": (o["at_tau_abs"]["coverage"], o["at_tau_abs"]["precision"]), "margin (s1 − s2)": (o["at_tau_margin"]["coverage"], o["at_tau_margin"]["precision"])})
    f_fmr = fig.lines(
        {"mutlak (s1)": (curves["abs"].coverage, curves["abs"].false_match_rate), "margin (s1 − s2)": (curves["margin"].coverage, curves["margin"].false_match_rate)},
        FIG_DIR / "coverage_fmr.png", "Kapsama – yanlış eşleşme oranı (katalogda olmayanların kabul edilme payı)", "kapsama", "yanlış eşleşme oranı", xlim=(0, 1), ylim=(0, 1))
    cond = bt[bt.attribute != "category_id"]
    f_break = fig.grouped_hbar(
        [f"{ATTR_TR[a]}" for a in ATTRIBUTES[:-1]],
        {lab: [float(cond[(cond.attribute == a) & (cond.value == v)]["recall@1"].iloc[0]) if ((cond.attribute == a) & (cond.value == v)).any() else 0 for a in ATTRIBUTES[:-1]] for v, lab in ((1, "değer 1 (küçük / hafif / yok / giyilmemiş)"), (2, "değer 2 (orta / önden)"), (3, "değer 3 (büyük / ağır / yandan-arkadan)"))},
        FIG_DIR / "breakdown_conditions.png", "Recall@1 — çekim koşuluna göre (test)", fmt="{:.3f}")
    cat = bt[bt.attribute == "category_id"].sort_values("recall@1", ascending=False)
    f_cat = fig.hbar([f"{r.label} (n={r.n})" for r in cat.itertuples()], list(cat["recall@1"]), FIG_DIR / "breakdown_category.png", "Recall@1 — kategoriye göre (test)", "Recall@1", fmt="{:.3f}")
    f_ex = openset_examples(model)

    # ---------- document
    r = Report(
        title="Adım 3 — Açık Küme Analizi ve Kırılımlar",
        subtitle=f"{MODEL_LABEL[model]} ile 'katalogda yok' kararı ve başarının çekim koşullarına göre değişimi",
        author="Ethem Belka Şahin",
        course="CMPE 9103 Computer Vision · Dönem Projesi İlerleme Raporu · İstanbul Üniversitesi",
    )
    r.h1("1. Bu rapor ne anlatıyor?")
    r.para(
        "Adım 2'de en iyi hazır modelin SigLIP olduğu bulundu. Bu adımda iki soru yanıtlanıyor. "
        "(1) Sistem, aradığı ürün katalogda yoksa bunu söyleyebilir mi? (Araştırma Sorusu 4) "
        "(2) Başarı; kapanma, ölçek, yakınlaştırma, bakış açısı ve kategoriye göre nasıl değişiyor? (Araştırma Sorusu 3) "
        "Her iki analiz de GPU gerektirmedi; Adım 2'de çıkarılan vektörler üzerinde yerel bilgisayarda yapıldı."
    )
    r.h1("2. Taahhüt ve kurulum")
    r.bullets([
        "Öneri: 'İlk sonucu yalnızca skoru bir eşiği (τ) aşıyorsa kabul et; mutlak benzerlik s1 ile margin s1 − s2'yi karşılaştır. τ yalnızca iç doğrulama ayrımında seçilir.'",
        "Ölçütler: AUROC (bilinen ↔ bilinmeyen), kapsama–kesinlik eğrisi, %95 kapsama veren τ'daki yanlış eşleşme oranı; Recall@1 kırılımları.",
        "'Katalogda yok' sorguları: style = 0 olan müşteri ürünleri (test: 3.137, val_internal: 1.339). Ek etiketleme yapılmadı.",
        f"τ, val_internal'da %95 kapsama için seçildi (mutlak: {ov['tau']['abs']:.3f}, margin: {ov['tau']['margin']:.4f}) ve test'e olduğu gibi uygulandı.",
    ])
    r.note(
        "Terimler: s1 = sorguya en benzer katalog öğesinin benzerlik skoru. s2 = farklı bir ürüne ait ilk öğenin skoru. "
        "Kapsama = eşiği geçip 'cevap verilen' sorguların oranı. Kesinlik = cevap verilenlerde ilk sonucun doğru olma oranı. "
        "AUROC = skorun iki grubu ayırma gücü; 0,5 yazı-tura, 1,0 kusursuz."
    )

    r.h1("3. Sonuç 1: Skorlar 'katalogda yok' demeyi öğrenmemiş")
    r.table(["Model", "AUROC bilinen↔yok (s1)", "AUROC bilinen↔yok (margin)", "AUROC doğru↔yanlış (s1)", "AUROC doğru↔yanlış (margin)"],
            [[MODEL_LABEL[k], v["auroc_abs"], v["auroc_margin"], v["auroc_correct_abs"], v["auroc_correct_margin"]] for k, v in all_m.items()],
            caption="Tablo 1. Dört modelde skorların ayırma gücü (test). İlk iki sütun: sorgu katalogda var mı? Son iki sütun: verilen ilk sonuç doğru mu?")
    r.para(
        f"Bilinen ile katalogda olmayan sorguları ayırmada her iki skor da yazı-tura düzeyinde (SigLIP: s1 {o['auroc_abs']:.2f}, margin {o['auroc_margin']:.2f}). "
        "Yani eğitimsiz bir model, 'bu ürün katalogda yok' diyemiyor. Nedeni Şekil 1'de görülüyor: katalogda olmayan bir ürün için de sistem "
        "neredeyse aynı skorla (ortalama s1 ≈ 0,83) benzer görünen bir ürün buluyor — model 'aynı ürün' ile 'çok benzer ürün' arasında fark görmüyor."
    )
    r.figure(f_hist_abs, "Şekil 1. Mutlak skor dağılımları. Üç grup üst üste biniyor; eşik koyarak ayırmak mümkün değil.", width_in=6.0)
    r.para(
        f"Buna karşılık ikinci soru — 'verdiğim ilk sonuç doğru mu?' — için margin anlamlı bir sinyal: AUROC {o['auroc_correct_margin']:.2f} "
        f"(mutlak skorla {o['auroc_correct_abs']:.2f}). İlk sonuç ile ikinci farklı ürün arasındaki fark büyükse sistem genellikle haklı."
    )
    r.figure(f_hist_margin, "Şekil 2. Margin dağılımları. Doğru bulunan sorgularda margin belirgin biçimde daha büyük; katalogda olmayanlar ise yanlış bulunanlarla aynı yerde.", width_in=6.0)

    r.h1("4. Sonuç 2: Kapsama–kesinlik dengesi")
    r.para("Eşiği yükselttikçe sistem daha az sorguya cevap verir (kapsama düşer) ama verdiği cevaplar daha doğru olur (kesinlik artar). Eğri bu dengeyi gösterir.")
    r.figure(f_cov, "Şekil 3. Kapsama–kesinlik. Margin eğrisi her kapsamada mutlak skorun üstünde; noktalar val_internal'da seçilen τ'nun test'teki konumu.", width_in=5.8)
    r.table(["Skor", "Kesinlik @%50 kapsama", "@%80", "@%95", "τ (val_internal)", "Test kapsaması", "Test kesinliği", "Yanlış eşleşme oranı"],
            [[n_tr, o[f"precision@cov50_{n}"], o[f"precision@cov80_{n}"], o[f"precision@cov95_{n}"], o["tau"][n], o[f"at_tau_{n}"]["coverage"], o[f"at_tau_{n}"]["precision"], o[f"at_tau_{n}"]["false_match_rate"]]
             for n, n_tr in (("abs", "mutlak s1"), ("margin", "margin s1−s2"))],
            caption="Tablo 2. SigLIP, test: farklı kapsamalarda kesinlik ve seçilen τ'nun sonucu")
    r.para(
        f"Yorum: Sorguların yarısına cevap vermekle yetinirsek margin ile kesinlik {o['precision@cov50_margin']*100:.0f}%'e çıkıyor "
        f"(genel R@1 {m['recall@1']*100:.0f}%). Ama katalogda olmayanların yine %{o['at_tau_margin']['false_match_rate']*100:.0f}'i eşiği geçiyor: "
        "eşik, yanlış cevapları azaltıyor; 'yok' demeyi sağlamıyor."
    )
    r.figure(f_fmr, "Şekil 4. Kapsama düştükçe yanlış eşleşme oranı da düşüyor ama hiçbir noktada bilinenlerden ayrışmıyor.", width_in=5.8)
    r.figure(f_ex, "Şekil 5. Katalogda olmayan sorgular için sistemin 'bulduğu' ilk 5. Hepsi yanlış ama görsel olarak çok benzer; 'benzer alternatif' senaryosu için yine de işe yarar.", width_in=6.3)
    r.note(
        f"'Benzer alternatif kalitesi': katalogda olmayan sorgularda ilk 5 sonucun %{m['openset_category_consistency@5']*100:.0f}'i sorguyla aynı kategoride "
        f"(bilinen sorgularda %{m['category_consistency@5']*100:.0f}). Aynısı yokken bile aynı türden ürün öneriliyor."
    )

    r.h1("5. Sonuç 3: Başarı nerede düşüyor?")
    r.figure(f_break, "Şekil 6. Recall@1, sorgu fotoğrafının çekim koşuluna göre. Ağır kapanma, küçük ölçek ve yandan/arkadan çekim en zor durumlar.", width_in=6.2)
    rows = []
    for a in ATTRIBUTES[:-1]:
        for rr in bt[bt.attribute == a].itertuples():
            rows.append([ATTR_TR[a], rr.label, rr.n, rr._5 if False else float(bt.loc[rr.Index, "recall@1"]), float(bt.loc[rr.Index, "recall@5"]), "✓" if rr.small_group else ""])
    r.table(["Koşul", "Değer", "n", "R@1", "R@5", "<30 örnek"], rows, caption="Tablo 3. Çekim koşullarına göre kırılım (test, SigLIP)")
    r.figure(f_cat, "Şekil 7. Recall@1 kategoriye göre. Baskılı/desenli üstler kolay; pantolon, etek ve askılı elbise zor (tek renk, az ayırt edici detay).", width_in=5.8)
    r.bullets([
        f"Kapanma: hafif → ağır arasında R@1 {bt[(bt.attribute=='occlusion')&(bt.value==1)]['recall@1'].iloc[0]:.2f} → {bt[(bt.attribute=='occlusion')&(bt.value==3)]['recall@1'].iloc[0]:.2f}; ürünün yarısı görünmüyorsa eşleştirme zorlaşıyor.",
        f"Ölçek: küçük ürünlerde {bt[(bt.attribute=='scale')&(bt.value==1)]['recall@1'].iloc[0]:.2f}, büyüklerde {bt[(bt.attribute=='scale')&(bt.value==3)]['recall@1'].iloc[0]:.2f}; küçük kutudan çıkan kırpıntı az piksel taşıyor.",
        f"Bakış açısı: giyilmemiş (düz serilmiş) ürünler en kolay ({bt[(bt.attribute=='viewpoint')&(bt.value==1)]['recall@1'].iloc[0]:.2f}), yandan/arkadan en zor ({bt[(bt.attribute=='viewpoint')&(bt.value==3)]['recall@1'].iloc[0]:.2f}); katalog fotoğrafları çoğunlukla önden.",
        "Yakınlaştırma: belirgin bir etkisi yok; orta yakınlaştırma en iyi.",
        f"Kategori: en kolay 'short sleeve top' ({cat.iloc[0]['recall@1']:.2f}), en zor 'trousers' ({cat[cat.label=='trousers']['recall@1'].iloc[0]:.2f}). 'short sleeve outwear' (n=32) küçük grup, sayısı güvenilir değil.",
        "Bu eğilimler DeepFashion2 makalesindeki Match R-CNN kırılımlarıyla aynı yönde (küçük ölçek, ağır kapanma, yandan/arkadan en zor).",
    ])

    r.h1("6. Ne öğrendik, sırada ne var?")
    r.bullets([
        "Eğitimsiz SigLIP ürünü iyi buluyor ama 'yok' diyemiyor. Bunun nedeni modelin görsel benzerlik öğrenmiş olması, aynı ürün kimliğini değil.",
        "Margin skoru, verilen cevabın güvenilirliği için kullanılabilir: düşük marginli cevaplar 'emin değilim' diye işaretlenebilir.",
        "Adım 4'te eğitilecek projeksiyon başlığının hedefi tam olarak bu: aynı ürünü birbirine yaklaştırıp farklı-ama-benzer ürünleri uzaklaştırmak. "
        "Başarılıysa hem R@1 hem de açık küme AUROC'u yükselmeli — ikisi de ölçülecek.",
        "Not: val_internal galerisi style=0 mağaza ürünlerini içermiyor, test galerisi (resmi liste) içeriyor; τ'nun test'te biraz farklı kapsama vermesinin nedeni bu. Rapora kısıt olarak yazılacak.",
    ])
    r.h1("7. Ortaya çıkan dosyalar")
    r.bullets([
        "outputs/results/<run_id>/{metrics.json, per_query.parquet, per_openset.parquet, coverage_curve_{abs,margin}.csv, breakdown.csv}",
        "reports/figures/03_openset/*.png — bu rapordaki şekiller",
        "src/fashret/evaluation/{open_set.py, breakdown.py}; tests/test_open_set.py (4 test)",
    ])
    out = r.save(ROOT / "docs/reports/03_acik_kume_ve_kirilimlar.docx")
    print("written", out)


if __name__ == "__main__":
    main()
