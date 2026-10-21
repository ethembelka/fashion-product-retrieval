"""Build docs/reports/04_projeksiyon_basligi.docx — what the projection head is, how InfoNCE / ArcFace train it,
and what it changed (closed set, open set, breakdowns). Written for a reader learning the subject.

Usage: python scripts/report_04_head.py   (after train_head.py + evaluate.py for variants head [and arcface], make_tables.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fashret.config import load_yaml  # noqa: E402
from fashret.evaluation.breakdown import ATTRIBUTES  # noqa: E402
from fashret.reporting import figures as fig  # noqa: E402
from fashret.reporting.docx_report import Report  # noqa: E402
from fashret.utils.io import read_json, read_table  # noqa: E402

FIG_DIR = ROOT / "reports/figures/04_head"
PROCESSED = ROOT / "data/processed"
ATTR_TR = {"occlusion": "Kapanma", "scale": "Ölçek", "zoom_in": "Yakınlaştırma", "viewpoint": "Bakış açısı", "category_id": "Kategori"}


def run_dir(model: str, variant: str, split: str) -> Path | None:
    r = sorted((ROOT / "outputs/results").glob(f"*_{model}_{variant}_{split}_s0"))
    return r[-1] if r else None


def fixed_examples(n: int = 3) -> Path:
    """Queries wrong at top-1 with zero-shot SigLIP and right with the head: query, zero-shot top-3, head top-3."""
    z = read_table(run_dir("siglip_vitb16", "zeroshot", "test") / "per_query.parquet")
    h = read_table(run_dir("siglip_vitb16+head", "head", "test") / "per_query.parquet")
    gm = read_table(ROOT / "data/features/siglip_vitb16/test/gallery_meta.parquet")
    gz, qz = np.load(ROOT / "data/features/siglip_vitb16/test/gallery_emb.npy"), np.load(ROOT / "data/features/siglip_vitb16/test/queries_emb.npy")
    gh, qh = np.load(ROOT / "data/features/siglip_vitb16+head/test/gallery_emb.npy"), np.load(ROOT / "data/features/siglip_vitb16+head/test/queries_emb.npy")
    cand = z.index[(z.first_hit_rank > 3) & (h.first_hit_rank == 1)].to_numpy()
    sel = np.random.default_rng(2).choice(cand, n, replace=False)
    rows, frame = [], []
    for i in sel:
        tz, th = np.argsort(-(gz @ qz[i]))[:3], np.argsort(-(gh @ qh[i]))[:3]
        cells = [(PROCESSED / z.crop_path[i], "sorgu")]
        cells += [(PROCESSED / gm.crop_path[j], f"eğitimsiz top-{k+1}") for k, j in enumerate(tz)]
        cells += [(PROCESSED / gm.crop_path[j], f"başlık top-{k+1}") for k, j in enumerate(th)]
        rows.append(cells)
        frame.append([fig.SERIES[0]] + ["#0ca30c" if gm.match_key[j] == z.match_key[i] else None for j in tz] + ["#0ca30c" if gm.match_key[j] == z.match_key[i] else None for j in th])
    return fig.image_grid(rows, FIG_DIR / "fixed_examples.png", "Eğitimsiz SigLIP'in kaçırdığı, başlığın bulduğu sorgular (yeşil = doğru ürün)", cell=1.35, frame=frame)


def main() -> None:
    cfg = load_yaml(ROOT / "configs/train/head.yaml")
    mz = read_json(run_dir("siglip_vitb16", "zeroshot", "test") / "metrics.json")
    mh = read_json(run_dir("siglip_vitb16+head", "head", "test") / "metrics.json")
    mzv = read_json(run_dir("siglip_vitb16", "zeroshot", "val_internal") / "metrics.json")
    mhv = read_json(run_dir("siglip_vitb16+head", "head", "val_internal") / "metrics.json")
    arc = run_dir("siglip_vitb16+arcface", "arcface", "test")
    ma = read_json(arc / "metrics.json") if arc else None
    tr = read_json(run_dir("siglip_vitb16", "head_train", "") / "train_summary.json") if run_dir("siglip_vitb16", "head_train", "") else None
    if tr is None:
        tr = read_json(sorted((ROOT / "outputs/results").glob("*_siglip_vitb16_head_train_s0"))[-1] / "train_summary.json")
    ta = None
    a_tr = sorted((ROOT / "outputs/results").glob("*_siglip_vitb16_arcface_train_s0"))
    if a_tr:
        ta = read_json(a_tr[-1] / "train_summary.json")
    hist = pd.DataFrame(tr["history"])
    bz = pd.read_csv(run_dir("siglip_vitb16", "zeroshot", "test") / "breakdown.csv")
    bh = pd.read_csv(run_dir("siglip_vitb16+head", "head", "test") / "breakdown.csv")
    pqh = read_table(run_dir("siglip_vitb16+head", "head", "test") / "per_query.parquet")
    poh = read_table(run_dir("siglip_vitb16+head", "head", "test") / "per_openset.parquet")
    cz = pd.read_csv(run_dir("siglip_vitb16", "zeroshot", "test") / "coverage_curve_margin.csv")
    ch = pd.read_csv(run_dir("siglip_vitb16+head", "head", "test") / "coverage_curve_margin.csv")
    cha = pd.read_csv(run_dir("siglip_vitb16+head", "head", "test") / "coverage_curve_abs.csv")
    hc, lc, sc, oc = cfg["head"], cfg["loss"], cfg["sampler"], cfg["optim"]

    # ---------- figures
    f_arch = fig.flow_diagram(["SigLIP vektörü\n768 sayı", "Doğrusal katman\n768 → 1024", "GELU\n(doğrusal olmayan)", "Doğrusal katman\n1024 → 256", "L2 normalize\n256 sayı"],
                              FIG_DIR / "head_arch.png", "Projeksiyon başlığı: SigLIP vektörünü yeni bir 256 boyutlu vektöre çeviren küçük ağ")
    f_curve = fig.lines({"val_internal Recall@1": (hist.epoch + 1, hist["val_recall@1"]), "val_internal Recall@5": (hist.epoch + 1, hist["val_recall@5"])},
                        FIG_DIR / "training_curve.png", "Eğitim boyunca iç doğrulama başarısı", "tur (epoch)", "Recall@K", ylim=(0.3, 1.0))
    f_loss = fig.lines({"InfoNCE kaybı (eğitim)": (hist.epoch + 1, hist["loss"])}, FIG_DIR / "training_loss.png", "Eğitim kaybı", "tur (epoch)", "kayıp")
    variants = {"Eğitimsiz SigLIP": mz, "SigLIP + başlık (InfoNCE)": mh}
    if ma: variants["SigLIP + başlık (ArcFace)"] = ma
    f_bar = fig.grouped_hbar(list(variants), {f"Recall@{k}": [v[f"recall@{k}"] for v in variants.values()] for k in (1, 5, 20)},
                             FIG_DIR / "before_after.png", "Recall@K — test", fmt="{:.3f}")
    f_cov = fig.lines({"eğitimsiz, margin": (cz.coverage, cz.precision), "başlık, margin": (ch.coverage, ch.precision), "başlık, mutlak": (cha.coverage, cha.precision)},
                      FIG_DIR / "coverage_before_after.png", "Kapsama–kesinlik (test)", "kapsama", "kesinlik", xlim=(0, 1))
    f_hist = fig.hist_overlay({"bilinen, ilk sonuç doğru": pqh.s1[pqh.top1_correct].to_numpy(), "bilinen, ilk sonuç yanlış": pqh.s1[~pqh.top1_correct].to_numpy(), "katalogda yok": poh.s1.to_numpy()},
                              FIG_DIR / "hist_abs_head.png", "Başlık sonrası mutlak skor s1 dağılımları", "s1")
    cond_z, cond_h = bz[bz.attribute != "category_id"], bh[bh.attribute != "category_id"]
    labels = [f"{ATTR_TR[r.attribute]} = {r.label}" for r in cond_z.itertuples()]
    f_break = fig.grouped_hbar(labels, {"eğitimsiz": list(cond_z["recall@1"]), "başlık": list(cond_h["recall@1"])}, FIG_DIR / "breakdown_before_after.png", "Recall@1 koşula göre — eğitimsiz vs başlık (test)", fmt="{:.3f}")
    f_fixed = fixed_examples()

    r = Report(
        title="Adım 4 — Projeksiyon Başlığı Eğitimi",
        subtitle="SigLIP vektörlerinin üstüne küçük bir öğrenilmiş katman: InfoNCE ile eğitim, ArcFace karşılaştırması, kazanç",
        author="Ethem Belka Şahin",
        course="CMPE 9103 Computer Vision · Dönem Projesi İlerleme Raporu · İstanbul Üniversitesi",
    )

    # ------------------------------------------------------------------ 1
    r.h1("1. Bu adımın sorusu ve gerekçesi")
    r.para(
        "Adım 2 ve 3'te görüldü: eğitimsiz SigLIP ürünü %36 oranında ilk sırada buluyor, ama 'benzer ama farklı' ürünleri aynı üründen ayıramıyor "
        "ve bu yüzden 'katalogda yok' diyemiyor. SigLIP'in vektörleri 'bu görüntüde ne var' sorusuna göre düzenlenmiş; biz 'bu hangi ürün' sorusuna göre "
        "düzenlenmiş vektör istiyoruz. Bu adımda SigLIP dondurulmuş bırakılır, üstüne küçük bir ağ (projeksiyon başlığı) eklenir ve yalnızca bu ağ, "
        "DeepFashion2'nin eğitim verisiyle 'aynı ürünü yaklaştır, farklı ürünü uzaklaştır' hedefiyle eğitilir. Soru: bu küçük ek ne kazandırır? (Araştırma Sorusu 2) "
        "Önerideki söz: 'en iyi omurga üstünde InfoNCE ile P×K batch'lerle eğitilmiş hafif bir projeksiyon başlığı; zaman kalırsa ArcFace karşılaştırması.'"
    )

    # ------------------------------------------------------------------ 2
    r.h1("2. Projeksiyon başlığı nedir?")
    r.para(
        f"İki katmanlı bir ağ (MLP). Girdi: SigLIP'in 768 sayılık vektörü. Birinci katman 768×{hc['hidden_dim']} bir ağırlık matrisiyle çarpıp {hc['hidden_dim']} sayı üretir; "
        f"GELU fonksiyonu her sayıyı doğrusal olmayan biçimde dönüştürür (negatifleri sıfıra yakın bastırır); ikinci katman {hc['hidden_dim']}×{hc['out_dim']} matrisle {hc['out_dim']} sayıya indirir; "
        f"sonuç L2-normalize edilir. Toplam ≈ {(768*hc['hidden_dim'] + hc['hidden_dim']*hc['out_dim'])/1e6:.1f} milyon ayarlanabilir sayı (SigLIP'in 86 milyonuna karşı). "
        f"Eğitimde %{int(hc['dropout']*100)} dropout: her adımda ara sayıların rastgele %{int(hc['dropout']*100)}'u sıfırlanır, ağ tek tek sayılara bağımlı kalmasın diye."
    )
    r.figure(f_arch, "Şekil 1. Başlığın yapısı. SigLIP hiç değişmez; yalnızca bu iki katmanın ağırlıkları öğrenilir.", width_in=6.3)
    r.para(
        "Neden SigLIP'i de eğitmiyoruz? Üç neden: (1) SigLIP'in vektörleri Adım 2'de bir kez çıkarıldı; başlık bu hazır vektörler üzerinde çalıştığı için eğitim "
        "GPU'da 3 dakika sürüyor, görüntü okumak gerekmiyor. (2) 86 milyon ağırlığı 16 bin ürünle eğitmek ezberleme riski taşır. (3) Önce ucuz adımın ne kazandırdığı "
        "ölçülür; kısmi fine-tune planın sonraki maddesidir."
    )

    # ------------------------------------------------------------------ 3
    r.h1("3. Eğitim nasıl yapıldı?")
    r.h2("3.1 Veri")
    r.para(
        f"train_fit ayrımı (Adım 1): {tr['n_train_rows']:,} vektör — 60.894 müşteri + 133.709 mağaza — {tr['n_products']:,} farklı ürün (match_key). "
        "Yalnızca hem müşteri hem mağaza fotoğrafı olan ürünler kullanılır (bir müşteri fotoğrafının 'doğru cevabı' olmalı). "
        "val_internal (6.236 sorgu, 14.533 galeri) her turun sonunda başarıyı ölçmek için; test hiç görülmez.".replace(",", ".")
    )
    r.h2("3.2 Batch'ler: P×K örnekleme")
    r.para(
        f"Her adımda rastgele P = {sc['P']} ürün seçilir; her üründen K = {sc['K']} görüntü: 1 müşteri + 1 mağaza. Batch = {sc['P']*sc['K']} vektör. "
        "Neden böyle? Kayıp fonksiyonu 'aynı ürün' çiftlerine ihtiyaç duyar; rastgele 128 görüntü alınsa aynı üründen iki tane düşme ihtimali çok düşük olurdu. "
        f"Bir tur (epoch) = her ürün bir kez seçilene kadar ≈ {tr['n_products']//sc['P']} batch."
    )
    r.h2("3.3 Kayıp fonksiyonu: InfoNCE (müşteri → mağaza)")
    r.para(
        f"Batch'teki {sc['P']} müşteri vektörü ile {sc['P']} mağaza vektörünün bütün ikili kosinüs benzerlikleri hesaplanır: {sc['P']}×{sc['P']} bir tablo. "
        f"Her müşteri satırı için doğru hücre, kendi ürününün mağaza fotoğrafı; diğer {sc['P']-1} hücre yanlış. Benzerlikler sıcaklık T = {lc['temperature']} ile bölünür "
        "(farkları büyütür: 0,80 ile 0,85 arasındaki fark 16 ile 17 olur), satıra softmax uygulanır (hücreler 0–1 arasına, toplamı 1'e getirilir) ve doğru hücrenin "
        "olasılığının −log'u alınır. Kayıp = satırların ortalaması. Kayıp 0'a yaklaşır ancak her müşteri fotoğrafı için kendi mağaza fotoğrafı batch'teki diğer 63 üründen "
        "açık ara daha benzerse. Ağırlıklar bu kaybı küçültecek yönde güncellenir (AdamW, öğrenme hızı "
        f"{oc['lr']}). Adım 2'de CLIP'in görüntü–metin için yaptığı şeyin aynısı, burada müşteri–mağaza fotoğrafı için."
    )
    r.h2("3.4 Tur sonu ölçüm ve erken durdurma")
    r.para(
        f"Her tur sonunda başlık val_internal'ın galeri ve sorgu vektörlerine uygulanır, Recall@1 hesaplanır. En iyi tur saklanır (best.pt). "
        f"{oc['patience']} tur boyunca iyileşme olmazsa durulur. Bu çalıştırmada {tr['epochs_run']} tur tamamlandı, en iyi val R@1 = {tr['best_val_recall@1']:.3f} "
        f"(tur {int(hist['val_recall@1'].idxmax())+1}); toplam süre {hist.seconds.sum():.0f} saniye (RTX 3060)."
    )
    r.figure(f_curve, "Şekil 2. Tur tur iç doğrulama başarısı. İlk turda 0,42'den 0,58'e sıçrıyor, sonra yavaş yavaş 0,66'ya çıkıyor.", width_in=5.6)
    r.figure(f_loss, "Şekil 3. Eğitim kaybı düzenli düşüyor; aşırı öğrenme belirtisi (kayıp düşerken doğrulama başarısının düşmesi) yok.", width_in=5.0)

    # ------------------------------------------------------------------ 4
    r.h1("4. Sonuçlar")
    r.h2("4.1 Kapalı küme — Tablo 2")
    rows = [[name, v["recall@1"], v["recall@5"], v["recall@10"], v["recall@20"], v["map"], v["category_consistency@5"], v["efficiency"]["embed_dim"]] for name, v in variants.items()]
    r.table(["Yöntem", "R@1", "R@5", "R@10", "R@20", "mAP", "Kat. tut.@5", "Boyut"], rows, caption="Tablo 1 (= makaledeki Tablo 2). Test: 12.375 sorgu, 36.958 galeri")
    r.figure(f_bar, "Şekil 4. Eğitimsiz SigLIP ile başlık eklenmiş SigLIP.", width_in=6.0)
    r.bullets([
        f"Recall@1 {mz['recall@1']:.3f} → {mh['recall@1']:.3f} (+{(mh['recall@1']-mz['recall@1'])*100:.0f} puan), mAP {mz['map']:.3f} → {mh['map']:.3f}. "
        f"Her 100 sorgudan {mh['recall@1']*100:.0f}'inde ilk sonuç doğru; {mh['recall@5']*100:.0f}'inde doğru ürün ilk 5'te.",
        f"val_internal'da aynı yönde: {mzv['recall@1']:.3f} → {mhv['recall@1']:.3f}.",
        f"Vektör boyutu 768'den {hc['out_dim']}'ya düştü: katalog indeksi 3 kat küçük, arama 3 kat hızlı.",
        f"Makaledeki Match R-CNN (top-1 0,268, top-10 0,574, top-20 0,665) ile karşılaştırma: {mh['recall@1']:.3f} / {mh['recall@10']:.3f} / {mh['recall@20']:.3f}.",
    ])
    r.figure(f_fixed, "Şekil 5. Eğitimsiz modelin ilk 3'e koyamadığı, başlığın ilk sırada bulduğu sorgular. Eğitimsiz model benzer türden ürünler getiriyor; başlık aynı ürünü seçiyor.", width_in=6.4)

    r.h2("4.2 Açık küme")
    oz, oh = mz["openset"], mh["openset"]
    r.table(["Ölçü", "Eğitimsiz", "Başlık"],
            [["AUROC var↔yok, mutlak s1", oz["auroc_abs"], oh["auroc_abs"]], ["AUROC var↔yok, margin", oz["auroc_margin"], oh["auroc_margin"]],
             ["AUROC doğru↔yanlış, mutlak s1", oz["auroc_correct_abs"], oh["auroc_correct_abs"]], ["AUROC doğru↔yanlış, margin", oz["auroc_correct_margin"], oh["auroc_correct_margin"]],
             ["Kesinlik @%50 kapsama (en iyi skor)", max(oz["precision@cov50_abs"], oz["precision@cov50_margin"]), max(oh["precision@cov50_abs"], oh["precision@cov50_margin"])],
             ["Yanlış eşleşme oranı @ τ (margin)", oz["at_tau_margin"]["false_match_rate"], oh["at_tau_margin"]["false_match_rate"]],
             ["Yanlış eşleşme oranı @ τ (mutlak)", oz["at_tau_abs"]["false_match_rate"], oh["at_tau_abs"]["false_match_rate"]]],
            caption="Tablo 2. Açık küme ölçüleri, test (τ her iki durumda val_internal'da %95 kapsama için seçildi)")
    r.para(
        f"En önemli değişiklik: mutlak skor artık bilgi taşıyor. 'Katalogda var mı' AUROC'u {oz['auroc_abs']:.2f}'den {oh['auroc_abs']:.2f}'ye çıktı; "
        f"katalogda olmayanların eşiği geçme oranı %{oz['at_tau_abs']['false_match_rate']*100:.0f}'dan %{oh['at_tau_abs']['false_match_rate']*100:.0f}'a indi. "
        "Eğitim 'aynı ürün = yüksek benzerlik, farklı ürün = düşük benzerlik' ilişkisini kurunca, katalogda olmayan ürün için en yüksek benzerlik de düşük kalıyor. "
        "Eğitimsiz modelde tam tersi geçerliydi (margin daha iyiydi). Yine de 0,68 kusursuzdan uzak: 'yok' kararı kısmen mümkün, güvenilir değil."
    )
    r.figure(f_hist, "Şekil 6. Başlık sonrası mutlak skor dağılımları (Adım 3'teki Şekil 1 ile karşılaştırın): 'katalogda yok' grubu artık sola kaymış, 'doğru' grubu sağa.", width_in=6.0)
    r.figure(f_cov, "Şekil 7. Kapsama–kesinlik. Başlıkla eğri bütünüyle yukarı taşındı; %50 kapsamada kesinlik "
             f"{max(oh['precision@cov50_abs'], oh['precision@cov50_margin']):.2f}.", width_in=5.8)

    r.h2("4.3 Çekim koşullarına göre")
    r.figure(f_break, "Şekil 8. Her koşulda Recall@1, eğitimsiz ve başlık. Kazanç bütün koşullarda var; zor koşullar (ağır kapanma, küçük ölçek, yandan/arkadan) hâlâ en düşük.", width_in=6.3)
    rows = []
    for a in ATTRIBUTES[:-1]:
        for z_, h_ in zip(bz[bz.attribute == a].to_dict("records"), bh[bh.attribute == a].to_dict("records")):
            rows.append([ATTR_TR[a], z_["label"], int(z_["n"]), z_["recall@1"], h_["recall@1"], round(h_["recall@1"] - z_["recall@1"], 3)])
    r.table(["Koşul", "Değer", "n", "Eğitimsiz R@1", "Başlık R@1", "Fark"], rows, caption="Tablo 3. Kırılım, test")

    # ------------------------------------------------------------------ 5
    r.h1("5. ArcFace karşılaştırması")
    r.para(
        "ArcFace (Deng vd., 2019) yüz tanımada yaygın bir kayıptır. InfoNCE'den farkı: her ürün için öğrenilen bir 'merkez' vektör vardır (16.374 ürün × 256 sayı); "
        "her görüntünün vektörü kendi ürününün merkezine yaklaştırılır, diğer merkezlerden uzaklaştırılır. Ek olarak doğru merkezle açıya bir marj (m) eklenir: "
        "görüntü yalnızca yakın değil, marj kadar fazladan yakın olmak zorundadır. Hesap: cos(θ + m) doğru sınıf için, cos(θ) diğerleri için; ölçek s ile çarpılıp softmax + çapraz entropi."
    )
    if ma and ta:
        mav = read_json(run_dir("siglip_vitb16+arcface", "arcface", "val_internal") / "metrics.json")
        oa = ma["openset"]
        r.para(
            "İlk deneme başarısız oldu: sınıf merkezleri başlıkla aynı öğrenme hızında (0,001) ve m = 0,3 ile eğitilince val R@1 0,08'de kaldı. Neden: 16.374 ürünün her biri bir turda "
            "yalnızca 2 görüntü görür; merkez vektörler bu hızla neredeyse hiç hareket etmiyor. ArcFace, sınıf başına yüzlerce örneğin olduğu yüz veri setleri için tasarlanmıştır. "
            "Merkezlere 10 kat öğrenme hızı (0,01) ve m = 0,2 verilince eğitim çalıştı:"
        )
        r.table(["Ölçü", "InfoNCE", "ArcFace"],
                [["val_internal R@1 (seçim ölçüsü)", mhv["recall@1"], mav["recall@1"]], ["test R@1", mh["recall@1"], ma["recall@1"]], ["test R@5", mh["recall@5"], ma["recall@5"]],
                 ["test R@20", mh["recall@20"], ma["recall@20"]], ["test mAP", mh["map"], ma["map"]], ["test AUROC var↔yok (s1)", oh["auroc_abs"], oa["auroc_abs"]],
                 ["test kat. tut.@5", mh["category_consistency@5"], ma["category_consistency@5"]], ["tur sayısı / süre", f"{tr['epochs_run']} / {hist.seconds.sum():.0f} s", f"{ta['epochs_run']} / {sum(h['seconds'] for h in ta['history']):.0f} s"]],
                caption="Tablo 4. İki kayıp fonksiyonu, aynı başlık, aynı veri, aynı tur sayısı")
        r.para(
            f"İki kayıp test'te başa baş: ArcFace R@1'de {ma['recall@1']-mh['recall@1']:+.3f}, mAP'de {ma['map']-mh['map']:+.3f}, R@20'de {ma['recall@20']-mh['recall@20']:+.3f}. "
            f"Mutlak skorla açık küme ayrımında ArcFace biraz daha iyi ({oa['auroc_abs']:.2f} / {oh['auroc_abs']:.2f}): her ürün için bir merkez öğrenmek, 'bu ürün değil' demeyi biraz kolaylaştırıyor. "
            f"Yöntem seçimi kurala göre val_internal'da yapılır; orada InfoNCE önde ({mhv['recall@1']:.3f} / {mav['recall@1']:.3f}), bu yüzden ana yöntem InfoNCE kalır. "
            "Test'teki küçük farklar tek tohumla ayırt edilemez; nihai raporda tohum tekrarıyla bakılacak. ArcFace ayrıca 16.374×256'lık merkez matrisini de öğrenir (4,2 milyon ek sayı) ve turu 2–3 kat yavaştır."
        )
    else:
        r.para("ArcFace çalıştırması henüz tamamlanmadı; bu bölüm sonuç geldiğinde doldurulacak.")

    # ------------------------------------------------------------------ 6
    r.h1("6. Dikkat edilenler")
    r.bullets([
        "Test hiçbir karara girmedi: tur seçimi ve τ val_internal'da; test bir kez, en sonda ölçüldü.",
        "pair_id düzeyinde ayrım sayesinde test ürünlerinin hiçbiri eğitimde yok (Adım 1).",
        "Tohum sabit (0); aynı komut aynı sonucu verir. Tohum tekrarı (ortalama ± sapma) nihai raporda yapılacak.",
        "Başlık hâlâ öğreniyor: 30. turda val R@1 düşmeye başlamamıştı. Daha uzun eğitim ve öğrenme hızı taraması küçük ek kazanç verebilir; planda sonraya bırakıldı.",
    ])
    r.h1("7. Sıradaki adım")
    r.bullets([
        "Plan madde 4: kısmi fine-tune (SigLIP'in son 2–4 bloğu açık). Bu kez görüntülerden geçmek gerekir: Colab L4'te tur başına ≈ 8 dk; 5–10 tur ≈ 1 saat. Süre tahmini onaya sunulacak.",
        "Nihai rapor için: tohum tekrarı, Tablo 1–2'nin LaTeX'i, şekillerin son hâli.",
    ])
    r.h1("8. Dosyalar")
    r.bullets([
        "outputs/checkpoints/<run_id>/{best.pt, last.pt, history.json}; outputs/results/<run_id>/train_summary.json",
        "data/features/siglip_vitb16+head/{val_internal,test}/ (başlık uygulanmış vektörler); outputs/results/*_head_*/ (değerlendirme)",
        "src/fashret/models/projection_head.py, data/samplers.py, engine/losses.py, engine/trainer.py; scripts/train_head.py; tests/test_training.py",
        "configs/train/head.yaml, head_arcface.yaml",
    ])
    out = r.save(ROOT / "docs/reports/04_projeksiyon_basligi.docx")
    print("written", out)


if __name__ == "__main__":
    main()
