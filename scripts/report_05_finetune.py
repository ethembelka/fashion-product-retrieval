"""Build docs/reports/05_kismi_fine_tune.docx — what fine-tuning the backbone means, how it was done on Colab, and the
three-way comparison zero-shot -> head -> fine-tune (closed set, open set, breakdowns).

Usage: python scripts/report_05_finetune.py   (after evaluate.py --variant ft and make_tables.py;
       outputs/checkpoints/ft/history.json optional — adds the training curve if present)
"""

from __future__ import annotations

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

FIG_DIR = ROOT / "reports/figures/05_finetune"
PROCESSED = ROOT / "data/processed"
ATTR_TR = {"occlusion": "Kapanma", "scale": "Ölçek", "zoom_in": "Yakınlaştırma", "viewpoint": "Bakış açısı", "category_id": "Kategori"}
MATCH_RCNN = {"top1": 0.268, "top10": 0.574, "top20": 0.665}
VARIANTS = [("siglip_vitb16", "zeroshot", "Eğitimsiz SigLIP"), ("siglip_vitb16+head", "head", "+ başlık (Adım 4)"), ("siglip_vitb16+ft", "ft", "+ son 4 blok fine-tune (Adım 5)")]


def run_dir(model: str, variant: str, split: str) -> Path:
    return sorted((ROOT / "outputs/results").glob(f"*_{model}_{variant}_{split}_s0"))[-1]


def fixed_examples(n: int = 3) -> Path:
    """Queries wrong at top-1 with the head and right with the fine-tuned model: query, head top-3, ft top-3."""
    h = read_table(run_dir("siglip_vitb16+head", "head", "test") / "per_query.parquet")
    f = read_table(run_dir("siglip_vitb16+ft", "ft", "test") / "per_query.parquet")
    gm = read_table(ROOT / "data/features/siglip_vitb16/test/gallery_meta.parquet")
    gh, qh = np.load(ROOT / "data/features/siglip_vitb16+head/test/gallery_emb.npy"), np.load(ROOT / "data/features/siglip_vitb16+head/test/queries_emb.npy")
    gf, qf = np.load(ROOT / "data/features/siglip_vitb16+ft/test/gallery_emb.npy"), np.load(ROOT / "data/features/siglip_vitb16+ft/test/queries_emb.npy")
    cand = h.index[(h.first_hit_rank > 3) & (f.first_hit_rank == 1)].to_numpy()
    sel = np.random.default_rng(4).choice(cand, n, replace=False)
    rows, frame = [], []
    for i in sel:
        th, tf = np.argsort(-(gh @ qh[i]))[:3], np.argsort(-(gf @ qf[i]))[:3]
        rows.append([(PROCESSED / h.crop_path[i], "sorgu")] + [(PROCESSED / gm.crop_path[j], f"başlık top-{k+1}") for k, j in enumerate(th)] + [(PROCESSED / gm.crop_path[j], f"fine-tune top-{k+1}") for k, j in enumerate(tf)])
        frame.append([fig.SERIES[0]] + ["#0ca30c" if gm.match_key[j] == h.match_key[i] else None for j in list(th) + list(tf)])
    return fig.image_grid(rows, FIG_DIR / "fixed_examples.png", "Başlığın kaçırdığı, fine-tune'un bulduğu sorgular (yeşil = doğru ürün)", cell=1.35, frame=frame)


def main() -> None:
    cfg = load_yaml(ROOT / "configs/train/finetune.yaml")
    M = {v: read_json(run_dir(m, v, "test") / "metrics.json") for m, v, _ in VARIANTS}
    MV = {v: read_json(run_dir(m, v, "val_internal") / "metrics.json") for m, v, _ in VARIANTS}
    B = {v: pd.read_csv(run_dir(m, v, "test") / "breakdown.csv") for m, v, _ in VARIANTS}
    C = {v: pd.read_csv(run_dir(m, v, "test") / "coverage_curve_abs.csv") for m, v, _ in VARIANTS}
    labels = {v: lab for _, v, lab in VARIANTS}
    hist_path = ROOT / "outputs/checkpoints/ft/history.json"
    hist = pd.DataFrame(read_json(hist_path)) if hist_path.exists() else None
    pqf = read_table(run_dir("siglip_vitb16+ft", "ft", "test") / "per_query.parquet")
    pof = read_table(run_dir("siglip_vitb16+ft", "ft", "test") / "per_openset.parquet")
    oc, sc = cfg["optim"], cfg["sampler"]

    f_arch = fig.flow_diagram(["Kırpıntı\n224×224", "Blok 1–8\n(dondurulmuş)", "Blok 9–12 + havuzlama\n(EĞİTİLİR, 34 M)", "Başlık\n(EĞİTİLİR, 1 M)", "256 boyutlu\nvektör"],
                              FIG_DIR / "ft_arch.png", "Kısmi fine-tune: ağın hangi parçaları değişiyor")
    f_bar = fig.grouped_hbar([labels[v] for v in M], {f"Recall@{k}": [M[v][f"recall@{k}"] for v in M] for k in (1, 5, 20)}, FIG_DIR / "progression.png", "Recall@K — üç aşama (test)", fmt="{:.3f}")
    f_map = fig.hbar([labels[v] for v in M], [M[v]["map"] for v in M], FIG_DIR / "map_progression.png", "mAP — üç aşama (test)", "mAP", fmt="{:.3f}")
    f_cov = fig.lines({labels[v]: (C[v].coverage, C[v].precision) for v in C}, FIG_DIR / "coverage_abs_3way.png", "Kapsama–kesinlik, mutlak skor (test)", "kapsama", "kesinlik", xlim=(0, 1))
    f_hist = fig.hist_overlay({"bilinen, ilk sonuç doğru": pqf.s1[pqf.top1_correct].to_numpy(), "bilinen, ilk sonuç yanlış": pqf.s1[~pqf.top1_correct].to_numpy(), "katalogda yok": pof.s1.to_numpy()},
                              FIG_DIR / "hist_abs_ft.png", "Fine-tune sonrası mutlak skor s1 dağılımları", "s1")
    cond = {v: B[v][B[v].attribute != "category_id"] for v in B}
    f_break = fig.grouped_hbar([f"{ATTR_TR[r.attribute]} = {r.label}" for r in cond["zeroshot"].itertuples()], {labels[v]: list(cond[v]["recall@1"]) for v in cond},
                               FIG_DIR / "breakdown_3way.png", "Recall@1 koşula göre — üç aşama (test)", fmt="{:.3f}")
    f_fixed = fixed_examples()
    f_curve = None
    if hist is not None:
        f_curve = fig.lines({"val_internal Recall@1": (hist.epoch + 1, hist["val_recall@1"]), "val_internal Recall@5": (hist.epoch + 1, hist["val_recall@5"])},
                            FIG_DIR / "training_curve.png", "Fine-tune boyunca iç doğrulama başarısı", "tur (epoch)", "Recall@K", ylim=(0.5, 1.0))

    r = Report(
        title="Adım 5 — Kısmi Fine-tune",
        subtitle="SigLIP'in son bloklarını DeepFashion2 ile eğitmek: ne demek, nasıl yapıldı, ne kazandırdı",
        author="Ethem Belka Şahin",
        course="CMPE 9103 Computer Vision · Dönem Projesi İlerleme Raporu · İstanbul Üniversitesi",
    )

    r.h1("1. Bu adımın sorusu")
    r.para(
        "Adım 4'te SigLIP dondurulmuş kaldı; yalnızca üstüne eklenen küçük başlık eğitildi ve R@1 0,36'dan 0,56'ya çıktı. Soru: SigLIP'in kendi ağırlıklarının bir kısmı da "
        "eğitilirse ne olur? Bu, planın en ağır maddesiydi (görüntülerden geçmek gerekir, GPU şart) ve bu yüzden sona bırakılmıştı. Önerideki ifade: 'zaman kalırsa omurganın son bloklarının kısmi fine-tune'u.'"
    )

    r.h1("2. Fine-tune ne demek, neden 'kısmi'?")
    r.para(
        "Adım 2 raporunda anlatıldığı gibi SigLIP 12 Transformer bloğundan oluşur; her blokta öz-dikkat ve MLP ağırlıkları vardır (toplam 93 milyon sayı). "
        "'Dondurmak' = bu sayıları hiç değiştirmemek. 'Fine-tune' = kayıp fonksiyonunun türevini bu sayılara kadar geri yayıp (backpropagation) onları da güncellemek; "
        "ağ, bizim verimizde 'aynı ürün / farklı ürün' ayrımına yarayan özellikleri kendi içinde üretmeye başlar. 'Kısmi' = yalnızca son 4 blok, dikkat havuzlama katmanı ve son normalizasyon "
        f"açık ({cfg['unfreeze_blocks']} blok ≈ 34 milyon sayı); ilk 8 blok ve parça gömme dondurulmuş."
    )
    r.figure(f_arch, "Şekil 1. Hangi parçalar eğitiliyor. Türev yalnızca eğitilen parçalara kadar geri gider; ilk 8 blok ileri hesaba katılır ama değişmez.", width_in=6.3)
    r.para("Neden hepsi değil? Üç neden:")
    r.bullets([
        "Veri az: 16.374 ürün, ürün başına birkaç fotoğraf. 93 milyon sayının tamamı bu kadar veriyle eğitilirse ağ eğitim setini ezberler, test'te düşer.",
        "Alt bloklar genel görsel özellikleri (kenar, doku, renk geçişi) hesaplar; bunlar her görev için aynıdır. Üst bloklar 'bu nesne nedir' düzeyindeki, göreve özgü bilgiyi taşır; değişmesi gereken yer orası.",
        "Maliyet: türev yalnızca son 4 bloktan geçince bellek ve süre yaklaşık üçte bire iner.",
    ])

    r.h1("3. Nasıl yapıldı?")
    r.h2("3.1 Veri ve örnekleme")
    r.para(
        "Adım 4'teki gibi train_fit, P×K örnekleme (P = 64 ürün, K = 2: bir müşteri + bir mağaza fotoğrafı → 128 görüntü/adım). Fark: artık vektörler hazır değil; "
        "her adımda 128 kırpıntı diskten okunur, 224×224'e getirilir ve ağın tamamından ileri geçer. Eğitimde iki veri çoğaltma işlemi: rastgele bir alt-dikdörtgen "
        f"(alanın %{int(cfg['augment']['random_resized_crop_scale'][0]*100)}–100'ü) kesilip 224×224'e getirilir ve %50 olasılıkla yatay çevrilir. Amaç: ağ aynı fotoğrafı her turda biraz farklı görsün, ezberlemesin. "
        "Doğrulama ve test görüntüleri çoğaltılmaz."
    )
    r.h2("3.2 Kayıp, öğrenme hızları, hassasiyet")
    r.bullets([
        f"Kayıp: InfoNCE müşteri → mağaza (Adım 4 raporu, 3.3), sıcaklık {cfg['loss']['temperature']}. Başlık Adım 4'te eğitilen ağırlıklardan başlatıldı; rastgele başlasaydı ilk turlarda omurgaya anlamsız türev gönderirdi.",
        f"İki öğrenme hızı: omurga blokları için {oc['lr_backbone']} (küçük: hazır ağırlıkları yavaşça kaydır), başlık için {oc['lr_head']}. İlk {oc['warmup_steps']} adımda hızlar sıfırdan doğrusal yükselir (ısınma), sonra kosinüs eğrisiyle sıfıra iner.",
        f"AdamW, ağırlık sönümü {oc['weight_decay']}; türev normu 1'e kırpılır (tek bir kötü batch'in ağırlıkları fırlatmasını önler).",
        "Karışık hassasiyet (AMP): ileri/geri hesaplar 16-bit, ağırlık güncellemeleri 32-bit. L4'te iki kat hız, yarı bellek.",
        f"Her tur sonunda val_internal'ın 20.769 kırpıntısı güncel ağdan geçirilip Recall@1 ölçülür; en iyi tur saklanır; {oc['patience']} tur iyileşme yoksa durulur; en fazla {oc['epochs']} tur.",
    ])
    r.h2("3.3 Çalıştırma")
    r.para(
        "Google Colab, L4 GPU. Defter tek seferde baştan sona çalıştırıldı: Drive bağlama, kodu GitHub'dan çekme, crops.zip'i açma, eğitim, en iyi modelle val_internal ve test "
        "özniteliklerini Drive'a yazma, GPU'yu bırakma. Her tur sonunda model Drive'a kaydedildiği için oturum düşse de kaldığı yerden sürer. "
        "Eğitilmiş vektörler (256 boyut) yerel bilgisayara indirildi ve Adım 2–3'teki aynı değerlendirme koduyla ölçüldü."
    )
    if hist is not None:
        r.para(f"{len(hist)} tur tamamlandı; en iyi val R@1 {hist['val_recall@1'].max():.3f} (tur {int(hist['val_recall@1'].idxmax())+1}); tur başına ≈ {hist.seconds.median()/60:.1f} dk.")
        r.figure(f_curve, "Şekil 2. Tur tur iç doğrulama başarısı.", width_in=5.6)
    else:
        r.note("Eğitim eğrisi (history.json) henüz Drive'dan indirilmedi; indirilince bu rapora tur tur başarı grafiği eklenecek.")

    r.h1("4. Sonuçlar")
    r.h2("4.1 Kapalı küme")
    rows = [[labels[v], M[v]["recall@1"], M[v]["recall@5"], M[v]["recall@10"], M[v]["recall@20"], M[v]["map"], M[v]["category_consistency@5"], MV[v]["recall@1"]] for v in M]
    r.table(["Aşama", "R@1", "R@5", "R@10", "R@20", "mAP", "Kat. tut.@5", "val R@1"], rows, caption="Tablo 1. Üç aşama, test (son sütun val_internal)")
    r.figure(f_bar, "Şekil 3. Her aşama bir öncekinin üstüne ekliyor.", width_in=6.0)
    r.figure(f_map, "Şekil 4. mAP: yalnızca ilk doğru değil bütün doğru fotoğrafların sırası iyileşiyor.", width_in=5.2)
    z, h, f = M["zeroshot"], M["head"], M["ft"]
    r.bullets([
        f"R@1: {z['recall@1']:.3f} → {h['recall@1']:.3f} → {f['recall@1']:.3f}. Fine-tune, başlığın üstüne +{(f['recall@1']-h['recall@1'])*100:.1f} puan; eğitimsiz modele göre toplam +{(f['recall@1']-z['recall@1'])*100:.0f} puan.",
        f"mAP {z['map']:.3f} → {h['map']:.3f} → {f['map']:.3f}; R@20 {f['recall@20']:.3f}: beş sorgudan dördünde doğru ürün ilk 20'de.",
        f"Makaledeki Match R-CNN (gerçek kutu): top-1 {MATCH_RCNN['top1']}, top-10 {MATCH_RCNN['top10']}, top-20 {MATCH_RCNN['top20']}. Bizim: {f['recall@1']:.3f} / {f['recall@10']:.3f} / {f['recall@20']:.3f} — top-1'de 2,4 kat.",
        f"Kategori tutarlılığı@5 {z['category_consistency@5']:.2f} → {f['category_consistency@5']:.2f}: 'benzer alternatif' önerileri de daha tutarlı.",
    ])
    r.figure(f_fixed, "Şekil 5. Başlığın ilk 3'e koyamadığı, fine-tune'un ilk sırada bulduğu sorgular.", width_in=6.4)

    r.h2("4.2 Açık küme")
    oz, oh, of = z["openset"], h["openset"], f["openset"]
    r.table(["Ölçü", labels["zeroshot"], labels["head"], labels["ft"]],
            [["AUROC var↔yok, mutlak s1", oz["auroc_abs"], oh["auroc_abs"], of["auroc_abs"]], ["AUROC var↔yok, margin", oz["auroc_margin"], oh["auroc_margin"], of["auroc_margin"]],
             ["AUROC doğru↔yanlış, margin", oz["auroc_correct_margin"], oh["auroc_correct_margin"], of["auroc_correct_margin"]],
             ["Kesinlik @%50 kapsama (mutlak)", oz["precision@cov50_abs"], oh["precision@cov50_abs"], of["precision@cov50_abs"]],
             ["Kesinlik @%50 kapsama (margin)", oz["precision@cov50_margin"], oh["precision@cov50_margin"], of["precision@cov50_margin"]],
             ["Yanlış eşleşme @ τ (mutlak)", oz["at_tau_abs"]["false_match_rate"], oh["at_tau_abs"]["false_match_rate"], of["at_tau_abs"]["false_match_rate"]]],
            caption="Tablo 2. Açık küme, test (τ val_internal'da %95 kapsama)")
    r.para(
        f"'Katalogda var mı' ayrımı {oz['auroc_abs']:.2f} → {oh['auroc_abs']:.2f} → {of['auroc_abs']:.2f}. Yön doğru ama adım küçük: ağ ürün kimliğini öğrendikçe katalogda olmayan ürüne "
        "verdiği en yüksek benzerlik düşüyor, fakat 'çok benzer başka ürün' hâlâ yüksek skor alıyor. Pratik kural değişmedi: skor, cevabın güvenilirliğini söyler "
        f"(%50 kapsamada kesinlik {max(of['precision@cov50_abs'], of['precision@cov50_margin']):.2f}), 'yok' demeye tek başına yetmez."
    )
    r.figure(f_hist, "Şekil 6. Fine-tune sonrası mutlak skor dağılımları: 'doğru' grubu belirgin biçimde sağda, 'yok' ve 'yanlış' grupları hâlâ örtüşüyor.", width_in=6.0)
    r.figure(f_cov, "Şekil 7. Kapsama–kesinlik, mutlak skor: üç aşama.", width_in=5.8)

    r.h2("4.3 Çekim koşullarına göre")
    r.figure(f_break, "Şekil 8. Recall@1 koşula göre, üç aşama. Zor koşullardaki (ağır kapanma, küçük ölçek, yandan/arkadan) göreli kazanç en büyük.", width_in=6.3)
    rows = []
    for a in ATTRIBUTES[:-1]:
        for z_, h_, f_ in zip(*(B[v][B[v].attribute == a].to_dict("records") for v in ("zeroshot", "head", "ft"))):
            rows.append([ATTR_TR[a], z_["label"], int(z_["n"]), z_["recall@1"], h_["recall@1"], f_["recall@1"]])
    r.table(["Koşul", "Değer", "n", "Eğitimsiz", "Başlık", "Fine-tune"], rows, caption="Tablo 3. Kırılım, test, Recall@1")

    r.h1("5. Dikkat edilenler ve kısıtlar")
    r.bullets([
        "Test yine hiçbir karara girmedi: tur seçimi val_internal'da; test tek seferde ölçüldü.",
        "Tek tohum; farklar büyük (R@1 +7 puan) ama nihai raporda tohum tekrarı yapılacak.",
        "val_internal ile test arasındaki fark (0,745 / 0,633) üç aşamada da benzer: test galerisi 2,5 kat büyük ve 13 bin style=0 dikkat dağıtıcısı içeriyor.",
        "Fine-tune'un maliyeti: Colab L4'te ≈ 40 dk ve 140 MB'lık model; başlık ise 3 dk ve 4 MB. Kazanç/maliyet oranı başlıkta çok daha yüksek; fine-tune 'son birkaç puan' için.",
    ])
    r.h1("6. Projenin geldiği nokta ve kalanlar")
    r.bullets([
        "Araştırma soruları: Q1 SigLIP (Adım 2); Q2 başlık +20 puan, fine-tune +7 puan daha (Adım 4–5); Q3 ağır kapanma / küçük ölçek / yandan-arkadan / pantolon en zor (Adım 3, 5); "
        "Q4 eğitimsiz skorlar 'yok' diyemiyor, eğitimle kısmen (AUROC 0,71), margin cevabın güvenilirliğini söylüyor.",
        "Kalanlar: tohum tekrarı, IEEE rapor (taslak Kasım), sunum (Aralık), teslim zip'i.",
    ])
    r.h1("7. Dosyalar")
    r.bullets([
        "src/fashret/engine/finetune.py, scripts/finetune.py, configs/train/finetune.yaml, notebooks/colab_finetune.ipynb, checkpoints/siglip_vitb16_head_infonce.pt",
        "data/features/siglip_vitb16+ft/{val_internal,test}/ (Drive'da da var); Drive outputs/checkpoints/ft/{best.pt,last.pt,history.json}",
        "outputs/results/*_ft_*/; reports/tables/table2_head_test.csv; docs/EXPERIMENTS.md",
    ])
    out = r.save(ROOT / "docs/reports/05_kismi_fine_tune.docx")
    print("written", out)


if __name__ == "__main__":
    main()
