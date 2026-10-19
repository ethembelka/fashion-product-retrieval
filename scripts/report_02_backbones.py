"""Build docs/reports/02_omurga_karsilastirmasi.docx from outputs/results + reports/tables.

Usage: python scripts/report_02_backbones.py   (run scripts/make_tables.py first)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fashret.reporting import figures as fig  # noqa: E402
from fashret.reporting.docx_report import Report  # noqa: E402
from fashret.utils.io import read_json, read_table  # noqa: E402

FIG_DIR = ROOT / "reports/figures/02_backbones"
PROCESSED = ROOT / "data/processed"
MODEL_LABEL = {"resnet50": "ResNet-50 (ImageNet)", "clip_vitb16": "CLIP ViT-B/16", "siglip_vitb16": "SigLIP ViT-B/16", "dinov2_vitb14": "DINOv2 ViT-B/14"}
MODEL_INFO = {
    "resnet50": "2015'ten kalma klasik evrişimli ağ; ImageNet'te 1000 sınıf tanımak için eğitilmiş. Fotoğrafı 2048 sayıyla özetler.",
    "clip_vitb16": "OpenAI'ın 400 milyon görüntü–metin çiftiyle eğittiği model; resmi ve açıklamasını eşleştirmeyi öğrenmiş. 512 sayı.",
    "siglip_vitb16": "Google'ın CLIP benzeri ama daha iyi bir kayıp fonksiyonuyla (sigmoid) eğittiği model; 768 sayı.",
    "dinov2_vitb14": "Meta'nın hiç etiket kullanmadan 142 milyon görüntüyle eğittiği model; genel görsel yapıyı öğrenmesiyle bilinir. 768 sayı.",
}
# Match R-CNN consumer-to-shop numbers from the DeepFashion2 paper (Ge et al., CVPR 2019); filled by verify step.
MATCH_RCNN = {  # Table 5, row "pose+class", ground-truth box (verified from the paper PDF, 19.10.2026)
    "top1": 0.268, "top5": "—", "top10": 0.574, "top20": 0.665,
    "note": (
        "DeepFashion2 makalesi (Ge vd., CVPR 2019, Tablo 5) Match R-CNN için gerçek kutuyla top-1 0,268, top-10 0,574, "
        "top-20 0,665 bildirir (en iyi satır: poz + sınıf öznitelikleri; tespit edilen kutuyla 0,244 / 0,522 / 0,617). "
        "Top-5 makalede yalnızca grafikte verilir. Match R-CNN bu veri üzerinde eğitilmiş bir modeldir; bizim sayılarımız "
        "hiç eğitilmemiş modellerden gelir. Kurulum farkları: makale 'her sorgunun galeride 5'ten fazla eşi olan' bir alt küme "
        "kullanır; biz resmi val listelerini olduğu gibi kullandık ve kenarı 32 px'ten küçük 5 kaydı attık."
    ),
}


def retrieval_examples(model: str, n_ok: int = 3, n_bad: int = 2) -> tuple[Path, Path]:
    """Gallery figures: queries + top-5 (correct ones framed) for successes and failures."""
    runs = sorted((ROOT / "outputs/results").glob(f"*_{model}_zeroshot_test_s0"))
    pq = read_table(runs[-1] / "per_query.parquet")
    g_meta = read_table(ROOT / f"data/features/{model}/test/gallery_meta.parquet")
    g_emb = np.load(ROOT / f"data/features/{model}/test/gallery_emb.npy")
    q_emb = np.load(ROOT / f"data/features/{model}/test/queries_emb.npy")
    rng = np.random.default_rng(0)
    ok = pq.index[(pq.first_hit_rank == 1)].to_numpy()
    bad = pq.index[(pq.first_hit_rank == 0) | (pq.first_hit_rank > 5)].to_numpy()
    sel_ok, sel_bad = rng.choice(ok, n_ok, replace=False), rng.choice(bad, n_bad, replace=False)

    def rows(sel):
        out = []
        for i in sel:
            sims = g_emb @ q_emb[i]
            top = np.argsort(-sims)[:5]
            out.append((PROCESSED / pq.crop_path[i], [(PROCESSED / g_meta.crop_path[j], g_meta.match_key[j] == pq.match_key[i], float(sims[j])) for j in top]))
        return out

    fa = fig.retrieval_grid(rows(sel_ok), FIG_DIR / f"{model}_success.png", f"{MODEL_LABEL[model]} — doğru bulunan sorgular")
    fb = fig.retrieval_grid(rows(sel_bad), FIG_DIR / f"{model}_failure.png", f"{MODEL_LABEL[model]} — ilk 5'te bulunamayan sorgular")
    return fa, fb


def main() -> None:
    t_test = pd.read_csv(ROOT / "reports/tables/table1_backbones_test.csv")
    t_val = pd.read_csv(ROOT / "reports/tables/table1_backbones_val_internal.csv")
    best = t_test.iloc[0]["model"]
    extract = {m: read_json(ROOT / f"data/features/{m}/test/extract_report.json") for m in t_test.model}

    r = Report(
        title="Adım 2 — Eğitimsiz Omurga Karşılaştırması",
        subtitle="Dört hazır modelin müşteri→mağaza ürün eşleştirmesinde ham başarısı (Tablo 1)",
        author="Ethem Belka Şahin",
        course="CMPE 9103 Computer Vision · Dönem Projesi İlerleme Raporu · İstanbul Üniversitesi",
    )

    r.h1("1. Bu rapor ne anlatıyor?")
    r.para(
        "Adım 1'de her kıyafet küçük bir resim (kırpıntı) hâline getirilmişti. Bu adımda dört hazır yapay zekâ modeli, "
        "hiç eğitilmeden, her kırpıntıyı bir sayı dizisine (vektöre) çevirdi; sonra her müşteri fotoğrafı için mağaza "
        "kataloğundaki en benzer vektörler bulundu. Soru şu: hangi hazır model, hiçbir ek eğitim olmadan, aynı ürünü "
        "en iyi buluyor? Proje önerisindeki Araştırma Sorusu 1 budur."
    )
    r.note(
        "Vektör ne demek? Model, bir resme bakıp onu örneğin 768 sayıyla özetler. Benzer görünen resimlerin sayıları "
        "birbirine yakın olur. 'Benzerlik' bu iki sayı dizisi arasındaki açının kosinüsüdür (1 = aynı, 0 = ilgisiz)."
    )

    r.h1("2. Başlangıç noktası ve taahhüt")
    r.para("Elimizde Adım 1'den gelen listeler vardı:")
    r.bullets([
        "Test galerisi: 36.958 mağaza görüntüsü (resmi val_gallery.json).",
        "Test sorguları: 12.375 müşteri görüntüsü (resmi val_query.json).",
        "val_internal: eğitim verisinden ayrılmış 14.533 galeri / 6.236 sorgu; model seçimi burada yapılır.",
    ])
    r.para("Öneride verilen söz: ResNet-50, CLIP ViT-B/16, SigLIP ViT-B/16 ve DINOv2 ViT-B/14 modelleri dondurulmuş hâlde, "
           "L2-normalize edilmiş vektörler ve FAISS ile tam kosinüs araması kullanılarak Recall@1/5/10/20 ve mAP ile karşılaştırılacak; "
           "sonuçlar DeepFashion2 makalesindeki Match R-CNN ile kıyaslanacak.")

    r.h1("3. Karşılaştırılan modeller")
    r.table(["Model", "Nedir?", "Vektör boyutu"], [[MODEL_LABEL[m], MODEL_INFO[m], int(t_test.set_index('model').loc[m, 'embed_dim'])] for m in MODEL_LABEL],
            caption="Tablo 1. Dört omurga")

    r.h1("4. Ne yaptık?")
    flow = fig.flow_diagram(["Kırpıntılar\n(crops.zip)", "Colab L4 GPU\n4 model", "Vektörler\n(.npy, Drive)", "FAISS tam arama\n(yerel CPU)", "Recall@K, mAP\n(Tablo 2)"],
                            FIG_DIR / "pipeline.png", "Adım 2 akışı")
    r.figure(flow, "Şekil 1. Kırpıntılardan başarı tablosuna.", width_in=6.3)
    r.bullets([
        "Google Colab'da L4 GPU ile 4 model × 2 ayrım (val_internal, test) için toplam ≈ 300 bin görüntü işlendi; "
        f"hız modeline göre {min(v['lists']['gallery']['images_per_s'] for v in extract.values()):.0f}–"
        f"{max(v['lists']['gallery']['images_per_s'] for v in extract.values()):.0f} görüntü/sn, toplam ≈ 9 dakika.",
        "Vektörler Drive'a yazıldı, yerelde FAISS IndexFlatIP ile (yaklaşık değil, tam arama) her sorgu için en iyi 100 sonuç alındı.",
        "Eşleşme kuralı: aynı pair_id ve aynı style. Kategori aynı ama ürün farklıysa yanlış sayıldı.",
        "Ek metrik (hoca görüşmesi, 16 Ekim): ilk 5 sonucun sorguyla aynı kategoride olma oranı — 'aynısı yoksa benzerini öner' senaryosu.",
    ])
    r.h2("4.1 Yol boyunca düzeltilen iki şey")
    r.bullets([
        "CLIP: OpenAI ağırlıkları 'QuickGELU' adlı bir aktivasyonla eğitilmiş; ilk çalıştırmada standart GELU kullanılmıştı. "
        "Düzeltilip CLIP vektörleri yeniden çıkarıldı.",
        "Ön işleme: Kırpıntılar dikey olduğu için (yükseklik/genişlik medyanı 1,3) 'ortadan kes' adımının yaka/etek ucunu attığından "
        "şüphelenildi; tüm modelleri 224×224'e sığdırma denendi. Kazanç olmadı (ResNet kötüleşti), bu yüzden her model kendi resmi "
        "ön işlemesiyle bırakıldı. DINOv2 için 448 px, farklı havuzlama ve kenar doldurma da denendi; sonuç değişmedi.",
    ])

    r.h1("5. Sonuçlar (Tablo 2 — test ayrımı)")
    r.para("Recall@K: ilk K sonuçta en az bir doğru ürün bulunan sorguların oranı. mAP: doğru ürünlerin sıralamadaki yerini bir bütün olarak ölçer. "
           "Kat. tut.@5: ilk 5 sonucun sorguyla aynı kategoride olma oranı.")
    rows = [[MODEL_LABEL[m["model"]], m["recall@1"], m["recall@5"], m["recall@10"], m["recall@20"], m["map"], m["category_consistency@5"], int(m["embed_dim"]), m["search_ms"]]
            for m in t_test.to_dict("records")]
    r.table(["Model", "R@1", "R@5", "R@10", "R@20", "mAP", "Kat. tut.@5", "Boyut", "ms/sorgu"], rows, caption="Tablo 2. Eğitimsiz omurga karşılaştırması — test (12.375 sorgu, 36.958 galeri)")
    bar = fig.grouped_hbar([MODEL_LABEL[m] for m in t_test.model],
                           {f"Recall@{k}": [float(v) for v in t_test[f"recall@{k}"]] for k in (1, 5, 20)},
                           FIG_DIR / "recall_bars.png", "Recall@K — test ayrımı", fmt="{:.3f}")
    r.figure(bar, "Şekil 2. Dört modelin Recall@1/5/20 değerleri. SigLIP açık ara önde.", width_in=6.0)
    rows_v = [[MODEL_LABEL[m["model"]], m["recall@1"], m["recall@5"], m["recall@10"], m["recall@20"], m["map"]] for m in t_val.to_dict("records")]
    r.table(["Model", "R@1", "R@5", "R@10", "R@20", "mAP"], rows_v, caption="Tablo 3. Aynı karşılaştırma — val_internal (6.236 sorgu, 14.533 galeri)")
    r.note("val_internal sayıları test'ten biraz yüksek; galeri daha küçük (14,5 bin vs 37 bin) ve style=0 dikkat dağıtıcıları içermiyor. "
           "Sıralama her iki ayrımda aynı; bu, sonucun tesadüf olmadığını gösterir.")

    if MATCH_RCNN:
        r.h2("5.1 DeepFashion2 makalesiyle karşılaştırma")
        r.para(MATCH_RCNN["note"])
        r.table(["Yöntem", "top-1", "top-5", "top-10", "top-20"],
                [["Match R-CNN (Ge vd., 2019)", MATCH_RCNN["top1"], MATCH_RCNN["top5"], MATCH_RCNN["top10"], MATCH_RCNN["top20"]],
                 [f"{MODEL_LABEL[best]} (eğitimsiz, bu proje)", *[round(float(t_test.iloc[0][f'recall@{k}']), 3) for k in (1, 5, 10, 20)]]],
                caption="Tablo 4. Makaledeki eğitimli temel yöntem ile en iyi eğitimsiz omurga")

    r.h1("6. Sonuçları nasıl okumalı?")
    s = t_test.set_index("model")
    r.bullets([
        f"{MODEL_LABEL[best]} en iyi: her 100 müşteri fotoğrafından {s.loc[best,'recall@1']*100:.0f}'inde ilk sonuç doğru ürün, "
        f"{s.loc[best,'recall@5']*100:.0f}'inde doğru ürün ilk 5'te. Hiç eğitilmeden.",
        f"ResNet-50, CLIP ve DINOv2 çok geride (R@1 %{s['recall@1'].iloc[1]*100:.0f}–%{s['recall@1'].iloc[3]*100:.0f}). "
        "DINOv2'nin genel amaçlı görsel özelliklerde çok iyi olmasına rağmen burada zayıf kalması dikkat çekici: müşteri fotoğrafındaki "
        "ayna, oda, poz gibi 'bağlam' DINOv2 vektörüne giriyor; SigLIP ise metinle eşleşme öğrendiği için nesnenin kendisine odaklanıyor.",
        f"Kategori tutarlılığı da SigLIP'te en yüksek (%{s.loc[best,'category_consistency@5']*100:.0f}): aynısını bulamadığında bile benzer türden ürün öneriyor.",
        "Vektör boyutu ile başarı arasında ilişki yok: ResNet'in 2048 boyutlu vektörü 512 boyutlu CLIP'ten biraz iyi, 768 boyutlu SigLIP'ten çok kötü.",
        "Arama hızı: 37 bin ürünlük katalogda sorgu başına 1 ms'nin altında (CPU). Ölçek sorun değil.",
    ])

    r.h1("7. Örnekler")
    ok_fig, bad_fig = retrieval_examples(best)
    r.figure(ok_fig, "Şekil 3. Başarılı sorgular: solda müşteri fotoğrafı, sağda en benzer 5 katalog görüntüsü; yeşil çerçeve = doğru ürün.", width_in=6.3)
    r.figure(bad_fig, "Şekil 4. Başarısız sorgular: doğru ürün ilk 5'te yok. Tipik nedenler: düşük ışık, kapanma, çok benzer varyantlar.", width_in=6.3)

    r.h1("8. Karar ve sıradaki adım")
    r.bullets([
        f"Devam eden tüm adımlar (açık küme analizi, projeksiyon başlığı eğitimi) {MODEL_LABEL[best]} üzerinde yapılacak.",
        "Adım 3: 'katalogda yok' kararı (mutlak eşik vs. margin, AUROC, kapsama–kesinlik eğrisi) ve occlusion/scale/zoom/viewpoint/kategori kırılımları.",
        "Adım 4: SigLIP vektörleri üzerinde küçük bir projeksiyon başlığı eğitimi (InfoNCE) — Araştırma Sorusu 2.",
    ])
    r.h1("9. Ortaya çıkan dosyalar")
    r.bullets([
        "data/features/<model>/<ayrım>/{gallery,queries,openset}_emb.npy + _meta.parquet (Drive'da da var)",
        "outputs/results/<run_id>/{metrics.json, per_query.parquet, config.yaml, git_hash.txt} — 8 çalıştırma",
        "reports/tables/table1_backbones_{test,val_internal}.csv, table1_backbones.tex, all_runs.csv",
        "docs/EXPERIMENTS.md — deney kaydı",
    ])
    out = r.save(ROOT / "docs/reports/02_omurga_karsilastirmasi.docx")
    print("written", out)


if __name__ == "__main__":
    main()
