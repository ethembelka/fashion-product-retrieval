"""Build docs/reports/02_omurga_karsilastirmasi.docx — what an embedding is, how the four backbones work, how the
metrics are computed, and the training-free comparison (Table 1). Written for a reader learning the subject.

Usage: python scripts/report_02_backbones.py   (after scripts/make_tables.py)
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
MATCH_RCNN = {"top1": 0.268, "top10": 0.574, "top20": 0.665, "det_top1": 0.244, "det_top10": 0.522, "det_top20": 0.617}  # DF2 paper Table 5, pose+class row


def run_dir(model: str, split: str) -> Path:
    return sorted((ROOT / "outputs/results").glob(f"*_{model}_zeroshot_{split}_s0"))[-1]


def retrieval_examples(model: str, n_ok: int = 3, n_bad: int = 2) -> tuple[Path, Path]:
    pq = read_table(run_dir(model, "test") / "per_query.parquet")
    g_meta = read_table(ROOT / f"data/features/{model}/test/gallery_meta.parquet")
    g_emb = np.load(ROOT / f"data/features/{model}/test/gallery_emb.npy")
    q_emb = np.load(ROOT / f"data/features/{model}/test/queries_emb.npy")
    rng = np.random.default_rng(0)
    ok = pq.index[(pq.first_hit_rank == 1)].to_numpy()
    bad = pq.index[(pq.first_hit_rank == 0) | (pq.first_hit_rank > 5)].to_numpy()

    def rows(sel):
        out = []
        for i in sel:
            sims = g_emb @ q_emb[i]
            top = np.argsort(-sims)[:5]
            out.append((PROCESSED / pq.crop_path[i], [(PROCESSED / g_meta.crop_path[j], g_meta.match_key[j] == pq.match_key[i], float(sims[j])) for j in top]))
        return out

    fa = fig.retrieval_grid(rows(rng.choice(ok, n_ok, replace=False)), FIG_DIR / f"{model}_success.png", f"{MODEL_LABEL[model]} — ilk sonucu doğru olan sorgular")
    fb = fig.retrieval_grid(rows(rng.choice(bad, n_bad, replace=False)), FIG_DIR / f"{model}_failure.png", f"{MODEL_LABEL[model]} — doğru ürünü ilk 5'e koyamadığı sorgular")
    return fa, fb


def cosine_example() -> tuple[list[list[object]], float, float]:
    """Tiny numeric example for the text: two 3-D vectors, normalise, dot product."""
    a, b = np.array([3.0, 4.0, 0.0]), np.array([1.0, 2.0, 2.0])
    an, bn = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return [["a", *a, float(np.linalg.norm(a)), *np.round(an, 3)], ["b", *b, float(np.linalg.norm(b)), *np.round(bn, 3)]], float(an @ bn), float(a @ b)


def main() -> None:
    t_test = pd.read_csv(ROOT / "reports/tables/table1_backbones_test.csv")
    t_val = pd.read_csv(ROOT / "reports/tables/table1_backbones_val_internal.csv")
    best = t_test.iloc[0]["model"]
    s = t_test.set_index("model")
    extract = {m: read_json(ROOT / f"data/features/{m}/test/extract_report.json") for m in t_test.model}
    pq_best = read_table(run_dir(best, "test") / "per_query.parquet")
    any_crop = PROCESSED / pq_best.crop_path.iloc[0]

    f_flow = fig.flow_diagram(["Kırpıntı\n(224×224×3 matris)", "Dondurulmuş\nmodel", "Vektör\n(512–2048 sayı)", "L2 normalize\n(uzunluk = 1)", "Kosinüs araması\n(FAISS)"],
                              FIG_DIR / "embed_flow.png", "Bir kırpıntıdan arama sonucuna")
    f_patch = fig.patch_grid(any_crop, 16, 224, FIG_DIR / "vit_patches.png", "Vision Transformer girişi: görüntü 16×16 piksellik parçalara bölünür")
    f_bar = fig.grouped_hbar([MODEL_LABEL[m] for m in t_test.model], {f"Recall@{k}": [float(v) for v in t_test[f"recall@{k}"]] for k in (1, 5, 20)},
                             FIG_DIR / "recall_bars.png", "Recall@K — test ayrımı", fmt="{:.3f}")
    f_ok, f_bad = retrieval_examples(best)
    cos_rows, cos, raw_dot = cosine_example()

    r = Report(
        title="Adım 2 — Eğitimsiz Omurga Karşılaştırması",
        subtitle="Görüntüden vektöre: dört hazır model nasıl çalışır, başarı nasıl ölçülür, hangisi en iyi (Tablo 1)",
        author="Ethem Belka Şahin",
        course="CMPE 9103 Computer Vision · Dönem Projesi İlerleme Raporu · İstanbul Üniversitesi",
    )

    # ------------------------------------------------------------------ 1
    r.h1("1. Bu adımın sorusu")
    r.para(
        "Adım 1'de her kıyafet bir kırpıntı (küçük JPEG) oldu. Bir müşteri kırpıntısına en çok benzeyen mağaza kırpıntısını bulmak için "
        "iki görüntüyü karşılaştırabilen bir ölçü gerekir. Piksel piksel karşılaştırma işe yaramaz: aynı tişört farklı ışıkta, farklı pozda, "
        "farklı arka planda tamamen farklı piksel değerleri verir. Bunun yerine her görüntü önce bir 'vektöre' çevrilir ve vektörler karşılaştırılır. "
        "Bu adımın sorusu: hiç ek eğitim yapmadan, hazır dört modelden hangisi bu iş için en iyi vektörü üretir? (Araştırma Sorusu 1)"
    )
    r.figure(f_flow, "Şekil 1. Her kırpıntı için izlenen yol. Model 'dondurulmuş': ağırlıkları hiç değiştirilmiyor.", width_in=6.3)

    # ------------------------------------------------------------------ 2
    r.h1("2. Temel kavramlar")
    r.h2("2.1 Görüntüden vektöre (gömme / embedding)")
    r.para(
        "Bir kırpıntı, 224×224×3 = 150.528 sayıdan oluşan bir matristir. Bir derin öğrenme modeli, bu matrisi girdi alıp çıktı olarak sabit uzunlukta "
        "bir sayı dizisi veren bir fonksiyondur: f(görüntü) = (v₁, v₂, …, v_D). D modele göre 512, 768 ya da 2048. Bu diziye vektör ya da gömme (embedding) denir."
    )
    r.para(
        "Fonksiyonun içinde milyonlarca ayarlanabilir sayı (ağırlık) vardır. 'Eğitim', bu ağırlıkların büyük bir veri kümesi üzerinde, belirli bir "
        "hedefi iyileştirecek şekilde yavaş yavaş değiştirilmesidir. 'Hazır (pretrained) model', başkalarının bu eğitimi yapıp ağırlıkları yayınladığı modeldir. "
        "'Eğitimsiz (zero-shot) kullanım' ise bu ağırlıkları olduğu gibi alıp hiç değiştirmeden kendi verimize uygulamak demektir; bu adımda yapılan budur."
    )
    r.para(
        "İyi bir vektörün özelliği şudur: aynı ürünün iki farklı fotoğrafı birbirine yakın vektörler verir, farklı ürünler uzak vektörler. "
        "Modelin eğitim hedefi bu özelliği ne kadar üretiyorsa, o model bizim işimizde o kadar iyi olur. Dört modelin eğitim hedefleri farklı; fark da buradan çıkar (Bölüm 3)."
    )
    r.h2("2.2 İki vektör ne kadar benzer? Kosinüs benzerliği")
    r.para(
        "İki vektörün benzerliği aralarındaki açıyla ölçülür. Önce her vektör kendi uzunluğuna bölünür (L2 normalizasyon); böylece her vektörün uzunluğu 1 olur "
        "ve yalnızca yönü kalır. Sonra iki birim vektörün nokta çarpımı (karşılıklı elemanları çarpıp toplamak) alınır. Bu sayı aradaki açının kosinüsüdür: "
        "1 = aynı yön (en benzer), 0 = dik (ilgisiz), −1 = zıt. Üç boyutlu küçük bir örnek:"
    )
    r.table(["vektör", "v₁", "v₂", "v₃", "uzunluk", "normalize v₁", "v₂", "v₃"], cos_rows, caption="Tablo 1. L2 normalizasyon örneği")
    r.para(
        f"a'nın uzunluğu √(3²+4²+0²) = 5; b'ninki √(1+4+4) = 3. Normalize edilmiş vektörlerin nokta çarpımı: 0,6·0,333 + 0,8·0,667 + 0·0,667 = {cos:.3f}. "
        f"Bu iki vektörün kosinüs benzerliği {cos:.3f}. (Ham nokta çarpım {raw_dot:.0f} olurdu; normalize etmeden uzun vektörler haksız yere 'benzer' görünür, bu yüzden normalize ediyoruz.) "
        "Bizim vektörlerimiz 768 boyutlu ama işlem aynı."
    )
    r.h2("2.3 Arama: her sorguyu bütün katalogla karşılaştırmak")
    r.para(
        f"Test kataloğunda {int(s.loc[best,'n_gallery']) if 'n_gallery' in s else 36958:,} vektör var. Bir sorgu vektörü için bütün katalog vektörleriyle nokta çarpımı alınır "
        "(tek bir matris çarpımı: sorgu 1×768, katalog 768×36.958 → 36.958 benzerlik sayısı), sayılar büyükten küçüğe sıralanır, ilk K tanesi cevaptır. "
        "Bu işi FAISS kütüphanesinin IndexFlatIP yapısı yapar: 'Flat' = bütün vektörler olduğu gibi bellekte, 'IP' = iç çarpım (inner product). "
        "Yaklaşık arama yöntemleri (katalogun bir kısmına bakıp hız kazanan) kullanılmadı; her sorgu için kesin sonuç alınır. "
        f"Sorgu başına süre {s.loc[best,'search_ms']:.2f} ms (CPU).".replace(",", ".")
    )

    # ------------------------------------------------------------------ 3
    r.h1("3. Karşılaştırılan dört model")
    r.para("Dördü de 'görüntü → vektör' fonksiyonudur; farkları mimari (hesabı nasıl yaptıkları) ve eğitim hedefi (ağırlıkların ne için ayarlandığı).")

    r.h2("3.1 ResNet-50 — evrişimli ağ, ImageNet sınıflandırmasıyla eğitilmiş")
    r.para(
        "Mimari: Evrişimli sinir ağı (CNN). Temel işlem evrişimdir: 3×3 piksellik küçük bir ağırlık tablosu (filtre) görüntünün üzerinde kaydırılır; "
        "her konumda filtre ile altındaki 3×3 piksel bloğunun ağırlıklı toplamı alınır ve tek bir sayı üretilir. Bir filtre görüntünün tamamı üzerinde "
        "gezince yeni bir 'harita' çıkar; 64 filtre 64 harita üretir. Bu haritalara aynı işlem tekrar uygulanır: 50 katman üst üste. Alt katmanlar kenar ve "
        "renk geçişi gibi basit şeylere, üst katmanlar desen, doku ve nesne parçalarına tepki verir. Her birkaç katmanda haritalar yarıya küçültülür (pooling), "
        "böylece üst katmanlar görüntünün daha geniş bölgesini görür. 'Res' (residual): her katman bloğu girdisini çıktısına ekler (çıktı = girdi + öğrenilen düzeltme); "
        "bu, 50 katmanlı derin ağın eğitilebilmesini sağlayan buluştur (2015)."
    )
    r.para(
        "Eğitim hedefi: ImageNet'teki 1,28 milyon fotoğrafı 1000 sınıfa (köpek cinsleri, araç türleri, nesneler…) ayırmak. Son katman bu 1000 sınıfın puanını verir. "
        "Biz son katmanı atıp bir öncekini kullanıyoruz: son haritaların her birinin ortalaması alınır (global average pooling) → 2048 sayı. "
        "Bu vektör 'ImageNet sınıflarını ayırt etmeye yarayan' bilgiyi taşır; kıyafet kimliği için eğitilmemiştir."
    )

    r.h2("3.2 Vision Transformer (ViT) — diğer üç modelin ortak mimarisi")
    r.para(
        "CLIP, SigLIP ve DINOv2 aynı mimariyi kullanır: Vision Transformer, 'Base' boyutu (ViT-B, 86 milyon ağırlık). Çalışması:"
    )
    r.numbered([
        "Görüntü 224×224'e getirilir ve 16×16 piksellik parçalara (patch) bölünür: 14×14 = 196 parça (DINOv2'de parça 14×14 piksel, 256 parça). Şekil 2.",
        "Her parça 16×16×3 = 768 sayıdır; bu sayılar bir ağırlık matrisiyle çarpılarak 768 boyutlu bir 'token'a dönüşür. Parçanın konumunu belirten öğrenilmiş bir vektör eklenir (ağ aksi hâlde parçaların sırasını bilmez).",
        "Başa, görüntüye ait olmayan özel bir token eklenir: CLS. Toplam 197 token.",
        "12 katman 'öz-dikkat' (self-attention): her token, diğer bütün token'lara 'bakar'. Somut olarak: her token için üç vektör hesaplanır (sorgu q, anahtar k, değer v = token × üç ayrı ağırlık matrisi); "
        "token i'nin token j'ye vereceği ağırlık qᵢ·kⱼ nokta çarpımından (softmax ile 0–1 arasına ve toplamı 1'e getirilerek) bulunur; token i'nin yeni değeri, bütün vⱼ'lerin bu ağırlıklarla toplamıdır. "
        "Böylece 'yaka' parçası 'etek ucu' parçasından bilgi alabilir — CNN'de bu ancak çok katman sonra mümkündür. Her katmanda ayrıca küçük bir tam bağlantılı ağ (MLP) her token'ı ayrı ayrı dönüştürür.",
        "12 katman sonunda CLS token'ının değeri (768 sayı) görüntünün özeti olarak alınır. (CLIP'te bunun üstünde bir projeksiyon daha var → 512.)",
    ])
    r.figure(f_patch, "Şekil 2. ViT'in gördüğü şey: 224×224 görüntü, 196 parça. Her parça bir token olur; dikkat mekanizması parçaları birbirine bağlar.", width_in=5.0)

    r.h2("3.3 CLIP — görüntü ve metni aynı uzayda eşleştirmeyi öğrenmiş")
    r.para(
        "OpenAI, 2021. İki ayrı ağ: görüntü için ViT-B/16, metin için bir Transformer. İnternetten toplanan 400 milyon (fotoğraf, açıklama metni) çifti ile eğitildi. "
        "Eğitim hedefi (kontrastif): bir grupta N çift alınır (örneğin N = 32.768). N görüntü vektörü ile N metin vektörünün bütün ikili kosinüs benzerlikleri hesaplanır (N×N tablo). "
        "Hedef, her satırda doğru eşin (köşegen) diğer N−1 yanlış eşten yüksek çıkması: her satıra softmax uygulanıp doğru hücrenin olasılığı artırılır. "
        "Sonuçta görüntü vektörleri, 'metinle anlatılabilen içeriğe' göre düzenlenir: 'kırmızı çiçekli elbise' yazısına yakın görüntüler birbirine de yakın olur. "
        "Biz yalnızca görüntü ağını kullanıyoruz; vektör 512 boyutlu."
    )
    r.note(
        "Bu projede fark edilen bir ayrıntı: OpenAI ağırlıkları 'QuickGELU' adlı bir aktivasyon fonksiyonuyla (x·sigmoid(1,702x)) eğitilmiş; kütüphanenin varsayılanı "
        "standart GELU'ydu. Ağırlıklar aynı ama ara hesaplar biraz farklı çıkınca vektörler bozuluyordu. Düzeltilip CLIP vektörleri yeniden çıkarıldı (256 örneklik denemede R@5 0,18 → 0,22)."
    )

    r.h2("3.4 SigLIP — CLIP'in geliştirilmiş kaybıyla eğitilmiş")
    r.para(
        "Google, 2023. Aynı kurulum (ViT-B/16 görüntü ağı + metin ağı, görüntü–metin çiftleri; WebLI veri seti, milyarlarca çift). Fark eğitim hedefinde: "
        "CLIP her satırı N'li bir yarış olarak değerlendirir (softmax, bütün grubun normalizasyonu gerekir). SigLIP ise her (görüntü, metin) hücresini bağımsız bir evet/hayır sorusu olarak "
        "ele alır: doğru çift için sigmoid(benzerlik) → 1'e, yanlış çiftler için → 0'a itilir. Bu, çok daha büyük gruplarla ve daha kararlı eğitim sağlar; yayınlanan "
        "sonuçlarda aynı boyutta CLIP'ten daha iyi görüntü vektörleri verir. Vektör 768 boyutlu. Ön işlemesi de farklı: görüntüyü kırpmadan 224×224'e sıkıştırır."
    )

    r.h2("3.5 DINOv2 — etiketsiz, kendi kendine öğrenmiş")
    r.para(
        "Meta, 2023. Ne sınıf etiketi ne metin kullanır (self-supervised). 142 milyon özenle seçilmiş fotoğrafla eğitildi. Yöntem: aynı fotoğraftan iki farklı "
        "görünüm üretilir (farklı bölge kırpılır, renk oynanır). 'Öğrenci' ağ bir görünümü, 'öğretmen' ağ diğerini görür; öğrenci, öğretmenin çıktısını tahmin etmeye zorlanır. "
        "Öğretmen ayrı eğitilmez; ağırlıkları öğrencinin ağırlıklarının yavaş hareket eden ortalamasıdır. Ek olarak görüntünün bazı parçaları gizlenir ve öğrenci "
        "gizli parçaların öğretmen çıktısını tahmin eder. Sonuçta ağ, 'aynı fotoğrafın parçaları birbiriyle tutarlı olmalı' ilkesinden görsel yapı öğrenir; "
        "nesne parçaları, doku ve geometri konusunda çok güçlüdür. Vektör 768 boyutlu (CLS token). Orijinal çözünürlük 518 piksel; biz 224'te çalıştırdık (konum vektörleri yeniden örneklendi)."
    )

    r.h2("3.6 Ön işleme ve hassasiyet")
    r.para(
        "Her modelin kendi resmi ön işleme adımı kullanıldı: ResNet kısa kenarı 235'e getirip ortadan 224×224 keser; CLIP ve DINOv2 224'e getirip ortadan keser; SigLIP ortadan kesmeden 224×224'e sıkıştırır. "
        "Sonra her kanaldan modelin eğitimde kullandığı ortalama çıkarılır, standart sapmaya bölünür (ResNet/DINOv2: ortalama 0,485/0,456/0,406, sapma 0,229/0,224/0,225). "
        "Hesaplar 16-bit kayan noktayla (fp16) yapıldı: hız iki kat, bellek yarı; 256 örnekte fp32 ile sonuç birebir aynı çıktı."
    )
    r.note(
        "Denenen ve reddedilen bir fikir: kırpıntılar dikey olduğu için (yükseklik/genişlik medyanı 1,3) 'ortadan kes' adımının yaka ve etek ucunu attığı düşünüldü; "
        "dört modele de 'sıkıştır' uygulandı. 256 sorguluk bir denemede ResNet kötüleşti (R@1 0,094 → 0,059), diğerleri değişmedi. Resmi ön işlemelere dönüldü. "
        "DINOv2 için 448 piksel, parça-ortalaması havuzlama ve kenar doldurma da denendi; hiçbiri 0,10'u geçmedi."
    )

    # ------------------------------------------------------------------ 4
    r.h1("4. Başarı nasıl ölçülüyor?")
    r.para("Her sorgu için katalog sıralanır; doğru ürünler (aynı match_key) bu sıralamanın neresinde? Dört ölçü, hepsi 0–1 arası, yüksek iyi.")
    r.definition("Recall@K", "doğru ürünlerden en az biri ilk K sonuç içindeyse sorgu 'başarılı'. Recall@K = başarılı sorgu sayısı / toplam sorgu. "
                 "K = 1 en katı ('ilk sonuç doğru mu'), K = 20 en gevşek. Makalelerde 'top-K accuracy' de denir.")
    r.definition("mAP (mean average precision)", "yalnızca ilk doğru sonuca değil bütün doğru sonuçların sırasına bakar. Bir sorgu için AP: her doğru sonucun bulunduğu sırada "
                 "'o sıraya kadar doğru oranı' hesaplanır, bunlar toplanıp katalogdaki doğru ürün sayısına bölünür. mAP bütün sorguların AP ortalaması.")
    r.para("Örnek: bir sorgunun katalogda 2 doğru eşi var; sistemin sıralamasında bunlar 2. ve 5. sırada çıktı:", "")
    r.table(["Sıra", "1", "2", "3", "4", "5", "6…"], [["Doğru mu?", "hayır", "EVET", "hayır", "hayır", "EVET", "hayır"], ["O sıraya kadar doğru oranı", "0/1", "1/2", "1/3", "1/4", "2/5", ""]],
            caption="Tablo 2. Tek sorgu için örnek sıralama")
    r.bullets([
        "Recall@1 = 0 (ilk sonuç yanlış), Recall@5 = 1 (5 içinde doğru var), Recall@20 = 1.",
        "AP = (1/2 + 2/5) / 2 = 0,45. İkinci doğru 50. sırada olsaydı AP = (1/2 + 2/50)/2 = 0,27: mAP geç bulunan doğruları cezalandırır, Recall@K cezalandırmaz.",
    ])
    r.definition("Kategori tutarlılığı@5", "ilk 5 sonucun sorguyla aynı kategoride (tişört, etek…) olma oranı. Hoca görüşmesinde (16 Ekim) istenen 'aynısı yoksa benzerini öner' "
                 "senaryosunun ölçüsü: sistem doğru ürünü bulamasa bile aynı türden bir şey mi öneriyor?")
    r.definition("Boyut, ms/sorgu, indeks MB", "verimlilik: vektör uzunluğu, sorgu başına arama süresi, kataloğun bellekte kapladığı yer (vektör sayısı × boyut × 4 bayt).")

    # ------------------------------------------------------------------ 5
    r.h1("5. Ne yapıldı?")
    r.bullets([
        "crops.zip Google Drive'a yüklendi; Colab'da L4 GPU ile açıldı (339 s).",
        "Dört model × iki ayrım (val_internal, test) × üç liste (galeri, sorgu, açık küme): toplam ≈ 300 bin kırpıntı vektöre çevrildi. "
        f"Hız modele göre {min(v['lists']['gallery']['images_per_s'] for v in extract.values()):.0f}–{max(v['lists']['gallery']['images_per_s'] for v in extract.values()):.0f} görüntü/sn; toplam ≈ 9 dakika; GPU belleği < 1 GB.",
        "Her liste için vektörler .npy (ham sayı dizisi) dosyasına, satır sırası listeyle aynı olacak şekilde yazıldı; sıra uyumu kodda doğrulanıyor. Drive'a ve yerel diske kopyalandı.",
        "Değerlendirme yerel bilgisayarda CPU ile: FAISS tam arama, ilk 100 sonuç, metrikler. Her çalıştırma outputs/results/<run_id>/ altına ayarları, git sürümünü ve metrics.json'u yazar.",
    ])

    # ------------------------------------------------------------------ 6
    r.h1("6. Sonuçlar — Tablo 1")
    rows = [[MODEL_LABEL[m["model"]], m["recall@1"], m["recall@5"], m["recall@10"], m["recall@20"], m["map"], m["category_consistency@5"], int(m["embed_dim"]), m["search_ms"]]
            for m in t_test.to_dict("records")]
    r.table(["Model", "R@1", "R@5", "R@10", "R@20", "mAP", "Kat. tut.@5", "Boyut", "ms/sorgu"], rows, caption="Tablo 3 (= makaledeki Tablo 1). Test ayrımı: 12.375 sorgu, 36.958 galeri")
    r.figure(f_bar, "Şekil 3. Recall@1/5/20. SigLIP her K'da diğerlerinin 3–5 katı.", width_in=6.0)
    rows_v = [[MODEL_LABEL[m["model"]], m["recall@1"], m["recall@5"], m["recall@10"], m["recall@20"], m["map"]] for m in t_val.to_dict("records")]
    r.table(["Model", "R@1", "R@5", "R@10", "R@20", "mAP"], rows_v, caption="Tablo 4. Aynı karşılaştırma val_internal'da (6.236 sorgu, 14.533 galeri)")
    r.para(
        "val_internal sayıları test'ten yüksek: galeri 2,5 kat küçük ve test galerisindeki 13 bin style = 0 dikkat dağıtıcısını içermiyor. "
        "Önemli olan sıralamanın iki ayrımda da aynı olması; sonuç tek bir veri kümesine özgü değil."
    )
    r.h2("6.1 Makaledeki eğitimli yöntemle karşılaştırma")
    r.para(
        "DeepFashion2 makalesi (Tablo 5) kendi yöntemi Match R-CNN için şu sayıları verir (gerçek kutuyla): top-1 0,268, top-10 0,574, top-20 0,665. "
        "Match R-CNN, bu veri setinin 191 bin eğitim fotoğrafıyla eğitilmiş, kıyafeti tespit edip vektör çıkaran özel bir ağdır. "
        "Top-5 makalede yalnızca grafikte var. Kurulum farkı: makale yalnızca 'galeride 5'ten fazla eşi olan' sorguları kullanır; biz resmi listeyi olduğu gibi kullandık."
    )
    r.table(["Yöntem", "Eğitim", "top-1", "top-5", "top-10", "top-20"],
            [["Match R-CNN (Ge vd., 2019), GT kutu", "DeepFashion2 ile eğitilmiş", MATCH_RCNN["top1"], "—", MATCH_RCNN["top10"], MATCH_RCNN["top20"]],
             ["Match R-CNN, tespit edilen kutu", "DeepFashion2 ile eğitilmiş", MATCH_RCNN["det_top1"], "—", MATCH_RCNN["det_top10"], MATCH_RCNN["det_top20"]],
             [f"{MODEL_LABEL[best]} (bu proje)", "hiç", *[round(float(s.loc[best, f"recall@{k}"]), 3) for k in (1, 5, 10, 20)]]],
            caption="Tablo 5. Eğitimli temel yöntem ile en iyi eğitimsiz omurga")
    r.para("Hiç eğitilmemiş SigLIP, bu veri için eğitilmiş Match R-CNN'i top-1'de 9 puan geçiyor. 2019'dan bu yana büyük veriyle eğitilmiş genel amaçlı modellerin gücü budur.")

    # ------------------------------------------------------------------ 7
    r.h1("7. Yorum")
    r.bullets([
        f"SigLIP: her 100 sorgudan {s.loc[best,'recall@1']*100:.0f}'inde ilk sonuç doğru ürün, {s.loc[best,'recall@5']*100:.0f}'inde doğru ürün ilk 5'te. "
        "Metinle eşleşme hedefi, modeli 'bu görüntüde hangi nesne var' sorusuna odaklar; arka plan, ışık, poz vektöre az girer.",
        f"CLIP aynı fikirle eğitilmiş ama çok geride ({s.loc['clip_vitb16','recall@1']:.3f}). Fark: daha küçük/kirli veri, softmax kaybı, 512'ye sıkıştırılmış vektör ve ortadan kesme.",
        f"DINOv2 ({s.loc['dinov2_vitb14','recall@1']:.3f}): genel görsel özelliklerde çok iyi bilinen bu modelin burada zayıf kalması dikkat çekici. Kendi kendine öğrenme 'aynı fotoğrafın parçaları' "
        "üzerinden yapıldığı için vektör, fotoğrafın bütününü (ayna, oda, vücut pozu) temsil eder; müşteri ve mağaza fotoğrafları arasındaki ortam farkını aşamaz. "
        "Yukarıda anlatılan çözünürlük/havuzlama denemeleri bunun bir ayar hatası olmadığını gösteriyor.",
        f"ResNet-50 ({s.loc['resnet50','recall@1']:.3f}): ImageNet sınıfları (köpek, araba…) kıyafet varyantlarını ayırmak için eğitilmemiş; 2048 boyutlu vektör bile yardım etmiyor.",
        f"Kategori tutarlılığı da SigLIP'te en yüksek (%{s.loc[best,'category_consistency@5']*100:.0f}): doğru ürünü bulamadığında bile çoğunlukla aynı türden ürün getiriyor.",
    ])

    # ------------------------------------------------------------------ 8
    r.h1("8. Örnekler")
    r.figure(f_ok, "Şekil 4. Doğru bulunan sorgular. Her satırda solda müşteri kırpıntısı, sağda en benzer 5 katalog kırpıntısı ve kosinüs skorları; yeşil çerçeve = aynı ürün. "
             "Baskılı tişört ve belirgin desenli etekler kolay.", width_in=6.3)
    r.figure(f_bad, "Şekil 5. Doğru ürünü ilk 5'e koyamadığı sorgular. Getirilenler 'yanlış' ama görsel olarak çok benzer (aynı desen, aynı kesim): "
             "model tür/desen benzerliği öğrenmiş, ürün kimliği değil.", width_in=6.3)

    # ------------------------------------------------------------------ 9
    r.h1("9. Karar ve sıradaki adım")
    r.bullets([
        f"Bundan sonraki adımlar {MODEL_LABEL[best]} vektörleri üzerinde yapılacak.",
        "Adım 3: 'katalogda yok' kararı ve başarının çekim koşullarına göre kırılımı — GPU gerekmez, bu adımın vektörleri yeter.",
        "Adım 4: SigLIP vektörlerinin üstüne küçük bir öğrenilmiş katman (projeksiyon başlığı) ekleyip DeepFashion2'nin kendi eğitim verisiyle eğitmek. "
        "Hedef, 'benzer ama farklı' ürünleri ayırmayı öğretmek (Şekil 5'teki hata türü).",
    ])
    r.h1("10. Dosyalar")
    r.bullets([
        "data/features/<model>/<ayrım>/{gallery,queries,openset}_emb.npy + _meta.parquet (Drive'da da var)",
        "outputs/results/<run_id>/metrics.json, per_query.parquet — 8 çalıştırma",
        "reports/tables/table1_backbones_{test,val_internal}.csv, table1_backbones.tex; docs/EXPERIMENTS.md",
        "src/fashret/models/ (dört sarmalayıcı), engine/extractor.py, evaluation/closed_set.py; tests/test_closed_set.py (elle hesaplanmış örnekler)",
    ])
    out = r.save(ROOT / "docs/reports/02_omurga_karsilastirmasi.docx")
    print("written", out)


if __name__ == "__main__":
    main()
