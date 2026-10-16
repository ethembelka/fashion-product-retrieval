# Karar Günlüğü
Yeni kararlar en üste, tarihli. Kaynak: (sohbet) = claude.ai sohbeti, (cc) = Claude Code, (hoca) = hocayla görüşme.

- 2026-10-16 (hoca/sohbet): Hoca "aynısını değil benzerini, top-5 / top-3 gibi" bulacağını sordu. Karar: çıktı en benzer K mağaza görselinin sıralı listesi; başarı = aynı ürün (pair_id + style) ilk K'da mı (Recall@K). Ek metrik: ilk K'daki sonuçların sorguyla aynı kategoride olma oranı ("benzer alternatif kalitesi"). Katalogda yoksa: "aynısı yok" + yine top-K benzer öneri.
- 2026-10-16 (sohbet): Proje önerisi yazıldı → `../Proposal/CMPE9103_Proposal_Ethem_Belka_Sahin.pdf` (teslim 18 Ekim).
- 2026-10-16 (sohbet): Test = resmi `val_query.json` (12.377) ve `val_gallery.json` (36.961). `img_info_dataframes` CSV'leri pair_id/style içermez → kullanılmaz.
- 2026-10-15 (sohbet): "Tek eğitim" kısıtı kaldırıldı; eğitimler süreye göre öncelik sırasıyla (CLAUDE.md → Eğitim planı).
- 2026-10-15 (sohbet): Retrieval yerelde çalışır (ağır kısım öznitelik çıkarma); Colab/Kaggle sadece kısmi fine-tune için.
- 2026-10-15 (hoca): DeepFashion2 ile retrieval projesi ve araştırma makaleleri (ILIAS CVPR 2025, AMES ECCV 2024) onaylandı.
- 2026-10-15 (sohbet): Yenilik: açık küme "katalogda yok" reddi; style = 0 kullanıcı ürünleri doğal katalog dışı sorgu.
