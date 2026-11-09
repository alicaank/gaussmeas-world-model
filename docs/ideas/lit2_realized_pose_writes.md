# Üretilen kareleri kestirilen pozla belleğe yazan sistemler (yerel kod/paper, alt-ajan, 2026-10-09; V/I)
Cevap: yalnız **Warp-as-History** (Pi3X) ve **AlayaWorld ViGeo** yolu kestirilen pozla yazıyor; Robust Dreamer örtük CUT3R dünya
çerçevesiyle kısmen. Diğer herkes komut edilen pozla yazıyor (DA3/CUT3R/MoGe'yi yalnız depth için koşsa bile). Koşullama her
sistemde komut edilen hedef pozda; tek kısmi istisna Warp-as-History (komut edilen göreli hareket, son karenin kestirilen pozu
üstüne bindirilir). Hiçbiri kestirilen-vs-komut yazım pozu ablasyonu raporlamıyor. Bu, M-real'in literatürde **boş** olduğunu
ve test edilmesi gerektiğini gösteriyor; uyarı: WorldPlay2, Alaya'nın chunk'lar arası metrik ölçek belirsizliğinden geometrik
tutarsızlık çektiğini yazıyor (worldplay2.txt:578-581) — tek seferlik prefix ölçeğiyle uyumlu.

| Sistem | Bellek | Yazım pozu | Depth | Koşullama pozu | Rapor |
|---|---|---|---|---|---|
| Warp-as-History | kare başına Pi3X pointmap, mesh+splat; kalıcı dünya çerçevesi yok (2 keyframe penceresi) | **Pi3X kestirimi** ("pose_source": "pi3x_estimated"; camera_warp.py:2140-2236), ölçek = ilk-kare depth medyan oranı (:353-415) | Pi3X | komut edilen göreli hareket ∘ kestirilen kaynak poz (camera_warp.py:2514-2515); kamera yalnız warp videosundan | ablasyon yok |
| AlayaWorld ViGeo | akan ViGeo nokta cache'i, ViGeo KV cache chunk'lar arası | `aligned_vigeo`: ViGeo pozları P0'a göre, son prefix kamerası planlanana rijit düzeltilir (rollout_trainer.py:3864-3954); ölçek prefix'te bir kez (vigeo_geometry.py:283-301) | ViGeo pointmap, warp artığı büyükse dinamik maske | planlanan kameralar (rollout_trainer.py:3967-3970) | kendi ablasyonu yok; WorldPlay2 eleştirisi |
| AlayaWorld FlashAlaya DA3 yolu | nokta | **komut** (DA3 poz-koşullu, align_to_input_ext_scale) | DA3 | komut | — |
| Robust Dreamer | latent Gaussian bellek | CUT3R-başlatmalı baş kendi global koordinatında pointmap → örtük kestirim; komut çerçevesiyle hizalama anlatılmıyor | CUT3R tarzı | komut (c_i) | poz ablasyonu yok |
| GEAR | kare başına mesh + Invisible Octree | komut (DA3 poz-koşullu; "pozsuz depth değerlendirme yörüngesiyle tutarsız olabilir", gear.txt:1146-1155) | DA3 | komut | — |
| Lyra 2 | kare başına depth+kamera, asla fuse edilmez | komut (DA3 (T_i, K_i) ile); opsiyonel --da3_use_predicted_pose | DA3 | komut | global bulut: CamCtrl 63.87 → 49.86 |
| GeoVerse | fuse edilmiş renkli bulut | komut; kestirilen raymap yalnız ölçek s* | tahmin depth | komut | "ilk depth hatası ve görünüş sapması bellek güncellemeleriyle birikir" |
| VMem | surfel | komut (preset_pose donuk; CUT3R yalnız depth/focal) | CUT3R | komut | — |
| WorldStereo | MoGe yalnız ilk kare; üretilenler komut w2c ile eklenir ("No alignment") | komut | — | komut | README kodla çelişiyor |
| WorldCrafter / WorldPlay / WorldMem | örtük/poz-indeksli | komut (WorldCompass DA3 yalnız RL ödülü) | — | komut | WorldPlay2: WorldPlay "birleşik poz sapması" |
| MoVerse, WorldWeave, Wonder, Memorizon, WorldPlay2, LongTake | üretilen kare geri yazılmaz / sıkıştırılmış | — | — | komut | — |
