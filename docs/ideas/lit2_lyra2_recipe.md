# Lyra 2 eğitim tarifi (yerel kod + paper'dan, 2026-10-09; alt-ajan okuması; V = kodda doğrulandı, I = çıkarım)
M = lyra/Lyra-2/lyra_2/_src/models/lyra2_model.py, E = _src/configs/experiment.py, P = papers/lyra2.txt, T = Lyra-2/TRAINING.md

(a) Temporal bozulma = self-augmentation: steps=1, guidance=1.0, every_k=2, prob=1.0, max_T=500, copy_chunk=True,
    encode_gt_with_clean_history=True (E:45-53); self_aug_i2v_ratio=0.3 (M:80). Çift iterasyonlarda, rank-0'da tek çekiliş;
    >0.3 ise self-aug (≈%35 tüm adımlar), değilse zorla I2V segment 0 (≈%15); tek adımlar düz (M:1223-1239, 1400-1404) V.
    t: logit-normal çekiliş (sigmoid(randn)) → shift=5 scheduler → ×500/1000 → en yakın geçerli timestep (M:1254-1268) V;
    paper U(0,0.5) der (P:464), kod logit-normal (I). Yalnız en yeni 20 latent kare bozulur, eski history temiz (M:1247-1284);
    tek denoise adımı CFG'siz, decode edilip videoya geri yazılır (M:1300-1386); augment'li önceki chunk VAE ile yeniden encode,
    hedef temiz cache ile (M:2293-2318); eski history orijinal temiz latent (M:2330-2331) V.
    Ana kayıpta history eps = x0 (temiz kalır), yalnız hedef gürültülü (M:1446-1450); kayıp yalnız üretilen bölgede, reweighting
    (M:1480-1484, E:57). Tek ortak timestep (wan2pt1_lyra2.py:824-825, 963): context t=0 ALMAZ V.
    İkinci bozulma yalnız spatial slotlara: noise_with_sigma, log σ ~ N(-3, 2) her adımda; latents + cond_latent[:, :16] (E:41-44,
    M:135-147, 355-401, 2446-2448) V; paper'da yok.
(b) Bellek dropout: spatial_memory_drop_rate=0.1 → 1..N slot düşer (M:2003-2025); anchor her zaman korunur (M:1491-1496; slot f1k1);
    eğitimde retrieval rastgele, çıkarımda greedy coverage (M:1985-1993; P:376-385); text drop 0.2 V.
(c) Kamera üç yoldan: (1) son history karesinin depth-warp RGB'si VAE'lenip cond_latent kuyruğuna (M:1759-1764, 1826-1831);
    (2) son history karesine göre 6-kanal ışın [o, d] (use_ray_o=True), pixel-shuffle 384 ch, blok başına cam_encoder Linear,
    yalnız Q/K girişine eklenir (inject_kq_only; M:1861-1891, wan2pt1_lyra2.py:100,170,192-196); (3) spatial slot başına warp
    edilmiş kanonik koordinat + normalize depth, buffer_encoder sin-cos → K/Q bias (M:1767-1801; P:411-422) V.
    RoPE standart 3D video RoPE, sabit FramePack slot indeksleri (M:461; wan2pt1_lyra2.py:341,652-658); depth'ten projekte DEĞİL V.
(d) Eğitilebilir: cam_encoder, buffer_encoder, self_attn, clean_patch_embeddings, patch_embedding (E:62); cross-attn/FFN/adaLN donuk;
    LoRA yok V. Init Wan2.1-I2V-14B-480P (40 katman, dim 5120). AdamW lr 3e-5, wd 1e-3, betas (0.9, 0.999); lambda-linear 100
    warmup, cycle 400k, f_max 0.99, f_min 0.4; grad clip 0.1; EMA kapalı; batch 1/GPU; FSDP bf16 V. ≈5.000 iterasyon, 64 B200
    (global batch ≈64) I. Veri DL3DV stride 2, >1.000 kare, ViPE poz/depth (T:30-36); paper depth DA3 der (P:551).
(e) Yazım pozu: eğitimde dataset ViPE pozu/depth; her 8. kare, pencereden >16 kare uzaksa cache'e (M:2383-2399). Çıkarımda bellek
    üretilen karelerden (lyra2_ar_inference.py:1015-1021), depth DA3 (DA3NESTED-GIANT-LARGE-1.1), opsiyonel MoGe ölçek;
    **varsayılan komut edilen poz** w2c_t = cam_w2c[:, f_abs] (:1318), DA3 giriş extrinsics'e hizalı (:353-354);
    opsiyonel --da3_use_predicted_pose: DA3 pozları cam_w2c'yi ezer ve cache'e yazılır (:1129-1269) V. Kare geometrisi ayrı
    saklanır, tek bulut yok (P:364-368).
(f) On-policy yalnız self-aug (1 adım, 1 chunk); tam Self-Forcing yok ("pahalı", P:450-454); context forcing yok; DMD 4-adım
    öğrenci (CFG distill + self-aug korunur, P:537-542), yalnız çıkarım kodu (LoRA; steps [1000,750,500,250], shift 5).
(g) Çıkarımda bellek gürültüsü yok; history temiz, hız sıfır (M:620, 644-648); guidance 5.0, shift 5.0, 35 adım.
Context: framepack f1k1f4s2f1s1f16k4f2k2f1k1_g20, max_segments=13: anchor + 5 spatial slot (4 alt-örnekli, 1 tam) + 20 temporal
slot + 20 yeni latent (80 kare) (E:39-40,54; P:281, 405-407) V.
