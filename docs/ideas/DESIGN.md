# Tasarım: mimari → eğitim → veri (2026-10-09; tarih kısıtı yok)
Mevcut runtime'dan doğrulanan gerçekler (base_restart_20261004/runtime, CLAIM_CHECK): history ve kaynaklar FramePack tarzı
in-context token (rol + paketlenmiş RoPE); scaffold 57 ch → conv projeksiyon + blok başına residual, **30 bloğun hepsine**;
kamera ışınları zero-init RayContextProjection ile; `context_timestep_zero`, `latent_history`, `retrieval_k` bayrakları var;
eğitilebilir küme self-attention + patch embedding (1,14B, 8-bit Adam, 32 GB'a sığıyor). Base'den ≈2.100 update'lik parent bu
kümeyle kamera/scaffold okumasını öğrendi.

## A. Mimari (tek çağrı)
| | Karar | Seçenek / açık nokta |
|---|---|---|
| A1 Omurga | Stock Wan2.2-TI2V-5B, 256×448, 4 latent = 16 kare/çağrı, 30 adım Euler flow, stock VAE stream (anchor'dan başlayan kalıcı cache) | — |
| A2 Token dizilimi | `[anchor (gözlenmiş, t=0)] [yakın native history: son K sampled latent, sıralı zaman pozisyonu, t=0] [bellek kaynakları: ≤2 nedensel native latent + kamera + rol + yaş + destek özelliği, t=0] [hedef: 4 latent, gürültülü, t]` | K (history kapasitesi): 19 slot (FLM) mi, 8 mi? Öneri: 8 — uzun history "history'ye yaslanma"yı büyütüyor (2×2) |
| A3 Kamera kanalı | ReSplat render'ı (scaffold: 48 render latent + validity + confidence + yaş …) her hedef latent için, mevcut conv + 30-blok residual. Plücker ışın yolu altyapı olarak kalır (zero-init), ayrı "kamera rotası" iddiası yok | Scaffold dropout %10 satır (null dal, CFG kalibrasyonu); kamera yanıtı scaffold'la birlikte ölçülür |
| A4 **Güvenilirlik kapısı** (yeni) | Hedef yama başına g∈[0,1]; girdi: hedef hidden state, scaffold özellikleri (validity, confidence, yaş, history–render uyuşmazlığı), kaynak destek özellikleri, flow zamanı. Uygulama: (i) scaffold residual × g, (ii) kaynak token'larına attention logit bias + log g (yalnız V'yi sıfırlamak yetmez: key softmax'ta kalır). **Hedef kamera / ışın yolu kapı dışında.** Küçük 2-katman conv, ~1M param. Başlangıç g=1 (foundation'la bit-eşit), g=0 kapalı-dal eşdeğerlik testi | Kapı yalnız residual'a mı (ucuz) yoksa attention'a da mı? Öneri: ikisi; ablasyon A1 yalnız residual |
| A5 Eğitilebilir küme | **Öneri: tüm self-attention + patch embedding + yeni modüller (parent'ın kümesi; 1,14B)** | Alternatif: Q/K/V/O LoRA r16 (~30M). Daha az VRAM ama parent'ta denenmedi; G1 düşerse bahaneyi ayırt edemeyiz |
| A6 Geometrik adresleme | **Lyra'yı kopyalamak yerine kanıta dayalı adresleme:** ReSplat render'ı her hedef yamada hangi kaynağın hangi yüzeyi gördüğünü zaten biliyor (Gaussian kaynak kimliği + görünürlük + confidence). Bu haritadan hedef-yama × kaynak-token attention bias'ı üretilir (yumuşak, top-2 aday, confidence ağırlıklı) ve kapı g ile çarpılır. Lyra'nın depth-projeksiyon RoPE'u tek, katı correspondence varsayar ve depth yanlışsa (cam) yanlış adresler; bizimki güvenilirlikle ağırlıklı ve karşıt family'lerle 'ne zaman yok sayılır' öğretilir. Ek depth projeksiyonu gerekmez, renderer hesaplıyor | v1'de bias kapalı (yalnız rol+ışın), S1 N2/N3 kontrolü eşit çıkarsa v1.5'te açılır; iddianın parçası olarak 'reliability-weighted geometric addressing' |
| A7 Yazma politikası | DA3 + pipeline pozları + sabit ölçek, zaman seyreltme, conf maskesi. Kalıcı harita yok. **Realized-pose yazımı eğitilmiş bir politika olarak S3 değişkeni:** önceki negatifler yalnız çıkarımda, komut-pozla eğitilmiş modelde ölçüldü; kural gereği eğitilmiş yönü kapatmaz. Kayıtlı satırlarda realized = komut (uyuşmazlık yok); uyuşmazlık S3b kendi-rollout satırlarında (bellek, nedensel gözlemci da3_anchor_rigid ile kestirilen poza yazılır, hedef kamera komut kalır) ve 'kaynak poz uyuşmazlığı' family'sinde öğretilir. Eğitim ve deployment aynı gözlemciyi kullanır | Kol: M-cmd (komut pozla yazım) vs M-real (realized yazım), aynı veri/bütçe; nATE + yapı korunumu karar verir |
| A8 Çıkarım = eğitim | Aynı history temsili, aynı 2 kaynak seçimi (kamera yakınlığı + destek), t=0 koşul, her çağrı ReSplat yeniden, evidence CFG 1,0, aynı init politikası (N2 açıksa scaffold-init, kapsama-duyarlı σ) | N2 kararı seam kollarından sonra |

## B. Eğitim
| | Karar |
|---|---|
| B1 Kayıp | Hedef latentlerde flow matching (stock convention). S3'te + λ_u·L_gate (u(p) işaretli, |u|>marj yamalarda maskeli BCE) + opsiyonel λ_c·L_clean_change (doğru yamalarda S1 tahmininden sapma cezası). Katsayılar dev'de, büyük koşudan önce sabit |
| B2 Aşamalar | **S1 foundation**: temiz + bellek-yok + kamera-fork + revisit satırları, kapı sabit açık, ≈1.500–2.000 update (parent ölçeği). Kapı G1: ilk çağrı LPIPS ≤ 0,13, kamera Sampson ≤ G4-400, netlik 0,9–1,3, temiz devam geç DINO ≥ 0,82, kaynak kullanımı (doğru/yanlış/yok kaynak farkı) ölçülebilir. **S2** kapı pilotu: S1 donuk, yalnız kapı, ≤200 update, u(p) etiketleriyle; kapalı-döngüde fayda yoksa teşhis. **S3** eşleştirilmiş: B1 (sabit görünürlük politikası) vs M (kapı + 5 family), aynı satır listesi/bütçe/seed, ≈1.500 update; S3b: öğrencinin kendi rollout satırları geçerlilik testiyle kademeli eklenir. Ablasyon A1 (utility etiketsiz kapı), A2 (birlikte-bozulma dağılımı, marjinaller eşit). N2 on/off aynı ağırlıkta çıkarım + eşleşik eğitim çifti |
| B3 Satır yapısı | Update başına 10 satır. S1: 5 ilk-çağrı/temiz devam, 3 ileri (history derinliği 1–5), 2 revisit/fork. S3: family örnekleme 30/30/20/10/10 (manifest hepsini tutar, loader kuralı). Kayıp tek çağrıda; çok-çağrı ufku ancak maliyet ölçülünce |
| B4 History bozulması | S1'de yok (temiz). S3'te family'ye göre: one-step yeniden üretim (**üretici = S1 öğrencisi**, on-policy; cap dağılımı kendi deployment LPIPS'ine kalibre) + chroma kayması + yerel smear; S3b'de gerçek rollout |
| B5 Optimizasyon | bf16, 8-bit Adam, parent LR grupları (backbone düşük, yeni modüller yüksek), microbatch 1, grad clip parent, checkpoint her 100 update + atomik; RNG/cursor kaydı |
| B6 Seçim | Dev (8 eski + 4 yeni sahne, seed 7): DINO + KID + netlik bandı + motion + HSV-S + b* + nATE + dikiş; LPIPS rapor. Seed 11 doğrulama. Test split kilitli, materialization'dan önce |
| B7 Değerlendirme | eval_stack + wbench + wbench_pose (warp_v2 düzeltmesiyle), yapı metriği (ortak destek maskesinde köşe/kenar/track korunumu), kapsanan/kapsanmayan ayrımı, katmanlar (cam/iç/dış), own-state 160 ve 320 kare, dış baseline tablosu (Lyra 2, WorldPlay, SANA, Fun-5B, Memorizon, VMem crop) |

## C. Veri üretimi
| | Karar |
|---|---|
| C1 Havuz ve split | D0 havuzu (7.945) − 60 held-out − 14 dev. **Yeni test split: 24 sahne**, katman dengeli (cam/iç/dış), hiç incelenmemiş, materialization'dan önce kilit (sha256). Dev: 8 eski + 4. Train: kalan; önce cache'li 4.156 |
| C2 Grup tanımı | (sahne, anchor adımı, hedef çağrı, history penceresi, ≤2 kaynak); görevler ileri / dönüş (ters kayıt, dinamik maske) / aynı-geçmiş kamera-fork ileri-geri. Her grupta ortak kayıtlı hedef + ortak flow seed |
| C3 Materyalizasyon (GPU) | D1: kayıtlı karelerde DA3 (pipeline pozları, sabit ölçek) + çok-görüş tutarlılık → `support ∈ {verified, none}`; RAFT dinamik maske (kayıt ve ters kayıt); temiz prefix VAE latentleri (stock stream, tek sözleşme, aynı hedef bit-eşit testi); kaynak pencereleri (anchor 1 RGB / hizalı 4 RGB). D2: 5 family'nin bağımsız müdahaleleri (history: one-step regen + chroma + smear; bellek: conf korunarak depth sapması, kaynak poz uyuşmazlığı, yanlış-sahne kaynağı (train split); ikisi; yok). Kanıt dalı önce, history sonra; loader preflight hash'le dal bağımsızlığını doğrular |
| C4 S3b satırları | Öğrenci 1–5 çağrı; geçerlilik: Sampson(komut) ≤ τ_c ve DINO(kayıt) ≥ τ_d; geçenler sampled latent + decoded RGB + DA3 ile satır; hedef = kayıtlı gelecek |
| C5 u(p) etiketleri | Öğretmen = S1; aynı target/noise/3 flow zamanı için kanıt dalı açık/kapalı; işaret kararlı ve |u| > marj yamalar etiket; offline cache |
| C6 Depolama | ≈1,3 MB/satır latent düzeyinde; 20k satır ≈ 26 GB; bütçe 30 GB, disk 71 GB. RGB kopyalanmaz, manifest + dedup cache |
| C7 QA ve sızıntı | D0 bitti; D1 çıktıları `geometry_qa.json`; manifest hash'leri; kasıtlı gelecek-kaynak / hedef-RGB-arşivde / rezerve-sahne sızıntı testleri reddedilmeli (audit_preparation.py zaten var) |

## Kullanıcı kararı bekleyenler
1. A5: tüm self-attention (öneri) mi, LoRA r16 mi?
2. A4: kapı residual + attention (öneri) mi, yalnız residual mı?
3. A2: history kapasitesi 8 (öneri) mi, 19 mu?
4. B2: S2 kapı pilotu yapılsın mı (öneri evet, ucuz kill-test)?
5. B4: S3 history bozulmasını S1 öğrencisi mi (öneri), base Wan mı üretsin?
6. C1: yeni 24 sahnelik test split (öneri evet)?
7. Klasör: production_data/ ve runtime/ `base_restart_20261008` altında tek yer (öneri evet; diğer oturumla paylaşılır)?
