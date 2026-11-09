# Farklı fikirler (2026-10-09): bellek, kamera, mimari, eğitim, veri, kayıp
Her madde: dar problem → mekanizma → veri satırı → kayıp → ucuz ayırt edici test → yakın prior → maliyet. Sıralama: kanıta uyum × maliyet.
Diğer oturumun ARCHITECTURE_AND_RESEARCH_ALTERNATIVES.md'siyle ilişki en sonda.

## Bellek
### B1. Özellik uzayında 3D bellek ("feature splatting" — modelin kendi temsiliyle hatırlamak)
- Problem: bellek RGB → DA3 → ReSplat → RGB render → VAE → token zincirinden geçiyor; her adım kayıplı ve modelin iç temsiline yabancı.
  2×2'de bellek açığın yalnız %32–35'ini kapatıyor; model render'ı "layout" olarak okuyor, görünüş olarak değil.
- Mekanizma: her Gaussian'a (veya yüzey noktasına) o yüzeyi gören karelerin **DiT ara-blok hidden state'lerinden** (ör. blok 10–15, 3072 → 128'e düşük-rank
  projeksiyon) toplanmış bir özellik vektörü bağla. Yeni görüşte RGB yerine **özellik splat'ı** hedef token grid'ine render edilir ve context token
  olarak girer (rol "feature-memory"). Güvenilirlik r Gaussian başına öğrenilen/güncellenen bir öznitelik (uyum sayacı) olur.
- Veri: aynı 5 family; memory-damaged = özellik toplanırken yanlış depth/poz.
- Kayıp: FM + r denetimi; özellik projeksiyonu FM'den öğrenir (ayrı kayıp yok).
- Test: aynı Gaussian seti, RGB-render token vs özellik-render token, S1'de 2 kısa kol; revisit DINO ve kapsanan-piksel LPIPS.
- Prior: Feature-3DGS/GeoVerse (Wan özelliklerini geometri denoiser'ına), Robust Dreamer (latent Gaussian: VAE latenti). DiT'in kendi
  hidden state'ini 3D'ye yazıp geri okuyan dünya modeli yok (yerel taramada).
- Maliyet: yüksek (ReSplat çıktısına özellik kanalı, toplama, 128-ch render). v2 adayı; en güçlü "farklı bellek".

### B2. İki zaman ölçekli bellek: kanıtlanmış kanonik yüzeyler + kare-başına geçici kayıtlar
- Problem: fuse etmek hata biriktiriyor (Lyra −14 CamCtrl, GeoVerse), fuse etmemek bellek otoritesini zayıf bırakıyor; D3b "icat edilen kaydı
  dondur" 3/3 çöküşü önledi ama genel değil.
- Mekanizma: her kare Lyra gibi ayrı kayıt (asla fuse edilmez); **yalnız ≥2 bağımsız görüşte (anchor + parallax'lı üretilmiş kare) fotometrik+
  geometrik uyum geçen yüzeyler** yavaş bir kanonik haritaya terfi eder. Render iki katmanlı: kanonik (yüksek r, düşük σ_init) + geçici (yalnız
  attention adresleme). İcat edilen içerik terfi edene kadar geçici kalır → kendi tekrarlarıyla "doğru" görünemez (ancestry: aynı icadın ardışık
  kopyaları bağımsız onay sayılmaz).
- Veri: dönüş satırları (terfi edenin kayıtla eşleşmesi), icat satırları (terfi etmemeli).
- Kayıp: yok (yazma kuralı); r-init kanonik katmandan.
- Test: eşit kapsama ve kaynak sayısıyla "terfi kuralı" vs "yaş/confidence" yazımı; ileri yürüyüşte terfi hiç olmazsa kural boş demektir.
- Prior: ChronoWorld prune, GEN3C align, SPACE counterfactual eviction; terfi + ancestry birleşimi yok. Diğer oturumun B alternatifiyle aynı aile.
- Maliyet: orta (yazma politikası + iki katmanlı render). v1'e girebilir: r girdisi olarak "terfi durumu" kanalı.

### B3. Yeniden-görme taahhüdü (predictive revisit memory)
- Problem: tutarlılık ancak geri dönünce ölçülüyor; model dönüşte ne göstereceğine dair taahhüt vermiyor.
- Mekanizma: her chunk'ta aynı forward'dan ucuz bir "beklenen görüş" (düşük çözünürlük, 2–3 revisit pozu, DiT başlığından) üretilip bellek olarak
  saklanır; geri dönüşte üretim bu taahhütle tutarlı olmaya itilir (r-init önseli olarak). Dünya modeli kendi gelecekteki gözlemini planlıyor.
- Kayıp: taahhüt ↔ kayıt (eğitimde) ve taahhüt ↔ sonraki üretim (cycle).
- Prior: JEPA tarzı tahmin, R2M revisit; üretken dünya modelinde yok. Maliyet yüksek, kanıt zayıf → follow-up.

## Kamera
### K1. Kamera komutu = render edilmiş **video** (16 kare), tek-latent endpoint render'ı değil
- Problem: model kamerayı render'dan okuyor; bizim render latent başına tek endpoint (4 render / 16 kare). Hareketin kendisi (ara kareler)
  modele hiç gösterilmiyor; dikiş ve çağrı içi hareket hatası buradan besleniyor olabilir.
- Mekanizma: belleği komut edilen yörünge boyunca **16 karede** render et, stock VAE ile 4 latente encode et (anchor'dan devam eden stream
  cache'iyle), context token olarak ver (Fun-Control'ün "control video"su, Alaya/MoVerse prefix'i, CamClone'un referans videosu). Hareket
  modelin kendi dilinde, VAE'nin zamansal sıkıştırmasıyla tutarlı.
- Veri: kamera-fork satırları zorunlu; scaffold-yok satırları ışın yedeği için.
- Test: endpoint-render vs video-render, aynı her şey; nATE_r, çağrı içi RPE, dikiş.
- Prior: standart (Alaya, MoVerse, Fun-Control); bizde yok. Maliyet düşük (render 16 kare ≈ 4× ReSplat render, VAE encode). **v1'e al.**

### K2. Kamera-icra kaybı (commanded-flow consistency)
- Problem: kamera doğruluğu yalnız ViPE ile sonradan ölçülüyor; eğitimde hiçbir kayıp "komut edilen hareketi uygula" demiyor.
- Mekanizma: kapsanan statik piksellerde komut edilen kameralar + scaffold depth'ten **beklenen akış** (runtime'daki pixel_motion zaten
  hesaplıyor) ile üretilen karelerin akışı arasında kayıp. Decode maliyeti: 4 latent → 16 kare 256×448 ucuz; akış için RAFT yerine hafif
  frozen akış probu veya **latent-uzayı akış probu** (küçük conv, kayıtlı çiftlerde önceden eğitilmiş, donuk).
- Kayıp: L_cam = ‖flow(x̂_0) − flow_expected‖ maskeli; yalnız düşük t'de (x̂_0 anlamlıyken), ağırlık küçük.
- Test: L_cam'li/siz eşit kol; nATE + başlangıç-çapalı drift; kalite korkulukları.
- Prior: Geometry Forcing (VGGT hizalama, kamera değil), CamVerse geometri ödülü (RL), Wonder GAN düşük-frekans (DMD). Doğrudan
  denetimli akış-tutarlılık kaybı dünya modelinde yok. Maliyet orta. **v1 ablasyon adayı** (M-real'den önce; realized sapmayı kaynağında azaltır).

### K3. Çift-görüş render (son history kamerası + komut kamerası)
- Model'e korespondans çiftini açıkça ver: aynı belleğin iki render'ı. CamClone "referans video" fikrinin geometrik hali. K1 varsa gereksiz
  olabilir (video render zaten çifti içerir). Follow-up.

## Mimari
### A1. Render token + r bias (lit2 sonucu; REDESIGN §8.1) — v1.
### A2. Erken-blok koşul: scaffold/render token'larını yalnız ilk 8–12 bloğa aç (AC3D), üst bloklar yalnız history/anchor/hedef okur.
  Kamera erken bloklarda karar veriliyor (AC3D %84 ilk %10 denoise); üst bloklara koşul vermek kaliteyi düşürüyor. Ucuz ablasyon.
### A3. Diğer oturumun M1'i (source-only H/E encoder'ları, 4-seçenekli yerel politika, core'a kontrolsüz giriş yok) ile r-init birleşik mi?
  Evet, uyumlu: r-init başlangıcı, 4-seçenek okumayı düzenler; ama o belgenin uyarısı doğru: init'e yazılan yanlış kanıtı okuma kapısı geri alamaz
  → r-init σ_lo tabanı ve yüksek-confidence-yanlış family'si taşıyıcı. Öneri: S1'de init on/off ve politika on/off 2×2 küçük pilot (≤200 update × 4).

## Eğitim ve veri satırları
### E1. Prefix-uzunluğu müfredatı + pencere-dışı (LongTake 0,10/0,20/0,45/0,20/0,05; WorldPlay %80 pencere dışı): history 0'dan 8'e, anchor
  uzaklığı 1–10 çağrı. "Anchor uzak, yalnız bellek bağlıyor" satırları (anchor 10 çağrı geride, kaynak 2) bellek kullanımını **zorlar**.
### E2. Hata geri dönüşüm arşivi (SVI/LongLive/Alaya ErrorBank): S1 öğrencisinin tek-adım sapmaları timestep kovalarında; history_damaged
  ve memory_damaged'ın görünüş yarısı buradan (Gauss gürültüsü değil; Robust Dreamer 0,275 vs 0,651).
### E3. Kendi-rollout satırları için hedef geçerlilik kuralı ayrıştırılsın: (a) kamera/gauge geçerli (Sampson ≤ τ) → kayıtlı gelecek hedef;
  (b) kamera geçerli ama renk/smear bozuk → **bunlar tam da düzeltmek istediğimiz satırlar**, DINO düşük diye atılmaz; (c) başka dünyaya
  kaymış → atılır (FL hatası). Alaya valid/invalid kayıp ayrımı gibi loglanır.
### E4. Kamera-fork + dönüş + "stop-turn" görevleri: aynı prefix, farklı gerçek gelecek; ters kayıt yalnız statik destekli bölgelerde.
### E5. Önce sabit r* (teacher-forced) ile init, sonra tahmin r; S3 sonu tamamen tahmin r — eğitim/çıkarım parity.

## Kayıplar
### L1. r denetimi (REDESIGN; yumuşak hedef). v1.
### L2. Görülmüş-yüzey tutarlılık kaybı: kapsanan ve **kanonik/terfi etmiş** (B2) yamalarda ‖x̂_0 − z_render‖·r*; icat/geçici yamalarda yok.
  "Kazanılmış otorite" kaybı; Alaya'nın invalid ×2'sinin tersi. Ucuz; v1 ablasyon.
### L3. Bölümlenmiş FM: kapsanan yamalar ağırlık w_c, kapsanmayan w_u (M3 eski plan). r ile çakışır; L2 varsa gereksiz.
### L4. Renk çıpası kaybı: çağrının kapsanan bölgesinin latent-uzayı kroma istatistiği (doğrusal probe) ↔ kaynak/anchor istatistiği.
  Ölçülen sürüklenme (HSV-S 69→83, b* 6→10) için hedefli, düşük riskli; ışık değişimi olan sahnelerde maske. v1 ablasyon.
### L5. Kamera-icra kaybı (K2). v1 ablasyon.
Kural: v1 ana koşuda FM + L1; L2/L4/L5 tek tek ablasyon (her biri ≤ 1 gün); hepsi aynı anda yığılmaz.

## Öncelik
| Fikir | v1 | Gerekçe |
|---|---|---|
| A1 render token, K1 video render | evet | standart ama bizde yok; kamera/dikiş/bellek okumasını aynı anda etkiler; ucuz |
| L1 r + r-init | evet | tek özgün mekanizma |
| B2 terfi katmanı (r girdisi olarak) | evet (yazma kuralı) | D3b kanıtı; r'ye "kazanılmış" sinyal |
| E1–E5 | evet | veri tarifi, literatürde en güçlü kaldıraç |
| L2, L4, L5, A2 | ablasyon | ucuz, hedefli |
| M-real | S3 ikinci kol | literatürde boş |
| B1 özellik belleği | v2 | en farklı bellek; mühendislik ağır |
| B3, K3 | follow-up | kanıt zayıf |

## Diğer oturumun belgesiyle ilişki
Uyum: stock base, native kayıtlar, ≤2 kaynak, 5 family, bağımsız müdahale, kamera yolu kapı dışı, S1→S2→S3, FAR/SPACE/WorldTrace yakın prior.
Fark: (1) orada scaffold-init ana yolda yok, burada r-init ana mekanizma → A3'teki 2×2 pilot karar verir; (2) orada scaffold residual,
burada render **token** (CamClone/AC3D kanıtı); (3) orada kamera ayrı ışın+poz encoder, burada kamera = render video + ışın yedeği;
(4) burada K2 kamera-icra kaybı ve B2 terfi katmanı yeni. Birleştirme makul: M1 okuma politikası + r-init + render token tek foundation'da.
