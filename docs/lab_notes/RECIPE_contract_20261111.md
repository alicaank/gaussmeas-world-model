# Contract recipe (decided 11 Nov 2026, evening) - data preparation + training settings

Status: data tables and the window index are built (CPU). The render cache v4 and every training below are NOT run yet.

## Data (built)
| item | file | content |
|---|---|---|
| window table | data_contract/windows.jsonl | all 67,547 candidate windows of the 891 train streams: per-call translation / rotation, anchor coverage, bank size (stock rule, contract rule), true revisits, render PSNR where cached |
| window index v4 | data_contract/index_v4.json | 4,423 windows, 5 per stream: 1 first-call + 4 continuation, weights speed (x30 in [.05,.15), x8 in [.15,.25), x1, x.3 above .5) x history (start >= 14) x revisit (x2) |
| bank rule | memory_contract.py | true revisits (near pose that the camera left in between, RELATIVE test: works at any speed / excursion), then first-seen owners; ONE function for cache builder, trainer (--bank contract) and roller (--bank contract) |
| validity | frontier.support_density | source pixels landing per target cell >= .15 of nominal; same function in the cache (meta_density_f16) and at inference (--frontier density) |

Corpus facts (windows.jsonl): per-call translation p10 / p50 / p90 = .128 / .353 / .790 gauge units, rotation 5.3 / 20 / 57 deg; 13% of the windows move < .15 units
(710 streams have at least one); a TRUE revisit exists for 27% of the windows with >= 13 latents of history (724 streams); contract bank non-empty in 96% of those
(stock rule 64%).

Stream-balanced mix the sampler will draw (old index -> index v4): continuation rows in [.05,.15) units 11% -> 33%, in [.15,.25) 18% -> 29%, >= .5 29% -> 15%;
rows with >= 13 latents of history 62% -> 76% (96% of the continuation rows); true revisit 16% -> 25%; contract bank filled 65% -> 75% of all rows (94% of
continuation rows). Slowest-first selection would reach 48% in [.05,.15): the photo set cannot go lower without video frames.

## Training settings
| setting | value | reason |
|---|---|---|
| loss | flow-matching velocity MSE on the 8 target latents, existing u-weighting; no extra term, no masking of the loss | invalid render cells are supervised with the real content like any other cell: that IS the frontier signal |
| time sampling | u_dist shift3 | unchanged from every stage so far |
| optimizer | AdamW, wd 1e-3, grad clip 1.0, bf16 | unchanged |
| lr backbone | 2e-5 constant | unchanged; every recipe so far used it |
| lr side modules (camera / metadata / role / PRoPE) | 1e-4 in A and B, 2e-5 from C on | they are zero-initialised and need the rate at first; later they moved 50-96% of their norm per continuation |
| warm-up | 50 steps at every fresh optimizer start | plan rule; the trainer cannot carry Adam state across stages with different flags |
| batch | accum 16 x 1 row (2 target chunks each) = 16 rows / 32 chunks per update (was 8; changed 23:40) | VRAM 24.5 GB with 4 bank slots; with Adam more rows, not more accumulation, buy SNR |
| bank | contract rule, max 4, deterministic 20% bank dropout on continuation windows (training only) | same function as inference; 6 slots do not fit with 2 chunks |
| render | cache v4 (8-view weights, 512x960), render_p .90 in B / C (10% render-free rows), metadata = density | corrected geometry + inference mask; render is always present at inference, the density mask teaches the partly-invalid case, 10% keeps the camera carriers alive (no clean experiment: the line was trained at .75; value logged per run) |
| window filter | STREAM-level in B / C: drop streams where >= half of the cached windows render below 20 dB (v3 cache: 119 of 888 streams hold 71% of the sub-20 windows); no per-window threshold | a per-window PSNR threshold is biased by motion: sub-20 share 10% below .15 units, 13% at .25-.5, 18% at .5-1, 36% above 1 unit (rotation: 11% -> 22%); recompute the stream flags on cache v4 |
| text dropout | 0.15 | makes guidance usable; costs nothing |
| corruption | none in A / B / C; optional stage D only (below) | three 250-update corruption recipes all broke self-consistency; 100-update doses did not |
| updates | A 350, B 250, C <= 250 with a checkpoint every 50 | C: stop before regeneration rises (recipes were clean at 100, broken at 250) |
| seeds | pilot 20261023, A 20261112, B 20261113, C 20261114; every final number on 2 inference seeds | runs diverge after ~160 frames from GPU nondeterminism alone |
| EMA | none | no evidence either way; one variable less |

Pool rows (stage C) stay undecided until the controls (gmc3, then the no-pool control) answer; if they enter: depth-1 items only, pose + content threshold
(latent cosine >= .48), regenerated with the stage's own model, at most two uses per item.

## Order
1 render cache v4 (GPU, ~1.75 h)  2 sampler dry run + distribution report next to the inference plans (CPU)  3 contract pilot from plain-350 (250 updates, no pool,
no corruption, text dropout .15) against B'  4 untrained control: plain-350 and the 50-step Memorizon teacher at test speed  5 main run from Memorizon: A -> B -> C.
Gates: A command execution + bank use; B first calls, vertical scale, frontier quality, out-and-back (inlier count + shift); C quality / fidelity / self-consistency.

## Distribution report (data_contract/distribution_report.json; run BEFORE training, already caught one mismatch)
First version of the revisit test used absolute "left" thresholds (.5 units / 30 deg): 27% of training windows had a revisit, 0% of the calls on the test's
out-and-back plans (their whole excursion is smaller). Made relative; now: training continuation rows 32% true revisit, bank filled 94% (mean 3.6);
inference forward plans 0% revisit / bank filled 67-83%; out-and-back plans 52-60% revisit (every way-back call) / bank filled 60-80%.
Per-call translation: training p10 / p25 / p50 = .09 / .13 / .21 units; test plans stride 1 = .025-.033, stride 2 = .05-.066, stride 4 = .11-.13, stride 8 = .21-.27.
-> inside the training range only from stride 4 on (p10-p25); the motion-targeted step should aim at >= .13 units per call, not .1.
Still to add after the cache build: density-mask ratio and render PSNR per row, next to the inference logs (showroom stride 4: 6.8% dropped).

## Stage D (optional, added 11 Nov 22:10): corruption for chains longer than the motion-targeted step covers
Run ONLY if the final clean model melts on the long plan: eval_gm/inputs_long (full 1009-pose plans of the two test scenes; at stride 4 = 15 calls, 960 source
frames, 1.9-2.1 gauge units of path, .11-.13 units per call). GM' reference on that plan is queued (gmp3_long_stride4).
Rules: (1) separate BRANCH - the clean checkpoint stays, corruption trains on a copy; (2) SMALL dose - checkpoints every 25 updates, at most 100, stop at the
checkpoint before regeneration rises (old line: 100 updates fixed half the collapse with regen .98 / 1.05; the defects appeared between 100 and 250);
(3) judged on THREE measures - quality (sharpness ratio, DINO), fidelity (LPIPS vs recording on anchor-visible cells), self-consistency (regen_x, out-and-back
inliers + shift) - never on DINO alone (GM-B' .787 was a different playground). Candidates, same short dose each: v2 (amplitude-preserving noise + moment
matching: best 100-update result, .792) and v2 + selective trust (anchor-seen cells clean; anchor_seen.py, --selective 1). Its shape depends on tonight's
control: if regeneration ~2 appears with corruption OFF, the continuation / pool is the cause and must be fixed first.

## Updates 11 Nov 22:30 (approved order + checks)
ORDER: stage A tonight directly from the Memorizon-50 teacher (train_a4.py; no render, so no cache needed) -> render cache v4 -> morning: A gate -> stage B on the
new A (fallback: B on plain-350 if the A gate fails; the plain-350 pilot is only a reserve). A gate: command execution ratio at test speed (stride 4), loss rises
when the bank is swapped with another scene's, first calls not worse than plain-350.
CHECK 1 bank fill: training continuation rows 94% without dropout; inference with the same function 67-83% of the calls (forward), 60-80% (out-and-back), empty in
the first calls -> deterministic 20% bank dropout (hash of the command poses; same in the cache builder). Stage A dry run (2800 rows, seed 20261112): bank filled
75.5% of continuation rows (60.5% of all rows), true revisit 33%, first call 19.9%, >= 13 latents of history 77.5%, per-call translation p10 / p25 / p50
.089 / .13 / .21, 34% in [.05, .15), no failures.
CHECK 2 PSNR filter bias: see the window-filter row (stream-level filter instead).
CHECK 3 step target: a RANGE, .13-.2 gauge units per call (stride 4-8 on the two test scenes), upper end to be re-chosen by frontier quality after the masked training.

## 11 Nov 23:40: stage A batch 8 -> 16 (user: "larger batch if it fits in memory")
A true micro-batch of 2 rows does not fit (one packed row peaks at 24 GB of the 32.6 GB card, and rows have different lengths), so the effective batch was
doubled with accumulation 8 -> 16 (same as the GM' setting). Same lr / warm-up / 350 updates -> 5600 rows instead of 2800, ~46 s per update, ~4.5 h.
Restarted in runs/a4_s2; runs/a4_s1 abandoned at update 20 (no checkpoint, ABANDONED file). Stage B / C keep accumulation 16.
gmc3 control (GM' + 250 updates, corruption OFF): regen_x 2.63 / 2.02 / 2.31 (GM' 1.04 / 1.14 / 1.19), showroom fidelity last third .42 vs .24 -> the
continuation itself (pool rows / op_arm recipe), not the corruption, re-draws the scene. Stage B (no pool, no corruption) gate measures the same thing.

## 12 Nov 04:40: stage A finished (runs/a4_s2, 350 updates, accumulation 16)
Loss (50-update means) .329 / .271 / .259 / .303 / .244 / .285 / .270; no nonfinite. One gradient spike: update 100, pre-clip norm 13018 (clipped to 1.0;
other logged norms .10-.16, 1.5 at 310, 3.8 at 350). That update held row c86ddde4 at 2.66 units per call (fm .70 / .92), a stream whose median speed is 2.3
units per call. 22 streams outside bad_streams_v3 have a median above 1.0 units per call (103 of 3843 windows, 2.7%) -> data_contract/fast_streams_v4.json,
EXCLUDED from stage B / C together with bad_streams_v3 (gauge scale unreliable there; the test plans never exceed .27). A gate: check 50 vs 100 vs 150 too.

## 12 Nov 06:15: render cache v4 built (4422 / 4422 windows, 1.8 h; contract bank, density metadata at support >= .15)
First-target PSNR p5 / p50 / p95 = 17.9 / 24.6 / 32.0 dB, 13% below 20 dB. Stream flags re-run with the v3 rule on the v4 cache: 86 streams (68 already in
bad_streams_v3, 18 new -> data_contract/flag_streams_v4.json; 51 of the v3 list no longer meet the rule but stay excluded).
Stage B / C exclusion = bad_streams_v3 + fast_streams_v4 + flag_streams_v4. On the remaining windows (3660):
  PSNR p5 / p50 / p95 19.9 / 25.4 / 32.4 dB, 5.7% below 20 dB (first calls 0.7%, continuations 6.9%)
  density-dropped fraction p25 / p50 / p75 / p95 .027 / .067 / .135 / .28 (first calls median .03, continuations .08); valid cells median .91
  -> inference showroom stride 4 dropped 6.8%: inside the training range (median). Remaining sub-20 dB continuation rows are kept (render_p .9 + conf
     metadata = the model must learn when not to trust the render).

## 12 Nov 06:45: stage-A gate launched (gate_a4.py -> gate_a4_camera.py)
Checkpoints a4_s2 50 / 100 / 150 / 250 / 350 + plain-350. (1) probe_bank_swap, 40 dev windows -> eval_gm/diag/bank_swap_a4.json; (2) stride 4, 7 calls, no
render, contract bank, seed 7, both dev scenes -> sota_eval/runs_fast/<ck>_s4; (3) ViPE (SOTA scorer, horizon 112) -> eval_gm/fast/camera_a4, execution vs
the strided plan (inference plan == ViPE batch plan, checked: max deviation 0). queue_v3 (old-line diagnostics) paused with SIGSTOP after its running
job and resumed automatically when the gate exits. train_gmp.py now takes --exclude (comma-separated lists, 8-char prefix match).

## 12 Nov 07:30: stage-A gate result -> a4_350 is the B start
Bank (40 dev windows, n 80): use = loss(none) - loss(own): 50 -.0002, 100 +.0032, 150 +.0087, 250 +.0052, 350 +.0053 (+-.0015); plain-350 +.0014 (n.s.);
harm (swap) ~ use at every checkpoint (.005-.006) -> a wrong bank costs what no bank costs (ignored, not copied). Own loss 350 .302 vs plain-350 .3075.
Execution at stride 4 (ViPE vs strided plan, frame 112; exec / lat / yaw ratio): showroom a4_350 .98 / .09 / .90, a4_250 .95 / .10 / .97, plain-350 .96 / .19 / 1.24,
a4_150 1.36 / .15 / 1.73; playground a4_350 1.35 / .11 / .92, a4_250 1.52 / .10 / 1.12, plain-350 1.25 / .35 / 1.39, a4_150 1.14 / .89 / 2.49 (a4_50 playground:
ViPE registration rejected). Translation over-execution on the playground is shared by plain-350 (and GM 1.09 at stride 1): partly ViPE scale there.
Visual (frames 0 / 32 / 64 / 96 / 112): a4_150 melts on both scenes (showroom crashes into a shelf, playground dissolves) - the update-100 spike region;
a4_250 / a4_350 keep both scenes; a4_350 best playground (structure kept, but the slide turns into ribbon-like streamers from frame 64), showroom keeps the
table / shelves (a dark object enters the left edge at 96-112); plain-350 playground degrades to a wire-frame structure by 96-112.
Decision: B starts from a4_350.

## 12 Nov 08:30: stage B launched as TWO paired arms (queue_b4.py), + corrections
Exclusions: stage A excluded ONLY the 119 streams of bad_streams_v3 (772 streams trained). The other 38 (22 fast_streams_v4 + 18 flag_streams_v4, 2 shared) were
found after / during A and WERE in A's data (5% of its streams). B / C exclude all 157 (734 streams, 3660 cached windows).
Gradient spikes are NOT one event at update 100 (correction of the 04:40 note): the per-update log of a4_s2 has 35 of 350 updates with a pre-clip norm above 1
and 12 above 100 (2, 5, 15, 64, 81, 84, 94 [39546], 100 [13018], 118, 126, 261 [369884], 306); median .16. Every earlier run of this trainer has them: plain-350
16 of 350 above 100 (max 3.7e6), B' 5 of 250, GM' 1, gmc3 7. So they are a property of the trainer, not of the c86ddde4 row; all are clipped to norm 1 (one
row then sets the direction of that update). Cause unknown (the printed log only shows every 10th update, which hid them). a4_150 follows the densest
cluster (94 / 100 / 118 / 126 / 134) - consistent with, not proof of, the spike explanation. From B on: --row_grad 1 logs the accumulated norm before each
row ('gacc'), which names the row; checkpoints after a spike cluster are judged with their neighbours.
B arms (same A checkpoint a4_s2/350, same rows / seed 20261113 / settings: contract bank, cache v4 + density metadata, render_p .9, text dropout .15,
accumulation 16, warm-up 50, 2e-5 / 1e-4, 250 updates, no pool / corruption):
  token    runs/b4_token_s1    4 render slots in the sequence (the line so far)
  channel  runs/b4_channel_s1  channel_render.py (a collaborator: aligned prior -> concat; retrieved frames stay tokens): the forward removes the render slots and adds
           Conv3d(50 -> dim), zero init, of (render latent x valid, valid, conf) on the SAME chunk's target tokens, next to the patch embedding. Starts exactly as
           a4_350 without render; tests_channel_render.py passes (zero-init equality, only chunk-0 targets change, render-free rows unchanged).
Dry run (token): 4000 rows, 3627 with render (90.7%), 0 missing windows. Comparison: loss with / without render, first-call fidelity + rotation, edges under
the density mask, oracle-render override of a drifted context, peak memory / seconds per update.

## 12 Nov 09:20: B restarted with outlier-row rejection (both arms; runs/b4_token_s2, runs/b4_channel_s2; s1 abandoned at update 16, no spike had occurred)
row_reject.py (--reject 50): after each row's backward the row's gradient norm is estimated from the accumulated norms, r = sqrt(after^2 - before^2); a row
with r > 50 x the running median of accepted rows (last 200) is dropped: the accumulated gradient is cleared and the accepted rows of that update are replayed
with their saved generator states, so the run's random stream (u, noise, text dropout) is identical with or without rejections -> the arms stay paired.
tests_row_reject.py: rejected update == the same update without that row; generator state unchanged. First 6 updates of s2 reproduce s1's losses exactly
(.3249 / .2874 / .2949); row norm estimates .005-.17, median .065 -> threshold ~3.3. Logged per row: grow, rejected; end of log: rejected rows / total.
Spike-cause checklist once rows are named: same streams (pose data, near-singular cameras in PRoPE) / same noise levels (time weight, ends) / random (bf16).
Note for the arm comparison: the channel arm's first losses equal a4_350 WITHOUT render on those rows (zero-init conv, no render slots); the token arm sees
render tokens from update 1 (as untyped context: its metadata / role inputs start at zero), so the two arms need not have the same loss at update 1.

## 12 Nov 10:40: cause of the gradient spikes = windows far from the anchor (token arm at update 133 of 250)
Token arm so far: 5 rows rejected of 2128 (updates 60, 94, 113, 116, 118; row norm estimates 392 / 1.34 / 196,419 / 1.17 / 192; median .024, threshold 1.2);
largest update norm since then .64 (before: 1e2-1e5). Loss 25-means .285 / .256 / .264 / .257 / .267 (stage A last 100: .270). No NaN, 65 s/update, peak 24.6 GB.
Cause checklist:
- same streams: YES. 9c4b29b9 start 92 = B's largest (196,419) AND stage A's largest (update 261, 369,884); bb9c1e10 start 75 = B update 60 (392) and A update 94
  (39,546); 9c4b29b9 is in 4 of A's 18 spike updates (7 appearances in A).
- noise level: no. Rejected rows have min u .009-.53; fm ~1 at u < .05 is normal for every row and is not a spike.
- random half-precision overflow: no (the same windows repeat across runs).
What the streams share: distance of the window's cameras from the anchor in model units ("far" = max |t| over recent + target slots, anchor = slot 0 of every row).
  B rows: all 6 rows with norm > 1 have far 20-39 (population p50 1.8, p95 7.7, p99 13.7); AUC .999. Row norm median by far: < 7.7: .022-.032, 7.7-10.8: .045,
  > 10.8: .133 (44 rows, 6 above 1). Stage A, update level: AUC .916; 16 of 18 spike updates contain a row with far > 10 (the other two: 9.0 and 9.8).
Mechanism: PRoPE (prope.py) multiplies q by P^T and k, v by P^-1; the anchor slot is in every row, so the relative translation anchor <-> target enters the
  logits raw. CPU check (scratchpad prope_numerics.py, unit-normal q / k): logit rms 1.5 (max 13) at far 1-3, rms 5-11 (max 80-153) at far 19-39. Re-gauging the
  cameras to the window does NOT help (logits identical in float64, bf16 error .08 -> .035 rms, both small): it is the size of the relative translation, not precision.
Test rollouts are not in this regime: dev plans reach 2.0 (showroom) / 4.7 (playground) model units at frame 1009.
Decision: stage B continues unchanged (rejection keeps these rows out, same rule in both arms). Stage C: exclude data_contract/far_windows_v4.json
  (far > 10: 104 of 3660 windows, 2.8%, 47 streams) on top of the 157 streams; row rejection stays on. A model-side bound on the PRoPE translation
  would change the function of the trained branch -> not now.

## 12 Nov 12:15: anchor distance at test time; near-threshold rejections marked (critique 12 Nov 11:50)
Commanded cameras' distance from the anchor, model units (scratchpad dev_far.py -> eval_gm/diag/dev_far.json; CPU):
- dev table (closed_loop_864, recorded poses, stride 1, 20 calls), max over the 320 frames: c3cfc2ff 5.2, 0d26e9b3 3.6, e83f25bf 1.3, 284b0dbc 6.6,
  5f800e62 (classroom) 1.6, **e7523f6f 15.7** (> 8 from frame 85, > 10 from frame 104 = call 7, peak calls 10-12, back to 4.1 at call 20; path 1.47 units/call, mpu 15.5).
  -> the classroom failure is NOT this regime. e7523f6f is the one dev scene inside it, and it is the one scene whose image collapses there in every arm:
  sharpness by 4-call block GM 630 / 388 / 66 / 7 / 54, GM-B 693 / 548 / 36 / 22 / 186, GM+mask 721 / 579 / 102 / 40 / 230; DINO GM .81 / .53 / .30 / .05 / .09,
  partial recovery in the last block when the camera comes back. Correlation on one scene, confounded (fastest scene, largest scale, VGGT/label disagreement .32,
  40-95% frontier). Causal test (inference only, after the B gate): same scene with the anchor slot dropped from the context once it is beyond 8 units.
- SOTA batch plans (1009 frames, stride 1), max: 02267acf 2.0, 0238de14 4.7, 334ff324 3.6, 61b27b61 6.0, a126efe8 5.1 -> none in the regime (plan geometry only; no cohort run touched).
- other development streams above 10: 48bbc292 10.5, 0a485338 12.4, bdce111e 11.7 (not in the dev table).
Rejections so far (reject_report.py, token arm at update 221): 9 of 3536 rows; 4 CLEAR (>= 3 x threshold: 392, 196,419, 192, 130) and 5 NEAR (1.2-2.9 x threshold).
  All 9 are far windows (19-39 units), as are the 4 accepted rows in the half-threshold band (10-21 units) -> the rule is not cutting ordinary hard rows.
  reject_report.py marks CLEAR / NEAR per run (runs/<run>/reject_report.json); watch the NEAR count with far < 10.
Later (not this week): make the anchor conditional -- out of the context (or replaced by a near reference frame) when view overlap is zero and distance is above
  a threshold, same rule in training and inference. Re-gauging cannot help (the encoding is already pairwise-relative).

## 12 Nov 12:05: decisions (critique)
- e7523f6f causal test (anchor dropped beyond 8 units): NOT queued. Low priority, after the B gate, only if the fast dev scenes enter the paper as a second
  table: the cohort is not in the regime, and the test is not clean (no training row is anchor-free -> a negative result would not refute the mechanism).
  Otherwise the conditional anchor stays a note for the next training recipe.
- Stage C expectation: with far_windows_v4.json excluded the rejection count should fall to ~0; if it does not, there is a second cause.
- Channel arm starts right after the token arm (same queue, no gate in between); both gates are read together. First updates of the channel arm: loss vs the
  token arm's first steps, peak memory, seconds per update.

## 12 Nov 12:25: long no-render rollouts inserted between the B arms (user: "20-30 cagrida calistir")
queue_long350.py: plain-350 and a4_350, stride 1, 30 calls (480 frames), no render, contract bank, seed 7, both dev scenes -> sota_eval/runs_fast/{plain350,a4_350}_long.
Purpose: the user's impression that plain-350 keeps the background while the GM lines change it; the matched 7-call stride-4 comparison did not show it
(plain-350 loses the apartment block by frames 96-112; seam ratio playground / showroom: plain 1.10 / 1.28, a4_350 1.07 / 1.17, GM .97 / 1.03, GM' 1.00 / 1.15),
but the old GM videos were 20-30 calls at stride 1. queue_b4 (2339222) is SIGSTOPped at token update 240 so the channel arm starts only after these 4 runs
(SIGCONT by queue_long350 on exit); channel arm delayed by ~40 min.

## 12 Nov 12:45: token arm finished; visuals before the channel arm (user)
b4_token_s2: 250 updates, 4.6 h, no NaN, 65 s/update, peak 24.6 GB. Loss 50-means .271 / .261 / .255 / .251 / .254 (stage A last 100: .270). Update norm median .13,
max 1.22. Rejected 12 of 4000 rows (7 CLEAR, 5 NEAR), all far windows (19-39 units), 6 streams; no rejection below 10 units. Unpaired row means, second half:
render rows .269, render-free rows .305 (first half .276 / .287).
queue_between.py replaces queue_long350.py (killed, child left to finish): 1 token@250 stride 4, 7 calls, contract inference (--resplat v8 --frontier density
--bank contract) 2 plain-350 / a4_350 long 3 token@250 long (30 calls). queue_b4 stays stopped until it exits; channel arm starts after (~14:00).

## 12 Nov 13:05: token arm first visuals + what runs before the channel arm (critique 13:00)
Token@250, stride 4, 7 calls, contract inference, seed 7 (eval_gm/sheets/token_*.jpg, compare6_*.mp4): showroom keeps table / wall, the marble-like column at the
right edge (GM, GM', GM' + inference mask) is gone -> first sign that the mask IN TRAINING fixes the first generation at the edge; the dark cylinder entering left
at 96-112 is inherited from stage A (a4_350 has it). Playground: structure kept to the end, but from frame 64 white wire-frame objects + small coloured parts
appear on the ground at the left; GM' leaves that area empty. Seam ratio playground / showroom: token 1.04 / 1.00, GM' 1.00 / 1.15, GM' + mask 1.01 / 1.09.
The GM' comparison is not fair (GM' had the on-policy stage) -> reference = B' (bp3_s1@250), run now in the same layout.
Hypothesis for the playground objects: the bank is filled now (empty in every earlier model) and a recalled railing / swing is placed at the wrong position.
queue_between2.py (replaces queue_between.py), before the channel arm: 1 token playground bank off / mask off ('--bank none' added to run_arm_ckpt.py +
source_policy.install_bank) 2 B' stride 4 both scenes 3 plain-350 / a4_350 long 4 token long (-> regen_x, LPIPS calls 1-3, vertical ratio on CPU during the
channel arm) 5 token out-and-back. Channel arm starts after (~15:00). NOT before the channel arm: the paired render / no-render loss (probe_render_loss.py only
knows cache v3 + stock; a v4 + density + channel-aware probe is written during the channel arm and run at the joint gate; unpaired row means so far .269 vs .305).

## 12 Nov 13:30: playground stray objects = own first generation carried by the render, not the bank, not the mask
Token@250, playground, stride 4, 7 calls (eval_gm/sheets/diag_0238de14.jpg, diag_zoom_0238de14.jpg, diag_render_0238de14.jpg, diag6_*.mp4):
- bank OFF ('--bank none'): the white wire-frame objects + red object are still there from frame 64, and the slide breaks up MORE at 96-112 -> bank hypothesis
  refuted; the bank helps the slide. Seam 1.06 (bank on 1.04).
- mask OFF: wire-frames fainter but present at 64-80, a red car-like object from frame 80, slide translucent / shredded at 96-112 -> not the mask; mask on is better. Seam 1.03.
- render dump of the standard run: from call 4 on the RENDER contains the white frames and the red object at that place, valid = 1 and confidence high
  (render valid .90-.97, conf .83-.89). They are the model's own earlier frames (thin distant equipment behind the bench, first drawn at frames 32-56 as white
  wire-frames) re-rendered at closer range; the model then follows its render. Self-consistent (generated content stays), the defect is the FIRST generation.
- B' (bp3_s1@250, fair reference: render, no on-policy): same class of objects at the same place (white frames left, red objects at the hedge, chair-like frames
  at the slide at 96-112). GM' draws a faint frame at 64-80 and then drops it. So this is the render-following behaviour without the on-policy stage, not a
  property of the contract recipe. B' keeps the slide's dark-green shape better; the token arm turns it teal and larger (as GM' does).
Showroom, B' vs token: B' has a dark panel entering left and a smeared bright object at the right edge at 96-112; token has the dark cylinder left (from stage A)
and a plant-like object right. Seam ratio playground / showroom: B' .95 / 1.09, token 1.04 / 1.00, GM' 1.00 / 1.15.

## 12 Nov 13:45: stage C decision (critique 13:35) + long no-render rollouts
Stage C HAS pool rows. Evidence: B' draws the same stray objects as the token arm, GM' (= + on-policy stage) draws a faint frame and drops it; the only
difference is the on-policy stage (see the render built from its own flawed output, produce the real content). Dose matters: 250 updates cleaned and kept
consistency (GM' regen 1.04-1.19), 500 redrew the scene. Stage C:
- pool generated with the WINNING B arm in the inference layout (motion-targeted step, density mask, contract bank); depth-1 items only; pose + content threshold
- 100-150 updates, checkpoint every 25-50; far_windows_v4.json excluded, row rejection on (expected rejections ~0)
- stop rule: wire-frames gone AND cross-boundary regeneration <= 1.3; if regeneration starts to rise take the previous checkpoint
Anchor fidelity vs consistency: the stray objects are distant thin content of the anchor enlarged wrongly on approach (consistent afterwards; fidelity to the
anchor is what breaks). A channel arm that follows the render more tightly will carry them too; not counted against it (same panel, same numbers for both arms).
Next B (contract change, not this round): confidence channel = support density itself (source pixels per cell, continuous) instead of alpha, so that an
approached surface with few source pixels reads as weak evidence and the model generates detail instead of copying.

Long no-render rollouts (stride 1, 30 calls, seed 7; eval_gm/sheets/long_*.jpg, long_*.mp4), frame change per frame within calls (recording = reference):
  playground: recording 4.6, plain-350 11.0, a4_350 3.3, B' 4.4, GM' 6.0;  showroom: recording 2.4, plain-350 5.8, a4_350 1.4, B' 2.6, GM' 2.4
- a4_350 hardly moves the camera at normal speed: frames 0-480 keep almost the anchor view on both scenes (the scene "does not change" because the command is
  not executed). At stride 4 the gate measured .98 / 1.35 execution -> without render the model executes large steps but not small ones. To confirm with ViPE.
- plain-350 over-moves and collapses: playground into ribbons from frame 240, showroom drifts into an invented brown wall from frame 320.
- B' / GM' (render) follow the command; GM' playground degrades at 400-480.
Seam ratio (boundary / within) playground / showroom: plain-350 1.08 / 1.21, a4_350 1.31 / 1.28 (small denominators), B' 1.18 / 1.49 (20 calls), GM' 1.20 / 1.64.
fast_metrics (321 frames, showroom / playground), change of generated cells relative to the recording, inside a call / across a boundary:
  plain-350 1.67 / 1.52 | 1.91 / 1.70;  a4_350 .29 / .68 | .32 / .58 (p48 .98 / .97: nearly static);  B' 1.07 / .94 | 1.10 / .98;  GM' .91 / 1.08 | 1.04 / 1.14;
  GM .80 / 1.07 | .90 / 1.16;  gmc3 1.17 / 1.17 | 2.63 / 2.02. plain-350 redraws everywhere (inside calls too, anchor-visible cells 2.5 / 1.7); a4_350 under-moves.

## 12 Nov 14:00: stage C dose in EXPOSURES per pool item; regeneration is two-sided (critique 13:50)
Dose reading: B' (no pool, render, 250 updates, fresh optimizer + gradient kicks) ~1.0; GM' (pool, 250 updates) 1.04-1.14; gmc3 (the same pool, 250 more) 2.63 / 2.02
with the in-call value still 1.17 -> the damage is the SECOND pass over the same pool, at the boundary (bases differ, so not a clean split).
GM' saw each pool item ~1.7 times (250 updates x 8 rows x ~.5 pool share / 600 items); gmc3 added 1.7 -> 3.3.
Rule: exposures per pool item = updates x 16 rows x pool share / pool items <= 1.5.
  100 updates x 16 x .5 / 600 = 1.33 (chosen default: checkpoints at 25 / 50 / 75 / 100 = .33 / .67 / 1.0 / 1.33 exposures)
  150 updates only with >= 800 pool items at share .5 (1.5), or share .375 with 600 items (1.5). The run's flag file records items, share and exposures.
Regeneration ratio is reported as distance from 1 (|ratio - 1|): below 1 is a defect too (stage A .3-.7 = standstill; GM's .83 earlier = frozen edge).
Token arm: regeneration next to the ViPE execution ratio at normal speed (gate_b4_camera.py: exec / lat / yaw ratio + per-call step ratio), to tell
"follows the slow command with the render" from "inherits stage A's standstill". queue_between3.py adds the ViPE runs before the channel arm.

## 12 Nov 14:10: token arm numbers (normal speed, contract inference, seed 7; showroom / playground)
fast_metrics, 20 calls (30 calls in brackets); regeneration as ratio to the recording, target 1:
| model | regen in-call | regen across boundary | snap1 | DINO last third | sharp | lum_end | p48 |
| B token 250 | .94 / .97 (.85 / .97) | 1.05 / 1.00 (.95 / 1.00) | .94 / .92 | .905 / .704 (.874 / .597) | .69 / .92 (.63 / .92) | -5.1 / -12.0 (-5.8 / -20.4) | .927 / .881 |
| B' | 1.07 / .94 | 1.10 / .98 | 1.26 / 1.29 | .856 / .747 | .93 / .77 | +22.1 / +8.3 | .927 / .893 |
| GM' | .91 / 1.08 | 1.04 / 1.14 (- / 1.19) | 1.18 / .90 | .899 / .647 (- / .462) | 1.20 / 2.04 | +21.1 / +13.4 | .926 / .856 |
| a4_350 (no render) | .29 / .68 | .32 / .58 | 1.08 / 1.01 | .872 / .835 | 1.30 / 1.33 | +11.5 / +19.8 | .982 / .969 |
| plain-350 (no render) | 1.67 / 1.52 | 1.91 / 1.70 | 1.28 / 1.17 | .837 / .517 | 1.42 / 2.01 | +11.6 / -7.5 | .915 / .825 |
-> token cross-boundary regeneration |ratio - 1| <= .05 on both scenes at 20 and 30 calls: a pool-free, kick-free continuation does NOT redraw; the redrawing of
   gmc3 (2.63 / 2.02) is the second pass over the pool. Not below 1 either (no standstill by this measure; ViPE execution pending).
gate_bp (calls 1-3): LPIPS 1-3 token .021 / .064, GM' .024 / .070, B' .031 / .081, a4_350 .059 / .169, plain-350 .154 / .198; call 1 alone token .009 / .036.
   Vertical / horizontal scale ratio token 1.000-1.002 up to frame 128 (showroom), .999-1.001 up to 96 and .990 / .978 at 128 / 160 (playground; GM' .965, B' .963-1.024).
NEW DEFECTS of the token arm (not in B' / GM', which brighten): it DARKENS (luminance in anchor-visible cells -5 / -12 at 20 calls, -6 / -20 at 30; visible as a
   brown floor in the showroom from frame 320 and dark vegetation on the playground) and the showroom loses sharpness (.69 -> .63 of the recording).
   To check on the channel arm with the same numbers; candidates: density mask zeroing render cells (darker mean input), no corruption stage.
Visual, 30 calls (eval_gm/sheets/long_*.jpg, long_*.mp4): token follows the command on both scenes; playground structure degrades from frame 320 (flattened
   slide, fence-like railings) but keeps roof / stairs to 480 where GM' has collapsed; showroom wall kept to 480, coloured blocks appear low on the right wall.

## 12 Nov 14:40: token arm executes the slow command; out-and-back consistency (channel arm started 14:30)
ViPE execution (gate_b4_camera.py; exec = measured displacement projected on the planned one, lat = perpendicular part, yaw ratio; showroom / playground):
| run | exec @160 | exec @320 | lat @320 | yaw ratio @320 | per-call step ratio (median) |
| B token 250, stride 1 | 1.05 / .97 | 1.07 / 1.02 | .03 / .00 | 1.02 / 1.01 | 1.07 / 1.00 |
| a4_350 (no render), stride 1 | .03 / .19 | .05 / .16 | .04 / .10 | .04 / .18 | .14 / .17 |
| B token 250, stride 4 (@112) | - | 1.08 / .98 | .01 / .01 | 1.02 / 1.02 | 1.07 / .98 |
| a4_350, stride 4 (@112) | - | .98 / 1.35 | .09 / .11 | .90 / .92 | 1.01 / 1.27 |
-> stage A without render stands still at normal speed (executes 3-19% of the command; its regeneration .3-.6 is this), although it executes stride-4 steps.
   With the render the token arm follows the slow command exactly (and fixes the playground over-execution at stride 4: 1.35 -> .98). Standstill NOT inherited.
Out-and-back (diag_return.py: LPIPS way out vs way back at the same pose, 30 calls; all / generated cells / generated cells at far gap; lower = more consistent):
  B token 250  .041 / .045 / .047 (showroom)   .126 / .158 / .184 (playground)
  GM'          .080 / .089 / .098              .365 / .405 / .470
  GM' + revisit bank  .050 / .051 / .056       .311 / .352 / .428
  B' + pool (gmbp3)   .239 / .205 / .197       .516 / .539 / .565
  Memorizon released  .033 / .021 / .033       .129 / .124 / .122   (executes ~1/3 of the command: flattered)
-> the token arm returns to what it generated ~2x (showroom) / ~3x (playground) better than GM', at Memorizon's level while executing the full command.

## 12 Nov 14:36: channel arm, first 6 updates vs token arm (same rows / u / noise verified on update 1)
loss token .3249 / .2874 / .2949 / .2780 / .3067 / .2945 (mean .298), channel .3034 / .2722 / .2717 / .2461 / .2804 / .2630 (mean .273): channel LOWER by
.015-.032 from update 1. Not equal, as noted on 12 Nov 09:20: at update 1 the channel arm IS a4_350 without render (zero-init conv, no render slots), the token
arm puts 4 slots of a role the stage-A model never saw into the sequence, which costs ~.02 at first. Seconds / update 47 vs 65 (-28%; 17 instead of 21 slots).
Peak 24.0 vs 24.6 GB (only -.6 GB: the peak is not set by the 4 slots). Row norm median .028 vs .062, no rejection yet. Expected end ~17:50.
