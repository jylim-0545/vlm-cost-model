# Vision-token 공유 — 누적 실험 결과 (FINDINGS)

출처: EfficientVLM token-sharing 연구(`REPORT_VTOKEN_UNIFY.md` 547줄 + `SUMMARY_vtoken_unify.md`),
구현 `scripts/{e1_stagea, d6_adapteronly, d12_holistic, d10_unified, e2_*}.py`. 이 문서는 그 결과를
cost-model repo의 `sharing/` 모듈 용어로 정리한 것이다. **이 모듈(train.py/cost.py)이 재현·계산하는
대상**이 무엇인지 한곳에서 본다.

> ⚠ **측정 신뢰도**: 단일-seed MMStar 절댓값은 **53.7 ± 2.3**(eval n=400~600, seed별 51~57)로 noisy.
> 1~2pt 차이는 노이즈다. **회복률(adapter/native)은 0.90 ± 0.02로 안정**(절댓값이 흔들려도 adapter·native가
> 같이 흔들려 비율은 일관). 그래서 **정규화 회복률로 보고**한다. multi↔single forgetting은 같은-seed
> paired라 신뢰(8.0 vs 5.0).

---

## 0. 셋업

- **hub** = stock `google/siglip-so400m-patch14-384` (1152-d, 729 tok/384px, no CLS). 인코딩 1회.
- **백본**(frozen): LLaVA-OV-7B(SigLIP 계열=hub와 동일), InternVL3.5-8B/4B(InternViT-300M, 1024-d),
  Qwen2.5-VL-7B(자체 ViT, 1280-d). Qwen3-VL은 deepstack+mrope로 splice 불가 → 제외.
- **adapter 출력 차원** = OV는 1152(VT 자리, projector 뒤따름), embed-splice 백본은 post-projector HID.
- **eval**: MMStar(fine, 이미지 4지선다, n=400) / NExT-QA(holistic, 비디오 5지선다, n=300) /
  MLVU(holistic 비디오, n=300). **forced-choice**(답 letter logit argmax), paired(같은 문항을 모든 config로).
- **native** = 백본이 자기 vision 토큰을 쓴 상한(어댑터와 **동일 splice 경로**로 채점 → 경로 drift 상쇄).
- **회복률** = adapter_acc / native_acc.

---

## 1. 어댑터 사다리 — raw → recon → E2E (LLaVA-OV, reader frozen)

| variant | 무엇 | holistic 회복 | fine 회복 | fine 절댓값 (native 56.2) |
|---|---|---|---|---|
| **raw inject** | z-score한 hub 토큰 그대로(학습 X, 차원만 맞음) | **96 ± 2%** | **83 ± 2%** | ~47.8 |
| **z-affine (ridge)** | `((x−μ)/σ)·W+b` 닫힌 해 token-matching (label-free) | (fig 제외, recon은 fine서 MLP보다↓) | — | — |
| **mlp_recon** | `1152→2048→1152` GELU MLP, native 토큰 MSE 재현 사전학습 | 95 ± 5% | **88 ± 2%** | 49.2 |
| **mlp_e2e** | recon 후 VQA gold-letter CE 미세조정 (+anchor+cosine) | 93 ± 4% | **91 ± 2%** | **53.0** |

- **holistic은 차원만 맞추면 됨**: raw 주입 96%, 학습 변환을 더해도 노이즈 내 추가 이득 없음.
- **fine은 학습 사다리**: raw 83 → recon +5 → E2E +3 = 91%. 절댓값 47.8→49.2→53.0(native 56.2에 3.2pt).
- 이 모듈: `mode ∈ {raw, ridge, mlp_recon, mlp_e2e}` (`sharing.train.AdapterTrainer`).
  ※ 동일-계열(OV) fine에서는 **raw(z-score passthrough)가 ridge(재현 fit)보다 높다** — hub가 곧 OV의
  SigLIP 계열이라 그대로 통과시키는 게, 제한된 데이터로 OV-VT를 재현하는 것보다 낫다. ridge/recon은
  cross-encoder에서 진가(차원·기하 정렬).

---

## 2. 미세조정 lr · 스케줄 sweep (d6c, recon 사전학습 49.2에서 출발, MLP-only, reader 고정)

| step | ft 1e-5 | ft 3e-5 | ft 1e-4 | ft 3e-4 |
|---|---|---|---|---|
| 600 | 46.5 | 47.0 | 40.5 | 29.0 |
| 3600 | 51.8 | 52.0 | 48.8 | 38.2 |
| **6600(최종)** | **51.8** | **53.0** | 46.5 | 37.0 |

- **작은 lr(1e-5/3e-5)**: 초기 dip 후 회복 → 49.2 넘어 **~52~53 plateau**("망가짐"이 아니라 transient).
  **ft 3e-5 = 53.0 = adapter-only 신기록**(reader-LoRA 52.5 초과, native 56.2에 3.2pt).
- **큰 lr(3e-4)**: 급락 후 회복 못 함(37) = 망가짐. 중간(1e-4)은 시작점 언저리 진동.
- 이 모듈: `--ft-lr`, `--ft-steps`, `--sched {const,cosine}`, `--warmup` (`ShareTrainConfig`).

---

## 3. recon-anchor + cosine — forgetting 극복하며 fine 향상 (d6e)

VQA-CE 손실에 **recon-anchor** 추가: `loss = CE + λ·‖adapter(stock) − nativeVT‖²` (λ=8) + cosine 스케줄.
→ 다른-과제 forgetting **14 → 6.3** 으로 줄면서 MMStar **49.5 → 53.0** 동시 향상. anchor가 adapter
출력을 native VT에 묶어 일반성을 유지(EWC류 continual-learning 기법의 적용).
- 이 모듈: `--recon-lambda 8` (`ShareTrainConfig.recon_lambda`). mlp_e2e에서 자동 적용.

---

## 4. 인코더 계열 의존성 — fine 공유의 핵심 결과 (Phase1, `d10_unified.py`)

단일 SigLIP hub + 경량 adapter, MMStar, reader frozen, D6e 레시피:

| 백본 | 인코더 | adapter / native | 회복률 |
|---|---|---|---|
| LLaVA-OV | SigLIP (= hub 계열) | 53.0 / 56.2 | **0.94** |
| InternVL-4B | InternViT-300M | 45.2 / 60.0 | **0.75** |
| InternVL-8B | InternViT-300M | 40.8 / 58.8 | **0.67–0.69** |
| Qwen2.5-VL | 자체 ViT | 39.8 / 60.8 | **0.65** |

→ **fine 공유는 백본 인코더가 hub와 같은 계열일 때만 강하다.** holistic은 cross-encoder도 0.86~0.99.
E1 master table(floor-보정 recovered): MMStar 0.29~0.67 / NextQA 0.86~0.93 / MLVU-topic 0.73~0.93.
- 이 모듈: 동일-계열(OV)은 `methods.HubShare("llavaov", mode=vtpatch)`로 학습/평가, cross-encoder는
  `embed` 주입(InternVL) — ridge 일반화 스토리. **cost.py의 절감은 이 sweet spot에서만 정확도와 양립**.

---

## 5. 언제 공유되나 (when-shareable)

| 상황 | 공유? | adapter / native |
|---|---|---|
| **holistic(영상)** | ✅ 됨 | 단순 raw~affine로 거의 무손실 |
| **fine, 동일-계열 백본**(OV) | ⚠ 학습하면 됨 | 53 / 56 (mlp+E2E) |
| **여러 과제 동시** | ⚠ trade-off | 51 / 56 |
| **fine, 다른-계열 백본**(InternVL/Qwen) | ❌ 아직 | 40 / 59 |

---

## 6. 한 어댑터로 여러 과제 — multi-task trade-off (Phase2/3)

한 OV 어댑터를 MMStar + A-OKVQA로 같이 학습(좌표 = (MMStar, A-OKVQA)):

| config | MMStar | A-OKVQA | 비고 |
|---|---|---|---|
| native | 56.2 | 97.3 | 상한(우상단) |
| single-task (MMStar만) | 51.8 | 91.3 | 다른 과제 떨어짐 |
| **multi-task λ=2** | 51.0 | **93.7** | 균형 best |
| multi-task λ=8 | 49.8 | 93.7 | fine 약간↓ |

→ 여러 과제 함께 학습하면 다른 과제 forgetting **6.0 → 3.7** 완화, fine은 약간 희생(trade-off).
- 이 모듈: `--multitask aokvqa` (train.py가 두 task 샘플 interleave) + `--forget aokvqa`로 forgetting eval.

---

## 7. 음의 전이 (negative transfer) — 잘못된 데이터로 E2E는 해롭다 (E2 Stage B)

affine 어댑터를 A-OKVQA LM-loss로 미세조정 → **MMStar가 오히려 악화**(회복 0.41 → 0.17). 해결은
**타깃-도메인 recon 사전학습 + 작은-lr + recon-anchor(λ) + cosine**. 외부 fine-VQA(D8): TextVQA/OCR
학습 시 MMStar 일반화 41 → 48.5 (데이터 *종류*가 양보다 중요, OCR>상식), 단 native 미달.
- 함의: mlp_e2e는 recon-anchor 없이 막 돌리면 퇴화할 수 있음 → 이 모듈 mlp_e2e 기본 recipe에 anchor 포함.

---

## 8. 재현 R² ≠ 정확도, 재현 천장, cross-token (D1–D4)

- **R² ≠ 정확도**: MLP가 재현 R²는 더 높지만(0.77 vs ridge 0.42) MMStar는 더 낮음 — MLP가
  massive-activation·고분산 차원에 과적합해 task-판별 저분산 방향을 뭉갠다. → 선형(ridge) ≥ MLP(fine 재현).
- **재현 천장 ~0.20–0.25**: 맵 형태·용량(d 1024/2048/3072)·데이터 불문 SigLIP→native 재현 R²는
  ~0.20–0.25에서 천장. native 토큰의 ~75%는 *어떤* 맵으로도 SigLIP의 함수가 아님. **단 R²≠정확도라
  ridge(R²0.24)도 holistic은 ~0.9 회복** → fine 격차는 hub 정보/해상도(SigLIP-384 < InternVL-448+타일) 탓.
- **cross-token이지 per-token 병목이 아님**(D1/D2 정정): per-token ridge held-out R² ~0.24–0.35,
  attention-resampler의 초기 우위는 과적합이었고 데이터 늘리면 ridge로 수렴. "차원 ≥ 이므로 무손실
  변환"은 성립 안 함(차원이 아니라 *함수 존재* 문제).

---

## 9. 비용 (이 모듈 `cost.py` 가 계산)

REPORT L375: **2-layer adapter = 6.88 GFLOPs/img · 4.72M params = SigLIP ViT(665 GFLOPs · 428M)의
~1%** (97배 가벼움). N백본 서빙 = 인코딩 1회 + adapter N개 → **N=4면 ~74% encode 절감**.

이 모듈은 그 위에 두 축을 cost-model로 계산한다 (`sharing/cost.py`, `demo_cost.py`):
- **(A) encode "1회·N개 서빙"**: baseline N×ViT vs shared ViT + N×adapter → FLOP/$ 절감(N=4 74%).
- **(B) canonical TokenStore**: 백본마다 native 토큰을 따로 저장(Σ bytes_i) vs hub 토큰 1벌 저장 →
  3백본·64f에서 308MB→107MB(65%↓), break-even N\*≈1~2 회/월. KV-reuse(REPORT_KVREUSE)의 "vision-token
  store가 cross-model에 유리" 결론과 합쳐 시스템 논거를 완성(= 연구의 미완 제안 **E4**).
- repo의 `6.88`(H100 $6.88/h)와 adapter의 6.88 GFLOPs는 **우연 일치** — 혼동 주의.

---

## 10. 출처 ↔ 이 모듈

| 연구 스크립트 | 이 모듈 |
|---|---|
| `e1_stagea.py` (ridge, cross-backbone, zero-finetune) | `train.py` mode=ridge + `methods.py` embed 주입 |
| `d6_adapteronly.py` (fine/MMStar: affine/ridgeaffine/mlp/mlp_recon, anchor, multitask, forgetting) | `train.py` (MMStar task) + `aokvqa_task` |
| `d12_holistic.py` (holistic/NextQA 비디오: mlp_recon+E2E+anchor+cosine) | `train.py` (nextqa task) |
| `d10_unified.py` (계열 의존성 표) | §4 (문서) |
| `run_B.sh` / `smoke_B.sh` (3-seed 4-GPU sweep) | `sweep.py` / `smoke.sh` |
| `REPORT_VTOKEN_UNIFY.md` L375 (FLOP/param) | `cost.py` 상수 + §9 |

논문 연결: hub-and-spoke ↔ Vision Wormhole·OneLLM; 선형/affine ↔ Model Stitching·Platonic;
recon 사전학습 ↔ MoVE-KD·distillation; recon-anchor ↔ continual learning(EWC). 상세는
`EfficientVLM/REPORT_VTOKEN_UNIFY.md`.
