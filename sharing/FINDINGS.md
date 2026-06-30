# Vision-token 공유 — 누적 결과 (FINDINGS)

EfficientVLM token-sharing 연구(`REPORT_VTOKEN_UNIFY.md`, `scripts/{e1_stagea,d6_adapteronly,
d12_holistic,d10_unified}.py`)의 결과를 이 모듈 용어로 정리한 문서. `sharing/`의 `train.py`·
`cost.py`가 무엇을 재현·계산하는지 한곳에서 본다.

> **수치 읽는 법**: MMStar 절댓값은 seed별 51~57로 흔들린다(eval n=400, 1~2pt는 노이즈). 반면
> **회복률 = adapter/native 는 0.90±0.02로 안정적**이라(둘이 같이 흔들려 비율은 일관) 보고는 회복률
> 기준으로 한다.

## 셋업

- **hub**: stock SigLIP-so400m (1152-d, 384px당 729 토큰). 이미지당 1회 인코딩.
- **백본**(전부 freeze): LLaVA-OV-7B(SigLIP 계열 = hub와 동일), InternVL3.5-8B/4B(InternViT, 1024-d),
  Qwen2.5-VL-7B(자체 ViT, 1280-d). Qwen3는 구조상(deepstack+mrope) 제외.
- **eval**: MMStar(fine, 이미지 4지선다) / NExT-QA·MLVU(holistic, 비디오). 답 letter logit argmax,
  같은 문항을 모든 조건으로 채점(paired). **native** = 백본이 자기 토큰을 쓴 상한(어댑터와 동일 경로로 채점).

## 1. 어댑터 사다리 (LLaVA-OV, reader frozen)

| mode | 방법 | holistic 회복 | fine 회복 |
|---|---|---|---|
| **raw** | z-score한 hub 토큰 그대로 (학습 X) | **0.96** | **0.83** |
| **ridge** | 닫힌 해 z-affine token-matching (label-free, 수초) | — | (fine은 raw보다 낮아 생략) |
| **mlp_recon** | MLP를 native 토큰 MSE로 재현 사전학습 | 0.95 | **0.88** |
| **mlp_e2e** | recon 후 VQA-CE 미세조정 (+recon-anchor, cosine) | 0.93 | **0.91** |

- **holistic은 차원만 맞추면 됨**(raw 0.96, 학습 더해도 노이즈 내 이득 없음).
- **fine은 학습 사다리**(raw 0.83 → recon 0.88 → e2e 0.91; 절댓값 47.8 → 49.2 → 53.0, native 56.2).
- 동일 계열(OV)에선 raw(z-score 통과)가 ridge(재현 fit)보다 높다 — hub가 곧 OV의 SigLIP이라 그대로
  넣는 게 낫다. ridge/recon은 차원·기하가 다른 cross-encoder에서 의미가 있다.

## 2. 미세조정 lr·스케줄 (recon 49.2에서 출발, MLP만 학습)

| step | lr 1e-5 | 3e-5 | 1e-4 | 3e-4 |
|---|---|---|---|---|
| 600 | 46.5 | 47.0 | 40.5 | 29.0 |
| 6600 (최종) | 51.8 | **53.0** | 46.5 | 37.0 |

작은 lr은 초기 dip 후 회복해 ~53 plateau(망가짐 아님). **3e-5가 53.0으로 adapter-only 최고**(native에
3.2pt). 큰 lr(3e-4)은 망가져 회복 못 함. → `train.py`의 `--ft-lr`, `--sched cosine`, `--warmup`.

## 3. recon-anchor (forgetting 완화)

VQA-CE 손실에 `+ λ·‖adapter(hub) − nativeVT‖²`(λ=8) + cosine을 더하면, 다른 과제 forgetting이 14→6.3로
줄면서 MMStar는 49.5→53.0으로 같이 오른다. anchor가 adapter 출력을 native에 묶어 일반성을 유지한다(EWC류).
→ `--recon-lambda 8` (mlp_e2e 기본).

## 4. 인코더 계열 의존성 (fine 공유의 핵심)

단일 hub + 경량 adapter, MMStar, reader frozen:

| 백본 | 인코더 | adapter / native | 회복률 |
|---|---|---|---|
| LLaVA-OV | SigLIP (hub와 동일 계열) | 53.0 / 56.2 | **0.94** |
| InternVL-4B | InternViT | 45.2 / 60.0 | 0.75 |
| InternVL-8B | InternViT | 40.8 / 58.8 | 0.67 |
| Qwen2.5-VL | 자체 ViT | 39.8 / 60.8 | 0.65 |

**fine 공유는 백본 인코더가 hub와 같은 계열일 때만 강하다.** holistic은 cross-encoder도 0.86~0.99로 무난.

## 5. 언제 공유되나

| 상황 | 공유? |
|---|---|
| holistic (영상) | ✅ 거의 무손실 (raw로 충분) |
| fine, 동일 계열 (OV) | ⚠ 학습하면 됨 (mlp_e2e 0.91) |
| 여러 과제 동시 | ⚠ trade-off (§6) |
| fine, 다른 계열 (InternVL/Qwen) | ❌ 아직 (0.65~0.75) |

## 6. 한 어댑터로 여러 과제 (multi-task trade-off)

OV 어댑터를 MMStar + A-OKVQA로 함께 학습:

| 조건 | MMStar | A-OKVQA |
|---|---|---|
| native (상한) | 56.2 | 97.3 |
| MMStar만 | 51.8 | 91.3 |
| **multi λ=2** | 51.0 | **93.7** |
| multi λ=8 | 49.8 | 93.7 |

함께 학습하면 다른 과제 forgetting이 6.0→3.7로 줄고, fine은 약간 희생된다(trade-off). λ=2가 균형.
→ `--multitask aokvqa` + `--forget aokvqa`.

## 7. 학습이 해로운 경우 / 재현의 한계

- **음의 전이**: 엉뚱한 데이터로 E2E하면 오히려 나빠진다(A-OKVQA로 affine 미세조정 → MMStar 회복 0.41→0.17).
  그래서 mlp_e2e는 recon 사전학습 + 작은 lr + anchor를 함께 쓴다.
- **재현 R² ≠ 정확도**: MLP는 재현 R²가 ridge보다 높아도(0.77 vs 0.42) MMStar는 더 낮다 — 고분산 차원에
  과적합해 판별 방향을 뭉갠다. 그래서 fine 재현은 선형(ridge) ≥ MLP.
- **재현 천장 ~0.20–0.25**: 맵 형태·용량·데이터를 늘려도 SigLIP→native 재현 R²는 여기서 막힌다. 단
  R²≠정확도라 ridge(R²0.24)도 holistic은 0.9 회복 → fine 격차는 hub의 정보/해상도(SigLIP-384) 문제다.

## 8. 비용 (이 모듈 `cost.py` 가 계산)

adapter(2-layer MLP)는 **6.88 GFLOPs·4.72M params = SigLIP ViT(665 GFLOPs·428M)의 ~1%**. 두 절감 축:

- **(A) encode "1회·N개 서빙"**: N×ViT → ViT + N×adapter. N=4면 ~74% 절감.
- **(B) 공용 TokenStore**: 백본마다 따로 저장(Σ bytes_i) → hub 1벌 저장. 3백본·64f에서 308→107MB(65%),
  break-even N*≈1~2회/월. KV-reuse(`REPORT_KVREUSE`)의 "vision-token store가 cross-model에 유리"와
  합쳐 시스템 논거가 된다(= 연구의 미완 제안 E4).
- 주의: repo의 `6.88`(H100 $6.88/h)와 adapter 6.88 GFLOPs는 우연히 같은 숫자다.

## 출처 ↔ 이 모듈

| 연구 스크립트 | 이 모듈 |
|---|---|
| `e1_stagea.py` (ridge, cross-backbone) | `train.py` mode=ridge + `methods.py` embed 주입 |
| `d6_adapteronly.py` (fine: affine/ridge/mlp/mlp_recon, anchor, multitask) | `train.py` (MMStar) + `aokvqa_task` |
| `d12_holistic.py` (holistic 비디오: recon+E2E) | `train.py` (nextqa) |
| `run_B.sh` / `smoke_B.sh` (3-seed sweep) | `sweep.py` / `smoke.sh` |
| `REPORT_VTOKEN_UNIFY.md` L375 (FLOP/param) | `cost.py` 상수 + §8 |
