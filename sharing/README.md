# Vision-token 공유 (sharing)

이 cost-model repo에 vision-token **공유(hub-and-spoke)**를 얹은 독립 모듈. 하나의 공용 vision
인코더(**SigLIP-so400m = hub**)로 이미지를 **한 번** 인코딩하고, 백본별 경량 **adapter**만 거쳐
여러 VLM(LLaVA-OV / InternVL / Qwen)이 그 토큰을 공유해 쓴다. 이 모듈은 (1) 그 공유의 **비용**을
이 repo의 cost-model 언어(encode FLOP·저장 byte·break-even)로 계산하고, (2) adapter **학습**
(ridge / mlp_recon / mlp_e2e, multi-task)과 native 대비 **회복률**을 재현한다.

기법·결과의 출처는 EfficientVLM의 token-sharing 연구(`scripts/{e1_stagea,d6_adapteronly,d12_holistic}.py`,
`REPORT_VTOKEN_UNIFY.md`)다. 누적 결과·표는 [`FINDINGS.md`](FINDINGS.md) 참조. 이 모듈은 그 연구의
미완 제안 **E4(저장·서빙 회계)**를 cost-model 쪽에서 실현한 것이다.

---

## 1. 아이디어 (hub-and-spoke)

```
                       ┌─ adapter_A ─→ LLaVA-OV  (frozen)
image ─→ SigLIP hub ─→ ├─ adapter_B ─→ InternVL  (frozen)
        (1회 인코딩)    └─ adapter_C ─→ Qwen      (frozen)
```

- **hub** = stock `google/siglip-so400m-patch14-384` → 384px당 **729 토큰 × 1152-d**.
- **adapter** = hub 토큰을 백본의 vision-token 공간으로 보내는 경량 변환. 백본 LLM·인코더는 **전부
  freeze**, adapter만 학습. adapter는 SigLIP ViT의 **~1% 연산**(2-layer MLP 6.88 GFLOPs vs ViT 665
  GFLOPs) → N개 백본 서빙 시 인코딩 1회 + adapter N개라 **encode가 N→1로 분할상환**.
- **주입 경로**(백본별):
  - `vtpatch` (LLaVA-OV, **동일 계열**): hub를 OV 자신의 픽셀 타일에 태워 adapter로 1152→1152 매핑,
    OV native projector+anyres는 그대로. ridge/mlp_recon/mlp_e2e + multi-task가 **전부 이 경로**.
  - `embed` (InternVL, **다른 계열**): hub 토큰을 백본 grid로 resample → adapter 1152→HID →
    input-embeds의 image 자리에 splice. cross-encoder **일반화 스토리(ridge)** 용.

> ⚠ **언제 되나**: 공유가 거의 무손실인 건 **holistic 과제**와 **동일-인코더-계열(OV)**일 때다.
> cross-encoder fine-grained는 15~21pt 손실. 비용 절감은 그 sweet spot 안에서만 정확도 보존과
> 양립한다(아래 cost 함수는 절감을 *계산*할 뿐, 어디서나 공짜라고 주장하지 않는다). FINDINGS §5.

---

## 2. 두 층 (의존성으로 분리)

| 파일 | 역할 | GPU | torch |
|---|---|---|---|
| `cost.py` | encode "1회·N개 서빙" FLOP/$ + canonical TokenStore 저장 회계 + break-even. 순수 산술, config 재사용 | ✕ | ✕ |
| `adapters.py` | 어댑터 모듈(`RidgeAffine`/`ZScoreMLP`) + 닫힌 해 ridge fit + save/load. 모델 불필요 | ✕ | ○ |
| `methods.py` | `HubShare`: hub 인코더 + 백본 로드 + 주입 패치(vtpatch/embed). 실모델 | ○ | lazy |
| `train.py` | `AdapterTrainer`: ridge/mlp_recon/mlp_e2e (+ recon-anchor, cosine, multi-task) + 회복률·forgetting eval | ○ | lazy |
| `demo_cost.py` | 비용 sweep (GPU-free) | ✕ | ✕ |
| `demo_train.py` | adapter 1종 학습 + native 대비 회복률 | ○ | lazy |
| `demo_latency.py` | hub-encode vs adapter vs native-encode 실측(~1% ViT 검증) | ○ | lazy |
| `sweep.py` | variant×seed 다중-GPU sweep → 회복률 표 CSV (`run_B.sh` 일반화) | ○ | lazy |
| `test_adapters.py` | GPU-free 단위테스트 (어댑터 수학 + cost) | ✕ | ○ |

`import sharing.cost`는 torch도 transformers도 끌어오지 않는다 → 이 repo 어느 env에서나 비용/테스트는 안전.

---

## 3. 사용법

### 비용 (GPU 불필요)

```bash
python -m sharing.demo_cost
python -m sharing.demo_cost --backbones internvl3.5-8b,llava-ov-7b,internvl3.5-4b --frames 64
python -m sharing.demo_cost --hub-encode-ms 17.8 --no-gpu-stall      # 실측 encode 주입
```

→ **(A) ENCODE**: N백본 공유 시 vision-encode 절감 — N=2 49%, **N=4 ~74%**, N=16 93%(adapter≈1%
ViT, REPORT L375과 일치). **(B) STORAGE**: 3백본(InternVL-8B+OV+InternVL-4B), 64프레임에서 native
308MB → **공용 hub 107MB (65% 절감)**, break-even **N\*≈1~2 회/월**(repo의 vt_reuse처럼 거의 항상 이득).

### 학습 + 회복률 (GPU)

```bash
# adapter 사다리 (fine=MMStar / holistic=NExT-QA), LLaVA-OV:
CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode raw       --n-eval 400
CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode ridge     --n-eval 400
CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode mlp_recon --n-eval 400
CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode mlp_e2e --recon-lambda 8 --ft-steps 600
CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task nextqa --mode raw --frames 4 --n-eval 200
# multi-task (trade-off + forgetting):
CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode mlp_e2e --multitask aokvqa --recon-lambda 8 --forget aokvqa
```

회복률 = adapter / native. 기대치(FINDINGS): holistic raw≈0.96; fine raw≈0.83 → mlp_recon≈0.88 →
mlp_e2e≈0.91. **절댓값은 noisy → 회복 비율로 본다**(report 권고). 3-seed 오차막대는 `sweep.py`로:

```bash
python -m sharing.sweep --gpus 0,1,2,3 --seeds 0,1,2 --tasks mmstar,nextqa \
    --modes raw,mlp_recon,mlp_e2e --out-csv logs/share_sweep/recovery.csv
```

### 지연 (실측, GPU)

```bash
CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_latency --backbone llavaov --runs 20
```

→ adapter가 native ViT의 몇 %인지 cuda-event로 실측하고, 측정된 `--hub-encode-ms`를 출력(=demo_cost에 주입).

---

## 4. 실행 환경

- **비용/테스트** (`cost.py`, `demo_cost.py`, `test_adapters.py`, `adapters.py`): GPU·모델 불필요.
  `cost.py`는 PyYAML만, `adapters.py`/테스트는 torch(CPU)만. 이 repo의 `vlmcost` env로 충분.
- **학습·실모델 데모** (`methods.py`, `train.py`, `demo_*train/latency`, `sweep.py`): **GPU 필요**.
  이 token-sharing 코드는 EfficientVLM 학습 박스(**transformers 4.57.6, torch 2.6, 4×RTX 4090**,
  conda env `vlmeval`)에서 검증됐다. cost-model의 H100/`vlmcost` 박스에서도 SigLIP+OV가 있으면
  돈다. 패키지·모델은 **이미 설치/다운로드 돼 있다고 가정**(설치 금지 — CLAUDE.md §1).
- **모델/데이터 경로**(기본값; `SHARE_HF_ROOT` 등으로 override): SigLIP·OV는 HF 캐시
  (`EfficientVLM/hf_cache/hub`, `/mnt/nas/VLM/hf/hub`), MMStar=`EfficientVLM/LMUData/MMStar.tsv`,
  NExT-QA csv=`EfficientVLM/data/nextqa_local_mc.csv` + 비디오 `/mnt/nas/yhlee/nextqa/NExTVideo`,
  A-OKVQA parquet=HF 캐시. demo_train 인자로 모두 바꿀 수 있다.

---

## 5. 구성 + 스모크

```
sharing/
  README.md          # 이 문서
  FINDINGS.md        # 누적 실험 결과·표(회복 사다리, sweep, 계열 의존성, multi-task, negative transfer ...)
  cost.py            # encode/저장/break-even (순수 산술, GPU-free)
  adapters.py        # RidgeAffine / ZScoreMLP + ridge fit (torch, GPU-free)
  methods.py         # HubShare: hub + 백본 + 주입 패치 (GPU, lazy import)
  train.py           # AdapterTrainer: ridge/mlp_recon/mlp_e2e + multi-task + 회복률/forgetting
  demo_cost.py       # python -m sharing.demo_cost      (GPU-free)
  demo_train.py      # python -m sharing.demo_train     (GPU)
  demo_latency.py    # python -m sharing.demo_latency   (GPU)
  sweep.py           # python -m sharing.sweep          (다중-GPU, LONG)
  test_adapters.py   # python -m sharing.test_adapters  (GPU-free)
  run_share_demo.sh  # 단위테스트 + 비용 데모 래퍼
  smoke.sh           # 풀 실행 전 tiny 스모크 (GPU-free 테스트 + 4-GPU 4잡)
```

풀 실행 전 스모크(권장 — `smoke-test-first`):

```bash
bash sharing/smoke.sh        # GPU-free 테스트 + 4-GPU tiny 학습 스모크 (crash/plumbing 점검)
```

스모크는 ridge/mlp_recon/mlp_e2e/raw + nextqa 비디오 로더 + multitask + forgetting의 **전 경로**를 tiny
설정으로 태운다(정확도는 n이 작아 무의미 — 회복 수치는 `sweep.py`로). 검증 완료(2026-06-30, 4×4090):
GPU-free 단위테스트 10/10, 4잡 모두 무에러 end-to-end.
