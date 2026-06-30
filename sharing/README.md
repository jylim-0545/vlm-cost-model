# Vision-token 공유 (sharing)

공용 vision 인코더(**SigLIP hub**)로 이미지를 한 번 인코딩하고, 백본별 경량 **adapter**만
거쳐 여러 VLM(LLaVA-OV / InternVL / Qwen)이 그 토큰을 공유해 쓴다. 이 모듈은 (1) 그 공유의
**비용**을 이 repo의 cost-model 언어로 계산하고, (2) adapter **학습**(ridge / mlp_recon /
mlp_e2e, multi-task)과 native 대비 **회복률**을 재현한다.

기법·수치의 출처는 EfficientVLM의 token-sharing 연구다(`REPORT_VTOKEN_UNIFY.md`,
`scripts/{e1_stagea,d6_adapteronly,d12_holistic}.py`). 누적 결과는 [`FINDINGS.md`](FINDINGS.md).
이 모듈은 그 연구가 남긴 "저장·서빙 회계(E4)"를 cost-model 쪽에서 구현한 것이다.

---

## 아이디어 (hub-and-spoke)

```
                       ┌─ adapter_A ─→ LLaVA-OV  (frozen)
image ─→ SigLIP hub ─→ ├─ adapter_B ─→ InternVL  (frozen)
        (1회 인코딩)    └─ adapter_C ─→ Qwen      (frozen)
```

- **hub** = stock SigLIP-so400m → 384px당 729 토큰 × 1152-d.
- **adapter** = hub 토큰을 백본의 vision-token 자리로 보내는 경량 변환. 백본은 전부 freeze,
  adapter만 학습. adapter는 ViT의 **~1% 연산** → N개 백본 서빙 시 인코딩 1회로 분할상환.
- **주입 경로**: `vtpatch`(LLaVA-OV, 동일 계열 — adapter 1152→1152, 학습 경로 전부 여기) /
  `embed`(InternVL, 다른 계열 — 1152→HID 후 image 토큰 자리에 splice).

> **언제 되나**: holistic 과제와 동일-인코더-계열(OV)에서 거의 무손실. cross-encoder의
> fine-grained는 15~21pt 손실. 비용 절감은 그 범위 안에서만 정확도와 양립한다(`cost.py`는
> 절감을 계산할 뿐, 어디서나 공짜라고 주장하지 않는다). 자세한 표는 FINDINGS §4·5.

---

## 구성

| 파일 | 역할 | GPU |
|---|---|---|
| `cost.py` | encode "1회·N개 서빙" + 공용 TokenStore 저장 회계 + break-even (순수 산술) | ✕ |
| `adapters.py` | `RidgeAffine`/`ZScoreMLP` + 닫힌 해 ridge fit + 학습된 어댑터 로더 | ✕ |
| `methods.py` | `HubShare`: hub + 백본 로드 + 주입 패치 | ○ |
| `train.py` | `AdapterTrainer`: ridge / mlp_recon / mlp_e2e (+recon-anchor, cosine, multi-task) + 회복률·forgetting eval | ○ |
| `demo_cost.py` / `demo_train.py` / `demo_latency.py` | 비용 sweep / 학습·회복 / 지연 실측 | ✕ / ○ / ○ |
| `sweep.py` | variant×seed 다중-GPU sweep (`run_B.sh` 일반화) | ○ |
| `test_adapters.py` | 단위테스트 10개 | ✕ |
| `run_share_demo.sh` / `smoke.sh` | 데모 래퍼 / 풀 실행 전 tiny 스모크 | 혼합 |

`import sharing.cost`는 torch도 transformers도 끌어오지 않으므로, 비용·테스트는 GPU 없이 어디서나 돈다.

---

## 사용법

**비용 (GPU 불필요)**
```bash
python -m sharing.demo_cost
python -m sharing.demo_cost --backbones internvl3.5-8b,llava-ov-7b,internvl3.5-4b --frames 64
```
→ encode 절감(N=4 ~74%), 공용 TokenStore 저장 절감(3백본·64f: 308→107MB), break-even N*.

**학습 + 회복률 (GPU)**
```bash
CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode ridge     --n-eval 400
CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode mlp_e2e --recon-lambda 8 --ft-steps 600
CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task nextqa --mode raw --frames 4 --n-eval 200
CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode mlp_e2e --multitask aokvqa --forget aokvqa
```
회복률 = adapter / native. 기대치: holistic raw ≈0.96; fine raw 0.83 → mlp_recon 0.88 → mlp_e2e 0.91.
절댓값은 noisy하므로 회복 비율로 본다. 3-seed 오차막대는 `sweep.py`로.

**학습된 어댑터 재평가 (학습 없이)**
```bash
CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --n-eval 400 \
    --load-adapter /home/yhlee/EfficientVLM/logs/d6_ci_s2_adapter.pt
```

**전체 sweep / 지연 측정**
```bash
python -m sharing.sweep --gpus 0,1,2,3 --seeds 0,1,2 --tasks mmstar,nextqa --modes raw,mlp_recon,mlp_e2e
CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_latency --backbone llavaov   # adapter가 ViT의 몇 %인지 실측
```

---

## 실행 환경

- **비용·테스트**(`cost.py`, `demo_cost.py`, `adapters.py`, `test_adapters.py`)는 GPU·모델 불필요.
- **학습·실모델 데모**는 GPU 필요. 이 코드는 EfficientVLM 학습 박스(transformers 4.57, 4×RTX 4090)에서
  검증했다. 패키지·모델 가중치는 이미 설치/다운로드 돼 있다고 가정한다(설치 금지 — CLAUDE.md §1).
  패키지 목록은 [`requirements.txt`](requirements.txt).
- **경로**: SigLIP·OV는 HF 캐시, MMStar/NExT-QA/A-OKVQA는 EfficientVLM 쪽 기본 경로를 가리킨다.
  `demo_train` 인자(`--mmstar-tsv` 등)나 `SHARE_HF_ROOT`로 바꿀 수 있다.

풀 실행 전에는 `bash sharing/smoke.sh`로 전 경로(ridge/recon/e2e/raw + 비디오 + multitask)를 tiny하게
점검한다(crash 확인용 — 회복 수치는 `sweep.py`).
