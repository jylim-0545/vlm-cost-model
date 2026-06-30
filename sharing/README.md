# Vision-token 공유 (sharing)

우리는 vision 토큰을 저장·재사용해 인코딩을 건너뛴다. 그런데 VLM마다 토큰이 달라 저장본은 한
모델에만 쓸 수 있다. 이 모듈은 그 한계를 푸는 실험을 재현한다: **이미지를 공용 인코더(SigLIP
hub)로 한 번만 인코딩하고, 백본별 작은 변환기(adapter)만 거치면 여러 VLM이 그 한 벌을 공유**할 수
있는가? 여기서 다루는 건 **정확도 쪽**(공유+변환한 토큰이 각 백본의 원본 정확도를 얼마나 회복하나)
이다. 비용 정량화는 이 모듈 범위 밖이다.

출처는 EfficientVLM의 token-sharing 연구(`REPORT_VTOKEN_UNIFY.md`, `scripts/{e1_stagea,
d6_adapteronly,d12_holistic}.py`). 실험 결과·세팅·용어는 [`FINDINGS.md`](FINDINGS.md)에 정리.

---

## 구조 (hub-and-spoke)

```
                       ┌─ 변환기_A ─→ LLaVA-OV  (고정)
image ─→ SigLIP hub ─→ ├─ 변환기_B ─→ InternVL  (고정)
        (1회 인코딩)    └─ 변환기_C ─→ Qwen      (고정)
```

- **hub**: stock SigLIP-so400m → 384px당 729 토큰 × 1152차원.
- **변환기(adapter)**: hub 토큰을 백본의 토큰 자리로 보내는 작은 신경망. 백본은 전부 고정, 변환기만
  학습. 네 종류(raw / ridge / mlp_recon / mlp_e2e — FINDINGS §1).
- **주입 방식**: `vtpatch`(LLaVA-OV처럼 hub와 같은 계열 → vision-tower 출력 자리를 직접 교체. 학습
  경로 전부 여기) / `embed`(InternVL처럼 다른 계열 → 차원 맞춰 토큰 자리에 끼워넣기).

> **언제 잘 되나**: 영상 전체 이해와 "같은 인코더 계열"(OV)에서는 거의 무손실. 다른 계열의 세밀
> 인식은 15~21점 손실. 자세한 표는 FINDINGS §4·5.

---

## 파일

| 파일 | 역할 | GPU |
|---|---|---|
| `adapters.py` | 변환기(`RidgeAffine`/`ZScoreMLP`) + 닫힌 해 ridge fit + 학습된 변환기 로더 | ✕ (torch) |
| `methods.py` | `HubShare`: hub + 백본 로드 + 주입 패치 | ○ |
| `train.py` | `AdapterTrainer`: ridge / mlp_recon / mlp_e2e (+recon-anchor, cosine, multi-task) + 회복률·망각 평가 | ○ |
| `demo_train.py` | 변환기 1종 학습 후 native 대비 회복률 출력 (또는 학습된 변환기 로드해 평가) | ○ |
| `demo_latency.py` | 변환기가 인코더 대비 얼마나 싼지 실측 | ○ |
| `sweep.py` | variant×seed 다중-GPU sweep (`run_B.sh` 일반화) | ○ |
| `test_adapters.py` | 변환기 수학 단위테스트 (GPU 불필요) | ✕ |
| `run_share_demo.sh` / `smoke.sh` | 데모 래퍼 / 풀 실행 전 tiny 스모크 | 혼합 |

---

## 사용법 (전부 GPU)

```bash
# 변환기 사다리 (세밀=MMStar / 영상이해=NExT-QA), LLaVA-OV:
CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode ridge     --n-eval 400
CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode mlp_e2e --recon-lambda 8 --ft-steps 600
CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task nextqa --mode raw --frames 4 --n-eval 200
# 여러 과제 동시 (맞교환 + 망각):
CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --mode mlp_e2e --multitask aokvqa --forget aokvqa
# 이미 학습된 변환기를 로드해 평가만 (학습 없음):
CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --n-eval 400 \
    --load-adapter /home/yhlee/EfficientVLM/logs/d6_ci_s2_adapter.pt
```

회복률 = 변환기 / native. 기대치: 영상 이해 raw ≈0.96; 세밀 raw 0.83 → mlp_recon 0.88 → mlp_e2e 0.91.
절댓값은 표본에 따라 흔들리므로 회복 비율로 본다. 3-seed 오차막대는 `sweep.py`:

```bash
python -m sharing.sweep --gpus 0,1,2,3 --seeds 0,1,2 --tasks mmstar,nextqa --modes raw,mlp_recon,mlp_e2e
```

---

## 실행 환경

- **단위테스트·변환기 수학**(`adapters.py`, `test_adapters.py`)은 torch만 있으면 GPU 없이 돈다.
- **학습·실모델 데모**는 GPU 필요. 이 코드는 EfficientVLM 학습 박스(transformers 4.57, 4×RTX 4090)에서
  검증했다. 패키지·모델은 이미 설치/다운로드 돼 있다고 가정한다(설치 금지 — CLAUDE.md §1). 패키지
  목록은 [`requirements.txt`](requirements.txt).
- **경로**: SigLIP·OV는 HF 캐시, MMStar/NExT-QA/A-OKVQA는 EfficientVLM 쪽 기본 경로를 가리킨다.
  `demo_train` 인자(`--mmstar-tsv` 등)나 `SHARE_HF_ROOT`로 바꿀 수 있다.

풀 실행 전엔 `bash sharing/smoke.sh`로 전 경로(ridge/recon/e2e/raw + 비디오 + multitask)를 tiny하게
점검한다(크래시 확인용 — 회복 수치는 `sweep.py`).
