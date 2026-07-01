# 사전 학습된 변환기 (adapters_pretrained)

token-sharing 연구에서 학습해 둔 변환기(adapter) 가중치 중 **핵심만 추려** Git LFS로 동봉한 것.
학습이 오래 걸리는 것만 담았다 — `raw`/`ridge`는 파일 없이 즉석 재현되므로(§) 넣지 않았다.

받기: `git lfs pull`  ·  포맷: `{state_dict, xm(z-score 평균), xs(z-score 표준편차), ...}`
(`sharing.adapters.load_study_adapter`가 읽는다).

| 파일 | 원본(log) | 종류 | 차원 | 무엇 | 재평가 |
|---|---|---|---|---|---|
| `fine_recon_s0.pt` | d6_B_fine_recon_s0 | mlp_recon | 1152→1152 | 세밀(MMStar), MLP 재현 사전학습 | ✅ OV |
| `fine_e2e_s0.pt` | d6_ci_s0 | mlp (e2e) | 1152→1152 | 세밀, recon 후 정답 미세조정 | ✅ OV |
| `multitask_l2.pt` | d6_mt_l2 | mlp (e2e) | 1152→1152 | MMStar+A-OKVQA 함께 학습 (λ=2) | ✅ OV |
| `crossbackbone_internvl8.pt` | d10_iv8_l8 | mlp | 1152→4096 | 다른 계열(InternVL-8B) 변환기 | ⚠ 보관용 |

**재평가 (OV 변환기, 학습 없이):**
```bash
CUDA_VISIBLE_DEVICES=0 python -m sharing.demo_train --task mmstar --n-eval 400 \
    --load-adapter sharing/adapters_pretrained/fine_e2e_s0.pt
```
재현 회복률(MMStar n=400, seed 0, native 56.2 공통): raw 0.81(즉석) · recon 0.87 · e2e 0.87.
(연구의 3-seed 평균은 raw 0.83 / recon 0.88 / e2e 0.91 — s0 단독이라 e2e가 평균보다 낮게 나옴.)

**cross-backbone**은 `embed` 주입이라 현재 `--load-adapter` 재평가 경로(OV 전용)로는 안 돌아간다 —
가중치·§4(계열 의존성) 근거 **보관용**. 재평가하려면 InternVL embed 평가 경로가 추가로 필요.

**raw / ridge**는 동봉하지 않는다: `--mode raw`(표준화 통과) / `--mode ridge`(닫힌 해, 수초)로 즉석 생성.
프리즈된 ridge 사본이 필요하면 `demo_train --mode ridge`로 만들어 저장하면 된다.
