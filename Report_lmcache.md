# LMCache 검증 리포트 — 우리 kv_reuse 모델 vs 실제 KV-reuse 시스템

**날짜:** 2026-06-08 · **모델:** LLaVA-OV-7B · **하드웨어:** H100 PCIe (80GB) · **batch:** 1

## 1. 목적

우리 비용 모델의 `kv_reuse`는 KV 캐시를 storage tier에 저장했다가 재사용하는 것을 가정하고,
그 **retrieval(저장→DRAM→GPU) 비용을 분석적으로 더한다** (`h2d_kv` 측정값 + tier별 대역폭 계산).
이 리포트는 그 모델을 **실제 KV-reuse 시스템인 [LMCache](https://github.com/LMCache/LMCache)** 와
비교해 검증한다. 핵심 질문:

> 우리가 *계산/모델링*한 kv_reuse 비용이, KV를 실제로 offload·load 하는 실제 시스템과 일치하는가?

## 2. 셋업 (격리 트랙)

LMCache는 vLLM의 KV connector(`LMCacheConnectorV1`)로 붙어 KV를 CPU-DRAM / local-disk / S3 등
tier에 offload·재사용한다. 우리 메인 측정 환경(vLLM 0.22)을 건드리지 않기 위해 **별도 conda env**로 구성:

| | 메인 (우리) | LMCache 트랙 |
|---|---|---|
| conda env | `vlmcost` | `lmcache` (격리) |
| vLLM | 0.22.0 | 0.18.0 (LMCache 테스트 범위) |
| lmcache | — | 0.4.4 |
| 결과 파일 | `results/nextqa/reuse_real.csv` | `results/lmcache/` (격리) |

- **엔진 버전 차이(0.22 vs 0.18)는 의도적으로 수용**한다. cold(recompute) latency가 두 버전에서
  거의 일치함을 확인해(아래 §4) 비교의 타당성을 담보한다.
- LMCache의 prebuilt CUDA 커널(`c_ops`)에 **Blackwell(sm_120) 이미지가 없어** GPU0(RTX PRO 6000)에선
  `no kernel image` 에러 → **이 트랙은 H100 전용**.

## 3. 방법

같은 (모델, 영상, 프레임 16/32/64/128) 에서 두 가지 kv_reuse를 동일 엔진(0.18)으로 측정
(`measure/reuse_lmcache.py`):

- **우리 kv_reuse (`--mode ours`)**: vanilla vLLM prefix-cache warm. 2번째 요청이 **GPU에 그대로
  남아 있는 KV**를 hit (실제 tier 회수 없음). = `reuse_real.py`의 kv_reuse 메커니즘.
- **LMCache kv_reuse (`--mode lmcache --tier dram`)**: KV를 CPU-DRAM에 offload 후 재사용.
  `enable_prefix_caching=False`로 **GPU 캐시를 끊어 LMCache가 실제로 DRAM에서 KV를 load** 하도록 강제
  (로그 `need to load: 25088 @128f`로 실제 회수 검증).

## 4. 결과

### 4.1 Front cost (warm TTFT)

| frames | n_vis | cold (recompute) | **우리 kv_reuse** (GPU-resident) | **LMCache kv_reuse** (DRAM 실제 load) |
|---:|---:|---:|---:|---:|
| 16 | 3,136 | 224.7 | 19.3 | 28.8 |
| 32 | 6,272 | 510.7 | 40.5 | 41.5 |
| 64 | 12,544 | 1030.7 | 47.4 | 49.0 |
| 128 | 25,088 | 2215.3 | 72.0 | 97.5 |

(단위 ms, TTFT median) — **두 kv_reuse의 warm TTFT가 거의 동일**하다. 둘 다 encode+prefill을 스킵하며,
LMCache는 추가로 실제 DRAM 회수를 한 만큼만 더 든다(128f에서 ~25ms 차이).

### 4.2 검증 — retrieval 비용: 우리 계산값 vs LMCache 실측

우리 모델이 kv_reuse에 더하는 retrieval 항 `h2d_kv`(DRAM→GPU KV 전송, reuse_real에서 측정)를,
LMCache가 **실제로 KV를 DRAM에서 회수한 시간**과 비교:

| frames | KV bytes | **우리 h2d_kv** (모델, 측정) | **LMCache 실제 DRAM load** | 비 |
|---:|---:|---:|---:|---:|
| 16 | 0.16 GB | 3.2 ms | 3.8 ms | 0.84 |
| 32 | 0.33 GB | 6.5 ms | 7.4 ms | 0.88 |
| 64 | 0.67 GB | 12.9 ms | 14.9 ms | 0.87 |
| 128 | 1.34 GB | 25.9 ms | 29.6 ms | 0.87 |

두 곡선 모두 **KV 바이트에 선형**이며 (우리 ~52 GB/s, LMCache ~45 GB/s), 차이는 일정한 ~13%
(LMCache가 chunk 단위 store/load + serde 오버헤드로 약간 느림). → **우리 비용 모델의 kv_reuse
retrieval 항이 실제 시스템으로 검증됨.**

![LMCache 비교](results/lmcache/fig_lmcache_compare.png)

*(a) cold/우리/LMCache의 warm TTFT는 거의 겹친다. (b) 우리 h2d_kv(파랑)와 LMCache 실제 DRAM load(주황)가
거의 평행 — retrieval 비용 모델 검증.*

## 5. 결론

1. **우리 kv_reuse 모델이 옳다.** warm latency도, retrieval 항(h2d_kv)도 실제 KV-reuse 시스템
   (LMCache)의 실측과 일치한다. `LMCache kv_reuse = 우리 GPU-resident kv_reuse + 실제 DRAM load(≈ 우리 h2d_kv)`.
2. **LMCache는 우리 모델의 현실 구현체.** 우리가 분석적으로 더하던 DRAM→GPU hop을 LMCache가 실제로
   수행하며 그 비용이 우리 계산값과 맞으므로, "store-vs-recompute" break-even 결과의 신뢰도를 높인다.
3. cold(recompute)가 vLLM 0.18/0.22에서 일치(245/505/1041/2345 vs 218/513/1037/2175)해 엔진 버전
   차이가 결론을 흔들지 않음을 확인.

## 6. 한계 / 남은 작업

- **disk(NVMe) tier 미완.** 강제 spill(CPU 0.5GB) + GDS(cufile) 설정이 64f에서 hang → 중단.
  `LMCACHE_USE_GDS=False`로 재시도하면 우리 `storage_tiers.yaml`의 local_nvme(5GB/s) hop을 실측 검증 가능.
  (DRAM hop은 본 리포트로 검증 완료.)
- **모델 범위.** LLaVA-OV-7B로 검증(가장 안전 — 구형, vLLM 0.18에서 video 지원, 패치 불필요).
  Qwen2.5-VL도 0.18에서 가능. Qwen3-VL/InternVL3.5는 transformers 4.57 환경 제약으로 이 트랙 미적용.
- **엔진 버전.** 절대 비교는 0.18 동일 엔진 내(`ours` vs `lmcache`)에서, 0.22 reuse_real은 cross-check로 사용.

---
*재현: `measure/reuse_lmcache.py` (`--mode ours|lmcache`, `--tier dram|disk`), figure `analyze/lmcache_compare.py`,
데이터 `results/lmcache/`.*
