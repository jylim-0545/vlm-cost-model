# VLM vision-token / KV 캐싱 — 워크로드 TCO 절감 리포트

**질문.** VLM 비디오 분석에서, 중간 추론 상태(vision token 또는 KV cache)를 *저장*해 두고 매 쿼리마다
재계산(recompute)하는 대신 재사용(reuse)하면 — 단, 이득이 나는 영상만 캐싱하고 나머지는 recompute한다고
할 때 — **전체 추론 TCO 중 실제로 얼마나 절감되는가?**

이 리포트의 **메인은 vision-token reuse (vt_reuse)** 다. KV reuse는 **비교군**으로만 함께 보인다.

**결론 (헤드라인).** 영상별 break-even 판단으로 **vision-token reuse**를 적용하면 4개 모델에서
**TCO ≈13–37 % (local NVMe) / ≈8–31 % (s3 same-region)** 를 절감한다(frame = 128, **batch = 8**, R = median age 50개월).
무엇보다 **vt_reuse의 절감률은 storage tier에 거의 둔감**하다(vision token은 바이트가 작아 retrieval이 싸다).
대조적으로 KV reuse는 빠른 tier에선 더 크게 절감하지만(40–65 %) 느린 tier에선 **붕괴**한다(InternVL-8B 0 %).

![vision-token reuse TCO 절감 (모델 × tier)](results/report/fig1_tco_saving_by_model.png)

---

## 1. 방법

### 1.1 비용 모델 (영상 1개, 보존기간 R 개월, 쿼리율 N = views/month)
모든 비용 primitive는 **실제 vLLM 서빙 경로에서 측정**했다(`measure/reuse_real.py`, H100, CUDA graphs,
median-of-5). **R은 워크로드 age_months의 median(≈50개월)으로 고정**(영상이 평균 살아있는 기간 = 현실적
보존기간; F는 그만큼 amortize, storage rent는 R에 비례). `T_decode`는 세 변종 공통이라 캐싱 판단에서 **상쇄**된다.

```
baseline_TCO      = N·R · (encode + prefill + decode)·gpu_rate                 # 매 쿼리 재계산
cache_saving(v)   = N·R · saving_per_q − F_once − storage_total(∝R)            # 캐싱 시 이득
  saving_per_q    = (front_baseline − front_reuse)·gpu_rate − retrieval_per_q
  F_once          = encode (vt) | encode+prefill (kv)        # 1회 구축비
  retrieval_per_q = storage→DRAM (bytes·egress + bytes/bandwidth·gpu_rate) + DRAM→GPU H2D
break-even rate   N* = (F_once + storage_total) / (R · saving_per_q)
```
- **vt_reuse (메인):** encode만 스킵 → `front_reuse = tok_inject`(prefill은 그대로 수행),
  `bytes = bytes_vt = n_vis·hidden·2` (작음).
- **kv_reuse (비교):** encode+prefill 스킵 → `front_reuse = kv_warm`, `bytes = bytes_kv`
  (vt보다 **8–29× 큼** → 느린 tier에서 retrieval 비용 폭증).

### 1.2 영상별 최적 판단
각 변종에 대해 영상별로 `cache_saving > 0`(= N ≥ N*)이면 캐싱, 아니면 recompute. 워크로드 절감은
`Σ_v max(0, cache_saving) / Σ_v baseline_TCO` (분모는 8.8만 영상 전체의 baseline, decode 포함 → "**전체** TCO 대비 %").

### 1.3 모델당 운영점 1개면 충분한 이유
비용은 **n_vis**(vision-token 수)만의 함수다 — 영상 내용·길이·해상도와 무관(측정: encode/prefill/decode/KV
전부 ∝ n_vis, 해상도 무관 ±10–15 %; [`docs/methodology_single_video.md`](docs/methodology_single_video.md)).
따라서 모델당 측정 운영점 1개로 비용이 특성화되고, N(쿼리율)은 영상별 실측, R은 median age로 고정한다.

---

## 2. 워크로드

`total_vpm.csv` — **88,217개 영상**, 각각 `views_per_month`(N)와 `age_months`(R). VPM은 극단적 heavy-tail:
**median ≈ 1.1 views/month**, max ≈ 6.9 M (소수 인기 head가 전체 쿼리량 지배). age는 **median ≈ 50개월(~4년)**,
max 245개월 — 영상별 보존기간 R로 사용한다.

![워크로드 분포: VPM(좌) + age(우)](results/report/fig2_vpm_ccdf.png)

**가정.** 영상별 n_vis가 없으므로 모든 영상을 하나의 운영점(**frame = 128, batch = 8**)에서 비용 산정했다.
N(쿼리율)은 영상별 실측, R은 median age(50개월)로 고정한다.

---

## 3. 결과

### 3.1 모델별 TCO 절감 — vt_reuse 메인 (frame = 128, **batch = 8**, R = median age 50개월, retrieval 포함, GPU stall)

| model | n_vis | **local vt %** | **s3 vt %** | [비교] local kv % | [비교] s3 kv % |
|---|---|---|---|---|---|
| InternVL-8B  | 32768 | **13.4 %** | **7.5 %**  | 40.1 % | 0.0 % |
| LLaVA-OV-7B  | 25089 | **16.0 %** | **9.2 %**  | 59.9 % | 5.7 % |
| Qwen2.5-VL-7B| 19136 | **36.6 %** | **31.4 %** | 65.3 % | 23.6 % |
| Qwen3-VL-8B  | 14080 | **26.3 %** | **21.4 %** | 42.0 % | 0.0 % |

굵은 값(vt_reuse)이 메인. **vt %는 local과 s3가 비교적 가깝다** — vision token은 바이트가 작아 retrieval이 싸서
tier에 둔감. 비교군 kv %는 local에선 더 크지만(batch↑로 decode collapse → front-end 비중↑ → 31–65 %) s3에선
큰 KV 모델(InternVL)에서 **0 %로 무너진다**(retrieval은 batch로 안 줄어듦).

### 3.2 vt_reuse vs kv_reuse (retrieval 포함) — tier별

**느린 s3:** kv_reuse는 InternVL-8B·Qwen3 **0 %**(큰 KV의 retrieval이 prefill 절약 초과), vt_reuse만 안정적으로
이득. **vt가 유일하게 살아남는 안전한 선택.**

![s3: KV vs vision-token reuse](results/report/fig4_kv_vs_vt_s3.png)

**빠른 local:** retrieval이 싸서 kv_reuse가 prefill까지 회수해 더 크게 절감. vt_reuse는 그보다 작지만 여전히
양(+). (kv는 high-reward·high-risk, vt는 어디서나 안전.)

![local: KV vs vision-token reuse](results/report/fig5_kv_vs_vt_local.png)

### 3.3 영상이 클수록(n_vis ↑) vt 절감 ↑ → plateau

vt_reuse는 **encode만 스킵**(prefill은 그대로 수행)한다. vt 절감%(=encode/cold_full)는 n_vis가 커질수록
오르다가 **~15 %에서 plateau**한다(InternVL-8B, batch=8). 작은 n_vis에선 출력 생성(decode) 비중이 커서 작은
encode를 스킵해봤자 미미하고, n_vis가 커지면 encode·prefill이 decode를 추월하며 vt%가 오른다. 다만 vt가
**스킵 못 하는 prefill이 super-linear**(attention n_vis²)라 결국 분모를 지배 → vt%는 상승을 멈추고 평탄해진다
(아주 큰 n_vis에선 감소; 그 정점은 context 한계 밖).

> **(중요) decode가 n_vis에 비례하는지는 batch에 달렸다.** decode(256 출력토큰)는 매 스텝 모델 weight를
> HBM→SRAM 로드 + KV(n_vis) attention을 한다. **batch=1**에선 weight 로드가 bottleneck(weight-bandwidth-bound)
> → decode가 n_vis에 거의 무관(고정). **batch↑**면 weight 로드가 B개에 amortize되어 **KV(vision token) 읽기가
> bottleneck → decode가 n_vis에 비례**(실측: n_vis 8× 시 decode가 batch=1은 1.2×, batch=16은 3.85× 증가).
> batch=8에선 그 중간이고 encode도 batch로 병렬화돼 함께 작아진다(encode 1004→550 ms). 이 보고서는 batch=8 기준.

> (참고) n_vis에 **super-linear**인 것은 **prefill**(attention n_vis²)인데, 그건 vt가 스킵하지 않고
> **kv_reuse가 스킵**하는 부분이다. 그래서 kv 절감%는 n_vis에 따라 vt보다 가파르게 오른다.

![vision-token reuse: TCO 절감 vs n_vis](results/report/fig3_saving_vs_nvis.png)

### 3.4 retrieval을 제외하면 (= retrieval이 공짜라면)

retrieval을 **완전히 제외**(storage→DRAM 대역폭·egress + DRAM→GPU H2D = 0; storage rent·F는 유지)한 경우.
**§3.2의 retrieval 포함과 대조**하면 retrieval의 역할이 분리된다.

**local NVMe:** retrieval이 원래도 거의 공짜라 §3.2와 거의 같다 — kv_reuse가 prefill까지 회수해 더 큼.

![local: retrieval 제외, KV vs vt](results/report/fig6_retr_local.png)

**s3 same-region:** §3.2에서 **0 %로 죽었던 kv_reuse가 68–76 %로 부활**한다(InternVL-8B 포함). 반면 vt_reuse는
§3.2(retrieval 포함)와 거의 동일(바이트가 작아 retrieval 영향 미미).

![s3: retrieval 제외, KV vs vt](results/report/fig7_retr_s3.png)

→ §3.2(포함) vs §3.4(제외) 대조: **느린 tier에서 KV reuse를 죽이는 건 storage rent가 아니라
retrieval(대역폭)이다.** vt_reuse는 retrieval에 강건해 tier·retrieval 조건과 무관하게 안정적으로 이득.

### 3.5 break-even — N_even / %영상 / %view (s3, retrieval 포함, vt vs kv)

`s3_same_region` + **retrieval 포함**(현실적 조건)에서, 각 변종이 이득이 되는 break-even 쿼리율 **N_even**
(views/month), 그 임계 위 영상이 **전체 영상의 몇 %**, 그 영상들이 **전체 view의 몇 %**인지. vt와 kv를 나란히 비교.
(batch = 8, R = median age 50개월, 전체 88,217개 영상, median vpm ≈ 1.1)

| model | n_vis | vt N_even | vt %영상 | vt %view | kv N_even | kv %영상 | kv %view |
|---|---|---|---|---|---|---|---|
| InternVL-8B  | 32768 | **11.73** | 21 % | 99.7 % | **never** | 0 % | 0.0 % |
| LLaVA-OV-7B  | 25089 | **10.97** | 22 % | 99.7 % | 138.24 | 7 % | 98.5 % |
| Qwen2.5-VL-7B| 19136 | **2.50**  | 38 % | 99.9 % | 26.10 | 15 % | 99.5 % |
| Qwen3-VL-8B  | 14080 | **3.42**  | 34 % | 99.9 % | **never** | 0 % | 0.0 % |

**핵심:**
- **vt_reuse:** N_even ≈ **2.6–12 views/month** — 임계 위 영상이 **전체 view의 ~99.7–99.9 %**를 커버. VPM이
  heavy-tail이라 임계 넘는 인기 영상이 사실상 모든 트래픽을 차지하고, 잘리는 long-tail(median ~1/mo)은 view
  볼륨 ≈0. 즉 **영상의 21–38 %만 캐싱해도 view ~100 % 커버.**
- **kv_reuse:** s3에서 **InternVL-8B·Qwen3는 `never`**(retrieval이 prefill 절약 초과 → 어떤 N에도 이득 없음),
  LLaVA/Qwen2.5도 N_even이 26–138로 높아 캐싱 가능 영상이 7–15 %로 적다. → s3에서 KV는 vt 대비 압도적으로 불리.

---

## 4. 핵심 발견

1. **vision-token reuse(메인)는 워크로드 TCO를 ≈13–37 %(local) / 8–31 %(s3) 절감**하고, **storage tier에
   둔감**하다(vt% local≈s3). 작은 바이트 덕에 retrieval이 싸기 때문 — 어디에 저장하든 안전하게 이득.
2. **KV reuse(비교군)는 high-reward·high-risk:** 빠른 local에선 prefill까지 회수해 더 크게 절감(40–65 %)하지만,
   느린 s3에선 큰 KV의 retrieval이 절약을 초과해 **InternVL-8B에서 0 %로 붕괴**한다.
3. **KV reuse를 죽이는 건 retrieval(대역폭)이다** (§3.4): retrieval을 빼면 s3에서도 KV가 부활(42–56 %)한다.
   storage rent가 아니라 큰 KV를 느린 링크로 가져오는 비용이 문제. KV/token 바이트 비율(8–18×)이 그 비용의 lever.
4. **break-even을 넘는 인기 영상이 트래픽을 지배:** s3에서 vt 캐싱 대상은 영상의 **21–38 %**뿐이지만, heavy-tail
   이라 그 영상들이 **전체 view의 ~99.7 %**를 커버한다. 즉 인기 1/4~1/3만 캐싱해도 사실상 모든 쿼리가 이득을 본다
   (나머지 long-tail은 자동 recompute, view 볼륨 ≈0).

---

## 5. 한계 & 재현

- **단일-n_vis 가정**(§2): 모든 영상을 한 운영점에서 비용 산정. 영상별 duration/해상도가 있으면 per-video
  n_vis로 재가중 가능.
- **R = 영상별 age** (median 50개월). 30일 고정 대비 TCO 절감%는 거의 불변, 캐싱률이 크게 오른다(F amortize).
- **swept tier 2개**(`local_nvme`, `s3_same_region`). egress 과금 인터넷 object store면 N*이 더 오르고,
  `--no-gpu-stall`(retrieval을 compute와 겹침)이면 s3에서도 kv가 부활(§3.4의 retrieval-제외와 유사).
- **decode_tokens = 256** 고정; 판단에서 상쇄되지만 절대 TCO 분모를 스케일.
- 재현: `python -m analyze.tco_report --frame 128` (그림 → `results/report/`), retrieval 제외는
  `--no-retrieval`(그림 `_noretr` 접미사). sweep: `python -m analyze.tco_workload --frame {16,32,64,128} [...]`.
