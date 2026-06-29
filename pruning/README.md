# Vision-token 가지치기 (pruning)

이 cost-model repo에 vision-token **pruning**을 추가한 독립 모듈. 중요한 vision token만
top-k로 남겨 (1) 저장/지연/break-even 비용이 얼마나 줄고 (2) 그래도 정답을 맞추는지 본다.
방법 자체는 공개 기법(FastV류 salience, SparseVLM)이고, 구현은 사내 pruning 연구 코드
(`qaware_vqa.py`)를 포팅했다.

---

## 1. Pruning method

둘 다 post-projector vision token을 attention 점수로 순위 매겨 top-k만 남기고(위치순
재정렬로 시간 순서 보존), **splice**(버린 토큰을 시퀀스에서 제거 — InternVL/LLaVA) 또는
**mask**(attention mask에서 열을 가림 — Qwen3)로 적용한다.

점수는 LLM 디코더의 attention에서 뽑는다. 두 방법은 **어느 층의 attention을 읽고**, **누가
주는 attention을 합산하는지**가 다르다:

| 방법 | 질문 의존 | 신호 | 읽는 층(layer) | 합산 대상(query 토큰) |
|---|---|---|---|---|
| **salience** (query-agnostic) | X | 중립 프롬프트에서 각 vision 토큰이 *받은* attention | 하위층 `2~7` | 전체 토큰 |
| **sparsevlm** (query-aware) | O | 실제 질문에서 **텍스트 토큰**이 각 vision 토큰에 준 attention | 상위층 `12~23` | 텍스트 토큰만 |

- *읽는 층* = 디코더의 어느 레이어들에서 attention 가중치를 꺼내 평균낼지 (하위층은 일반적
  중요도, 상위층은 질문-관련 정렬이 잘 드러나는 경험적 선택).
- *합산 대상* = "각 vision 토큰이 받은 attention"을 어느 query 토큰 기준으로 더할지 — 전체면
  질문과 무관(salience), 텍스트만이면 "질문이 그 토큰을 얼마나 보나"(sparsevlm).

**salience**는 질문과 무관해서 ingest 때 한 번 계산해 저장하면 되고, **sparsevlm**은 질문마다
계산한다. 구현 출처: 사내 pruning 연구의 `qaware_vqa.py`(`attn_to_visual` 외).

---

## 2. 사용법

`Pruner`는 **모델을 자체적으로 로드하는 독립 래퍼**다. 지원되는 3개 모델이면 기존 추론을
`Pruner`로 태우기만 하면 된다(표준 HF `generate`를 감싼 형태 — 추가만 하면 됨). 이미 다른
곳에서 로드해둔 모델 객체에 패치로 얹는 형태는 아니므로, 그 경우엔 저수준 함수
(`prune` / `select_topk` / `answer_splice` / `answer_mask`)를 본인 모델에 맞춰 끼운다.

**모델 선택 = `Pruner(which)`**, **방법 선택 = 어느 `score_*`를 부르느냐**(별도 플래그 없음).
두 방법 모두 같은 `feats`(이미지의 vision 토큰)를 쓰고 점수만 다르다.

```python
from pruning.methods import Pruner, PROMPT
pr    = Pruner("internvl")                # ← 모델 선택 (1회 로드)
inp   = pr.build(img, question + PROMPT)  # 답변에 쓸 질문 입력 (기존 proc(...) 출력도 가능)
feats = pr.feats_of(inp)                  # post-projector 토큰 [n_vis, hid] (이미지에만 의존 → 두 방법 공용)

# 방법 1) salience (query-AGNOSTIC): 중립 프롬프트로 점수
scores = pr.score_salience(pr.build(img, "Describe the image."))
ans    = pr.prune_and_answer(inp, feats, scores, keep=0.25)

# 방법 2) sparsevlm (query-AWARE): 실제 질문으로 점수 (질문마다 다시 계산)
scores = pr.score_sparsevlm(inp)          # inp 에 이미 질문이 들어있음
ans    = pr.prune_and_answer(inp, feats, scores, keep=0.25)
```

추상화 레벨 3단:

- `pr.prune_and_answer(inp, feats, scores, keep)` → 답변 문자열까지
- `pr.prune(feats, scores, keep)` → **가지치기만**: `(kept_feats[k,hid], idx[k])` 반환(저장·전달용)
- `select_topk(scores, k)` → 인덱스만; 적용은 `answer_splice(inp, feats[idx])` / `answer_mask(inp, idx)`

> **query-aware 주의:** sparsevlm은 점수가 질문에 의존하므로 **질문이 담긴 입력으로 매 질문
> `score_sparsevlm`을 호출**해야 한다(중립 프롬프트 X). salience는 한 번 계산하면 모든 질문에 재사용.

### 지원 모델 / 늘리기

현재 `which`는 **`internvl` / `llava15` / `qwen` 셋만** 미리 정의돼 있다. 점수 계산이 모델의
LM attention 모듈(`eager_attention_forward`)을 패치하는 게 유일한 결합점이라, 새 아키텍처는
`methods.py`의 `_MODELS`에 한 줄 추가하면 `Pruner("mymodel")`로 쓸 수 있다(동작 동일, 지원 범위만 확장):

```python
_MODELS["mymodel"] = ("org/My-VLM-HF", "transformers.models.<arch>.modeling_<arch>", (H, W))
# 임베딩 시퀀스에서 토큰 제거가 되면 _SPLICE에 "mymodel" 추가(splice), 아니면 mask 모드
```

`build` / `feats_of`가 그 모델의 입력·`get_image_features` 시그니처와 맞는지만 확인.

---

## 3. 실행 환경

**별도 환경이 필요 없다 — 이 repo의 `vlmcost`/vLLM 환경(transformers 5.9) 하나로 다 돌아간다**
(transformers 4.57.6 / 5.9 양쪽 검증). 필요 패키지는 [`requirements.txt`](requirements.txt) 참고.

- **비용/테스트** (`cost.py`, `demo_cost.py`, `test_methods.py`): GPU·모델 불필요. 이 repo 환경
  그대로(PyYAML + torch + 표준 라이브러리).
- **점수 계산·실모델 데모** (`methods.py`, `demo_latency.py`, `demo_accuracy.py`): GPU 필요.
  점수는 HF transformers의 eager-attention 경로로 뽑는데 **transformers 4.57·5.9 모두 동작**하고,
  5.9는 이 repo에 이미 깔려 있어 그대로 쓰면 된다. (jylim의 측정은 vLLM 엔진에서 돌지만, vLLM 엔진
  자체는 per-token attention을 노출하지 않으므로 점수만 같은 env 안에서 HF 경로로 계산한다.)

`import pruning` / `import pruning.cost`는 torch·transformers를 **불러오지 않는다**(methods가
지연 임포트) → 이 repo 환경에서 비용/테스트 쪽 import는 안전하다.

---

## 4. 구성

모든 파일이 `pruning/` 한 곳에 모여 있다(기존 코드는 건드리지 않음).

```
pruning/
  README.md             # 이 문서
  methods.py            # salience/sparsevlm 점수 + prune + splice/mask + 지연 측정 (실모델, tf4.57)
  cost.py               # keep-ratio → 저장/break-even (순수 산술, GPU 불필요)
  demo_cost.py          # 비용 sweep      → python -m pruning.demo_cost      (GPU X)
  demo_latency.py       # 실측 지연 분해  → python -m pruning.demo_latency   (GPU)
  demo_accuracy.py      # 정확도 sanity   → python -m pruning.demo_accuracy  (GPU)
  test_methods.py       # 단위 테스트     → python -m pruning.test_methods    (GPU X)
  run_pruning_demo.sh   # 단위테스트 + 비용 데모 래퍼 (bash pruning/run_pruning_demo.sh)
```

---

## 5. 데모 & 결과

### 비용 (GPU 불필요)

이 repo의 본래 질문(= vision token을 **저장·재사용 vs 매번 재계산**, 언제 이득인가)에
가지치기를 얹은 것이다. 토큰을 k개로 줄이면 저장 footprint와 break-even **N\*** (몇 회/월
이상 조회돼야 캐시가 이득인가)가 어떻게 내려가는지 본다(bytes는 config에서 정확값, latency는
실측 CSV `--base-csv` 또는 대표값). 즉 이 모듈이 cost-model repo에 직접 기여하는 부분.

```bash
python -m pruning.demo_cost                                          # 대표값
python -m pruning.demo_cost --base-csv results/nextqa/reuse_real.csv  # 실측 latency
```

→ s3_same_region 기준 keep 1.0→0.11에서 저장 **268→30 MB**, break-even **~10 → ~0.3 회/월**
(단조 감소). 더 많은 영상이 캐시 대상이 된다.

### 지연 분해 (실측, GPU)

cuda event로 stage를 재서 비용 모델의 세 변형으로 조합: cold = encode + prefill(full),
vt_reuse = prefill(full)(encode 생략), vt+prune = prefill(keep). n_vis는 기본 video 규모
(`--n-vis 8192` ≈ 32 frame) — 단일 256토큰 이미지는 prefill이 weight-bandwidth bound라 거의
안 줄지만, video 규모에선 prefill이 n_vis에 super-linear라 효과가 보인다. prefill은 실 LLM
forward로 측정(합성 vision embedding — 지연은 내용 무관), encode는 실 1프레임 × frame 수.

```bash
CUDA_VISIBLE_DEVICES=<빈GPU> python -m pruning.demo_latency --which internvl --n-vis 8192
```

실측 (InternVL3.5-8B, n_vis=8192≈32f, 4090):

```
  encode/frame = 17.8 ms  → encode(full 32f) ≈ 571 ms (vt_reuse는 생략)
    keep   n_vis  prefill_ms |  cold TTFT   vt TTFT  vt+prune
    1.00    8192      1122.7 |     1693.9    1122.7    1122.7
    0.50    4096       531.9 |     1693.9    1122.7     531.9
    0.25    2048       249.2 |     1693.9    1122.7     249.2
    0.10     819       103.7 |     1693.9    1122.7     103.7
```

→ vt_reuse는 encode(571ms)를 통째로 건너뛰고, 가지치기는 prefill을 1123→104ms(9%)로 줄인다.

### 정확도 sanity (실측, GPU)

```bash
CUDA_VISIBLE_DEVICES=<빈GPU> python -m pruning.demo_accuracy \
    --which internvl --tsv <TextVQA_VAL.tsv> --bench textvqa --n 30 --keeps 1.0,0.5,0.25,0.1
```

- **데이터셋: TextVQA validation** (`LMUData/TextVQA_VAL.tsv`, VLMEvalKit/LMUData 포맷;
  컬럼 index/image[base64]/question/answer). 앞 n개 샘플 사용.
- **채점: VQA soft-accuracy** (예측이 정답 후보와 일치한 수/3, 최대 1.0 — `score_ans`의 `textvqa`
  분기). GQA는 `--bench gqa`로 normalized exact match.

실측 (InternVL3.5-8B, TextVQA val, n=30):

```
  full (no prune): 68.9%
    keep   salience  sparsevlm
   1.000      68.9%      68.9%
   0.500      68.9%      65.6%
   0.250      60.0%      72.2%
   0.100      36.7%      53.3%
```

→ salience는 keep 0.5까진 무손실, 0.25~0.1에서 저하(68.9→36.7%); **낮은 keep에선 query-aware인
sparsevlm이 salience보다 우위**(0.25: 72 vs 60, 0.1: 53 vs 37) — 예상(query-aware 이점)대로다. n=30은
추세 확인용(약간 노이즈). 이미지 VQA로, 알고리즘 충실성 점검이지 영상 비용 스토리가 아니다.
