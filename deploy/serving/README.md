# deploy/serving — 서빙 모델 표와 어댑터

pod-start.sh 가 `serving.sh` 를 source 하고 다음 차례로 부른다.

    serving_detect → serving_select <별칭> <어댑터> → serving_fetch → serving_prepare → serve_lane k port

- **고르기**: 설정 파일(CF)의 `MODEL=<별칭>`·`ADAPTER=auto|hyperqwen-8cf642e|vllm-patched|vllm-stock` 만 받는다.
  - 주소·리비전·파일 sha 는 `models.toml` 과 `pins/` 에서만 온다.
- **이미지**: HyperQwen 8cf642e(`/app/docker/entrypoint.sh` 가 있음) 또는 공식 `vllm/vllm-openai:v0.30.0`.
  - `serving_detect` 가 가리고, `image_info.py` 한 줄을 남긴다: vllm 판, HyperQwen series sha, nvcc.
- **어댑터**

  | 어댑터 | 이미지 | 하는 일 |
  |---|---|---|
  | hyperqwen-8cf642e | HyperQwen | 지금과 같다. 차선 0 은 `entrypoint.sh single`(prepare·verify), 그 밖은 `PREPARE=0 VERIFY=0 start_qwen.sh` |
  | vllm-patched | 둘 다 | 이미지의 vllm 을 `$S/overlay/<묶음>` 으로 복사해 `patches/<묶음>` 을 덧댄 사본으로 `vllm serve`(PYTHONPATH). HyperQwen 이미지에서는 이미지 패치 47개를 거꾸로 풀어 0.30.0 원본을 되살린 뒤 덧댄다 |
  | vllm-stock | 공식 | 고치지 않은 vllm 그대로 |

- **패치 묶음** `patches/vllm-0.30.0`: HyperQwen 8cf642e 패치 12개(NOTICE).
  - `series`: 순서
  - `pristine.sha256`: 덧대기 전 원본 69개 파일
  - `applied.sha256`: 덧댄 뒤 23개 파일
  - `SHA256SUMS`: 묶음 파일들
  - `patchset.py` 가 GNU patch `--fuzz 0` 과 같은 규칙으로 덧대고, 위 표들로 앞뒤를 대조한다. 어긋나면 HALT 한다.
- **모델 표 칸**

  | 칸 | 뜻 |
  |---|---|
  | `repo`·`rev`·`pins` | 받을 저장소·리비전(40자)·파일 sha |
  | `dir` | `$S/models/` 아래 폴더 |
  | `served_name` | `--served-model-name` |
  | `adapters` | 허락하는 어댑터. `auto` 는 이미지에서 되는 첫 것 |
  | `tp` | 카드 수 |
  | `serve.*` | vllm-patched·vllm-stock 의 `vllm serve` 인자 |
  | `hyperqwen.*` | hyperqwen 어댑터의 CTX·SPEC |
  | `fast` | HyperQwen FAST_VARIANT 사본 |
  | `chat_template_fix` | 틀 손질 |

- **새 모델 넣기**
  1. 표에 한 줄을 더한다.
  2. `pins/<별칭>.sha256` 를 만든다(HF LFS oid + 작은 파일은 그 리비전에서 계산).
  3. `uv run python -m unittest tests.test_serving` 을 돌린다.
- **실제 원본 시험**: `STEWART_VLLM_SRC=<v0.30.0 vllm 폴더>`. HyperQwen 이미지 흉내판은 `STEWART_HQ_APP`·`STEWART_HQ_PKG`.

## 표 두 장과 새 칸

- `models.toml`(공개)과 `models.local.toml`(공개하지 않는 줄: edited 별칭·그 계획·보관 서버 주소)을 serving.py 가 합쳐 읽는다. local 이 없어도 돈다.
  - 별칭·계획·이미지는 두 표에서 겹치면 안 된다. default 는 local 것이 이긴다. home·quant 칸은 local 이 더한다(겹치면 안 됨).
  - 공개 표만이면 default = qwen38-27b-orig-w4a16q(변환 전이라 pending — 고르면 오류).
- `pending = true`(home 별칭만): 변환·promote 전. files 를 두지 않고 고를 수 없으며 계획에도 넣지 못한다. promote 가 찍은 files 를 붙이고 이 줄을 뺀다.
- `quant_src = {repo, rev, pins, pad_to, order}`: deploy/quant 변환 대상(pins 는 deploy/quant 기준, pad_to 0 = 0 채우기 없음). `[quant] audit = [원본, 편집본]`.
- `chat_template = {file, sha256}`: 묶음의 templates/ 고정본. serving_prepare 가 sha 를 대조해 모델 폴더의 chat_template.jinja 로 쓰고 tokenizer_config.json 의 chat_template 칸을 뺀다. 손질(translate)은 그 위에 한다.
- 계획 차선 `spec = "mtp" | "off"`: 그 차선만 추측 디코딩을 바꾼다(hyperqwen 은 SPEC env, vllm-* 는 --speculative-config 를 뺀다). 계획 a-spec-ab 가 쓴다.
- 품질 점검은 차선의 추측 디코딩이 mtp 면 GSM8K 만(prompt_logprobs 를 보내지 않는다).
- 감시기 /status 의 lanes[k].spec_decode: 추측 디코딩 누계(drafts·draft_tokens·accepted_tokens·acceptance·mean_accept_len·per_pos). lanes[k].config.spec 은 그 차선이 실제로 쓰는 값.
