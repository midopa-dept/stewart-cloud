# deploy/quant — 우리 4비트 변환 조리법

변환 대상은 서빙 표의 `quant_src`(공개 표 models.toml + models.local.toml)에서 온다. 묶음을 만들 때 `targets.json` 으로 싣고, run-convert.sh 가 order 차례로 한 대여 안에서 이어 돈다.

| order | 별칭(세트 = 별칭-bf16h·-int8h) | 원본 BF16 | 0 채우기 |
|---|---|---|---|
| 1 | qwen38-27b-orig-w4a16q | Qwen/Qwen3.8-27B@37561816e66b5b9153b1b64c10968c18c1e6f4a0 | 없음(17408) |
| 2 | qwen38-27b-demo-w4a16q | example-org/Demo-Qwen3.8-27B@739e3c5b89849f6c238ce1e5b70008612ae42cdd | 없음 |
| 3 | qwen38-27b-demo2-w4a16 | example-org/Demo2-Qwen3.8-27B@7adbeac6b5ade1c13fbed1198a6e95be0475508c | 17536 |

- 조리법(아래)은 셋이 같다. 다른 것은 원본과 0 채우기 여부뿐이다.
- 채팅 템플릿은 표의 고정본(`templates/qwen38-27b-chat.jinja`, sha c3cf9e34…)을 결과에 넣고 tokenizer_config 의 chat_template 칸은 뺀다.
- 감사(`quant.audit` = [원본, 편집본]): 1·2 의 원본이 둘 다 받아지면 `audit_delta.py` 가 층별 ΔW 요약을 낸다(아래). 그 뒤 원본을 지운다.
- 디스크: 원본(55.6 GB)은 그 대상의 양자화가 끝나면 지운다(감사 쌍은 감사 뒤). 대상 셋이면 봉우리가 약 190 GB(demo2 0 채우기 때)라 250 GB 를 쓴다.
- 최대 켜짐: MAX_HOURS 가 없으면 4 × 대상 수 + 2 시간.

## 왜 0 채우기가 먼저인가(demo2)

- demo2 는 층마다 뉴런 하나를 더했습니다.
  - 본체 64층의 MLP 중간 크기는 17409 입니다.
  - MTP 층은 17408 그대로입니다.
  - config 의 `intermediate_size` 는 하나(17409)뿐입니다.
- 그래서 문제가 둘 생깁니다.
  - int4 그룹(128)과 Marlin 의 나눗셈 조건에 맞지 않습니다.
  - MTP 를 읽을 때 모양이 안 맞습니다.
- 셋 다 17536(128×137)으로 0 을 채웁니다.
  - gate·up 의 새 행 = 0, down 의 새 열 = 0 → 새 뉴런의 출력은 정확히 0 입니다.
  - 실제 머리로 센 대상: 본체 192 + MTP 3 = 195 텐서, +0.25 GB.

## 차례(run-convert.sh, 단계마다 표시 파일 — 다시 켜면 이어서. 대상 단계는 done/<별칭>/…)

| 단계 | 하는 일 | 멈추는 조건 |
|---|---|---|
| 0 | 이미지 /app 파일(HyperQwen 8cf642e)·묶음 sha 대조 | 다르면 |
| 1 | auto-round 0.14.2(휠 sha 대조)·accelerate 1.14.0·datasets 5.0.1·py-cpuinfo 9.0.0·pydantic 2.13.4 설치. 이미지 패키지 판은 constraints 로 묶음 | 이미지 판이 하나라도 바뀌면 |
| 3 | pile-10k@127bfed·품질 자료 3개(sha 대조), 그 뒤 오프라인. auto-round 의 load_dataset 이 오프라인으로 읽는지 먼저 확인 | 실패하면 |
| 대상마다 2 | 원본 BF16 받기(리비전 고정, pins 대조). 받자마자 감사 쌍이 갖춰졌으면 감사 | sha 대조 실패 |
| 대상마다 4·5 | (pad_to 가 있을 때만) `pad_intermediate.py --to <pad_to>`, 전후 KL(전체 어휘, wikitext-2 8×256) | 평균 KL > 1e-4 또는 top-1 < 99.9% |
| 대상마다 6 | `quantize_autoround.py`(아래 조리법) → `graft_missing.py`(mtp.*·비전을 BF16 로) → 채팅 템플릿 고정본 | 빠진 모듈이 남으면 |
| 대상마다 7 | 양자화 전후 KL(숫자만) | — |
| 대상마다 8 | int8 head 판: HyperQwen `docker/prepare.sh`(FAST_VARIANT=0 DFLASH2=0) — lm_head·임베딩·MTP int8 + draft head 40,960 | — |
| 대상마다 9 | `verify.sh --no-server`, `start_qwen.sh`(CTX=long SPEC=mtp PREFIX_CACHE=1), `quality_battery.py --gsm-only`(mtp 서빙에 prompt_logprobs 를 보내지 않는다 — PPL 쪽은 7단계 KL), /metrics 의 추측 디코딩 수락 누계(spec.json) | verify FAIL·서버 안 섬·품질 실패 |
| 대상마다 10 | 두 결과의 파일 sha 목록(기록용) | — |
| 11 | `build_export.py` → export.json(대상마다 세트 둘·보고, 감사 요약) | 끝난 대상이 없으면 |
| 끝 | 보관 서버가 :8090 에서 끌어가고 `POST /export/done` 을 보내면 정지. 120분 안에 안 오면 정지. HALT 는 이유 한 줄 → 900초 → 정지 | — |

## 감사(audit_delta.py, CPU·numpy)

- 원본·편집본 BF16 의 같은 이름 텐서를 바이트로 견준다. 같으면 계산하지 않는다.
- 바뀐 텐서마다: 층·종류·모양·상대 크기 ‖ΔW‖/‖W‖·최대 절댓값, 2차원이면 특잇값 에너지 비율 e1·e2·e4·e8·e16·e32(무작위 저차 SVD, 씨앗 0)와 r90·r99.
- 요약: 바뀐 층 구간·처음·끝, 탑별(본체·MTP·비전·임베딩·lm_head·그 밖) 바뀐 수, 행렬 종류별 바뀐 수.
- 출력은 audit-delta.json·.tsv 의 요약 수치뿐이고 export.json 의 audit 에 실린다. ΔW·특이 벡터 같은 텐서 값은 어디에도 쓰지 않는다.

- 처음에 테일넷(`stewart-quant`, tag:gpu, 파드와 같은 키 방식)에 들어온다. 그리고 `quantsrv.py` 를 띄운다.
  - `:8001`: `/status`·`/logout`(serve·eval 의 감시기와 같은 자리)
  - `:8090`: `/export`·`/export/<별칭>/<이름>`·`/export/done` — 보관 서버(`home.json` 의 server_ip, 서빙 표에서 옴)만
- 이 인스턴스 → 집 연결은 없다. HF 토큰·사람 입력도 없다.

## 조리법(바꾸지 않는다, 모델 A = ababaka 카드와 같은 판)

- 양자화 설정

  | 칸 | 값 |
  |---|---|
  | 도구 | auto-round 0.14.2, transformers 5.15.0, torch 2.13+cu130, compressed-tensors 0.17.0(HyperQwen 8cf642e 이미지) |
  | `AutoRound(...)` | scheme W4A16(int4·g128·대칭), dataset NeelNanda/pile-10k, nsamples 128, seqlen 2048, batch_size 4, iters 200, seed 42, device_map 0, trust_remote_code True, quant_nontext_module False |
  | BF16 로 남김 | layer_config bits=16 은 `linear_attn.in_proj_a/b`. lm_head·비전·MTP 도 BF16 |
  | 저장 | `quantize_and_save(format="llm_compressor")` |

- 결과 두 벌(HF 저장소 커밋 둘)
  1. **bf16heads**: 공식 vLLM 그대로 읽힌다. 몸통 KL 의 기준이다.
  2. **int8heads**: 모델 A 와 같은 모양이라 3090 한 장에서 150k 로 hyperqwen·vllm-patched 두 어댑터에 바로 선다. 서빙 표의 별칭(pending 줄)은 이 세트를 promote 한 것이다.

## 결과 받기(집)

`deploy/modelstore/README.md` 의 흐름을 따른다: `sudo modelstore pull stewart-quant…` → `sudo modelstore promote … <별칭>`.

## 셸 비상구와 손으로 이어 가기(v13)

- 이 인스턴스는 `tailscale up --ssh` 로 붙는다. 집(테일넷 ACL 의 ssh 규칙이 허락한 쪽 — 보관 서버 tag:server 등)에서
  `tailscale ssh root@<로그의 tailscale name 줄 이름>` 으로 root 셸에 들어온다. 키 파일은 없다(테일넷 신원).
  이 인스턴스에서 집으로 여는 연결은 그대로 없다(ACL 에 tag:gpu 를 src 로 하는 허락이 없다).
  `--ssh` 가 안 되면 로그에 "tailscale up --ssh 실패 — 셸 비상구 없이 붙는다" 를 남기고 그냥 붙는다.
- HALT 했을 때(열린 세트가 없고 테일넷이 붙은 뒤): 켜진 채 `HALT_HOLD_MIN`(60)분 사람을 기다린다. 그동안
  - `touch /workspace/quant/hold` — 지울 때까지 더 기다린다(HALT 부터 `HALT_HOLD_MAX_MIN` 360분 상한, A100 1.7 $/hr 로 약 $10)
  - `touch /workspace/quant/rerun` — run-convert.sh 를 처음부터 다시(단계 표시로 이어서, 고친 `files/` 로). 테일넷·quantsrv 는 그대로 쓴다
  - `touch /workspace/quant/stop-now` — 바로 정지
  - export.json 이 생기면(아래 손 내보내기) 받기로 넘어가 집이 다 받거나 받는 활동 없이 120분이면 정지
- 손 절차는 묶음 안 스크립트로 친다(v13.1 — 시험대 tests/test_run_convert.py TV131 에서 그대로 돌린 것):
  - 손 내보내기: `bash /workspace/quant/files/manual-export.sh` — 받은 세트를 10단계 sha 로 대조(VERIFY-OK) → `$Q/t` 로 옮김(MOVED)
    → 단계 표시 → export.json → 맨 끝 `EXPORT-OK`(앞이 다 지났을 때만). 하나라도 실패하면 그 자리에서 멈추고 아무것도 덮지 않는다.
  - 손으로 이어 가기: `bash /workspace/quant/files/manual-continue.sh` — 지금 도는 run-convert 를 멈춰 두고(STOP, 끄지 않음 —
    끄면 부모 boot 가 900초 뒤 장비를 끈다) 같은 컨테이너에서 새 판을 띄운다. 자기·부모 셸은 빼고 멈춘다(pkill -f 가 ssh 셸까지 멈춘 일이 있었다).
    새 판 출력은 컨테이너 로그로 간다(래퍼 logs 로 보인다). 표지 `STOPPED <pid…>` → `STARTED <pid>` → `CONTINUE-OK`.
  - 집에서 들어가는 줄: `ssh root@<이름>.<테일넷>.ts.net`(Tailscale SSH — ACL 의 ssh 규칙이 허락한 쪽만).
- 복사 받기 판정(v13.1): 복사 중에는 `/workspace/import` 안을 이름으로 찾지 않는다(목록만 읽는다). 실제로 판정이 없는
  done 을 이름으로 찾아 둔 사이 호스트 쪽 rsync(copy_direct)가 done 을 만들자 겹친 파일 시스템이 "없음"을 기억해 ESTALE 이 됐다(짐작).
  판정이 막히면 10분 진행 줄에 첫 이유("판정: 없음: done/…", "ESTALE …")를 찍는다. ESTALE 은 컨테이너를 다시 켜면(stop → start) 풀린다.
- 테일넷 길(v13.1): 템플릿이 `-p 70000:70000/udp` 를 열고, tailscaled 를 Vast 가 준 `VAST_UDP_PORT_70000` 로 띄워 `PUBLIC_IPADDR:그 포트`
  를 광고한다(장비는 대칭 NAT). 부팅 1분 뒤 "tailscale 길(보관 서버 …): … direct|relay" 한 줄. 직결이 중계보다 느리면(먼 지역 사이)
  `touch /workspace/quant/ts-relay` 뒤 컨테이너를 다시 켜면 무작위 포트(중계)로 붙는다.
- 받기(modelstore pull, v13.1): `.part` 가 있으면 그 뒤부터 이어 받는다(quantsrv 가 Range 를 받는다). 평균 속도 문턱은 없애고
  (min_rate_Bps 0), 읽기가 stall_s(300)초 동안 0 바이트면 끊는다 — 다시 치면 잇는다. 파일마다 "처음 30초 N MB/s" 한 줄.
- 위험과 좁힌 것: 들어오는 쪽이 root 셸을 얻는다. 장비에 있는 것은 변환 결과·HF 공개 모델·이 인스턴스 범위의 Vast 키
  (/dev/shm/vast-key — 이 인스턴스 정지·삭제)·테일넷 노드 상태다. 쓰기 토큰·집의 비밀은 없다.
  들어올 수 있는 쪽은 ACL 의 ssh 규칙(src)만이다. 보관 서버 테일넷 신원이 새면 그 쪽이 빌린 장비에 root 로 들어올 수 있다
  — 장비에서 집으로는 여전히 못 나온다.
- 정의 차례 시험: `tests/shell_order.py`(함수·전역 변수가 쓰이는 줄보다 앞에서 정의되는지, trap 문자열 포함) 와
  `tests/test_run_convert.py`(run-convert.sh 실물을 처음부터 끝까지, 집 쪽은 modelpush pull·promote 실물).

## 묶음

`uv run python deploy/quant/make_quant_bundle.py --out <폴더>` 로 만듭니다.

- `quant-files.b64` 는 gist 에 올립니다.
- Docker Options 는 `docker-options.txt` 그대로(`-e PU=<raw> -e PS=<sha> -e SB=<boot>`)이고, `PU` 칸만 사람이 넣습니다. Args 는 `args.txt` 그대로입니다.
- 기동: Args 가 SB(`boot-quant.sh` 를 주석만 뺀 판의 base64)를 풀어 sha256 을 대조한 뒤에만 돌립니다.
  `boot-quant.sh` 가 받기·대조·풀기를 하고 `run-convert.sh` 를 부릅니다. PU·PS 없음, 받기 실패, sha 불일치, 풀기 실패,
  변환 단계 HALT, run-convert 가 돌아옴 — 모두 이유 한 줄 → 900초 → 스스로 정지(CONTAINER_API_KEY 는 헤더 파일로, 명령 줄에 없음).
  SB 가 없거나 sha 가 다르면 Args 는 `HALT: boot SB` 를 남기고 잡니다(256자에 정지 요청을 넣을 수 없음 — 파드와 같음).
- 이미지는 HyperQwen 8cf642e(digest 고정), 디스크는 ≥ 250 GB(대상 셋)입니다.
