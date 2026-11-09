# stewart-cloud

**Rent a GPU on Vast.ai, serve an open model with vLLM, and reach it only over your own Tailscale network — no public ports.** stewart-cloud is the command-line wrapper (find, rent, watch, log, destroy; rebuild-and-verify the pod bundle; publish it to a private gist and a pinned Vast template) plus the pod bundle that runs on the instance: it joins the tailnet, downloads a model pinned by revision and per-file sha256, starts vLLM through one of two image adapters with a checked vLLM 0.30.0 patch set, and stops itself when idle or on any boot error. Optional parts push models from your own server to the pod and convert BF16 checkpoints to AutoRound W4A16 with KL checks. macOS only (keys live in the login keychain); no model weights are distributed. The rest of this README is in Korean.

Vast.ai 에서 GPU 를 빌려 열린 모델을 vLLM 으로 띄우고, 공개 포트 없이 내 Tailscale 망으로만 [stewart](https://github.com/midopa-dept/stewart) 에 붙이는 도구 모음입니다. 처음 쓰는 사람이 계정 준비부터 빌리기·모델 올리기·붙이기·끄기까지 따라 할 수 있게 썼습니다.

- **`stewart_cloud/`**: 명령줄 도구 `stewart-cloud`입니다. offer 를 찾고, 빌리고, 보고, 로그를 읽고, 지웁니다. 파드 묶음을 다시 만들어 대조(verify)하고, gist 와 Vast 템플릿으로 올립니다(publish).
- **`deploy/vast/`**: 인스턴스가 켜질 때 도는 파드 묶음입니다. 테일넷에 붙고, 고정한 모델을 받고, vLLM 을 띄우고, 놀면 스스로 멈춥니다.
- **`deploy/serving/`**: 모델 표, 이미지 두 가지(HyperQwen·공식 vLLM)용 어댑터, vLLM 0.30.0 패치 묶음과 그 검사기입니다.
- **`deploy/modelstore/`**(선택): 내 서버의 모델 파일을 테일넷으로 파드에 밀어 넣습니다. 파드는 내 쪽으로 연결을 열지 못합니다.
- **`deploy/quant/`**(선택): 빌린 A100·H100 에서 BF16 체크포인트를 AutoRound W4A16 으로 바꾸고, 바꾸기 앞뒤를 KL 로 잽니다.

이 저장소는 모델 가중치를 배포하지 않습니다. 공개 모델 표에는 원본 Qwen3.8-27B 의 변환 줄 하나만 있고, 그 줄은 변환 전이라 아직 고를 수 없습니다(아래 "모델 올리기").

## 읽기 전에

- **macOS 전용입니다.** 키는 로그인 키체인(`/usr/bin/security`)에서 읽고, verify 는 `sandbox-exec` 안에서 빌더를 돌립니다.
- **돈이 듭니다.** `up` 은 시간 단위 과금을 시작합니다. 정지한 인스턴스도 지우기 전까지 디스크 요금이 나갑니다.
- 코드 주석과 출력은 한국어입니다.
- 한 사람이 자기 계정으로 쓰는 것을 전제로 했습니다. 여러 사람이 한 Vast 계정을 같이 쓰는 경우는 시험하지 않았습니다.

## 짜임

```
내 Mac ──(Tailscale, tag:gpu 포트만)──▶ 빌린 GPU(Vast.ai)
  stewart chat ─────────────────────────▶ ├─ tailscaled(사용자 공간 모드, 공개 포트 없음)
  stewart-cloud up/status/down ─(Vast API)▶├─ vLLM 127.0.0.1:8000(eval 은 차선마다 :80k0)
                                           ├─ 감시기 127.0.0.1:8001(/status, 쉬면 정지)
                                           └─ 리비전 + sha256 으로 고정한 모델 파일
```

1. 템플릿의 짧은 Args 가 Docker Options 의 `SB`(boot.sh)를 풀고 sha256 을 대조합니다.
2. boot.sh 가 비밀 gist(`PU`)에서 묶음을 받아 `PS`(sha256)와 대조한 뒤 풉니다.
3. pod-start.sh 가 Tailscale 에 붙고, 모델 표에서 고른 모델을 받아 vLLM 을 띄웁니다.
4. 어느 단계든 실패하면 까닭 한 줄을 남기고 900초 기다린 뒤 스스로 정지합니다(HALT).

`up` 은 지금 템플릿을 API 로 다시 읽어 `PS`·`SB`·Args 가 publish 기록과 같을 때만 빌립니다.

## 준비물

| 무엇 | 쓰임 |
|---|---|
| macOS, `/usr/bin/python3`(Command Line Tools) | 도구 실행기. 표준 라이브러리만 씁니다 |
| [uv](https://docs.astral.sh/uv/) 와 CPython 3.13 | 시험, 묶음 빌더, verify 실행기(`py313`) |
| 선불 크레딧이 있는 Vast.ai 계정 | GPU 대여. 자동 충전은 끕니다 |
| Vast API 키 두 개 | 빌리기용(misc·instance_read·instance_write), 템플릿용(user_read·user_write) |
| Tailscale 계정 | 나와 파드 사이의 사설망 |
| GitHub 계정과 fine-grained 토큰 | 묶음을 실어 나를 비밀 gist. 토큰 권한은 Gists read/write 하나 |

## 1. Vast.ai 계정

1. 계정을 만들고 크레딧을 넣습니다. **자동 충전은 끕니다.** 그러면 지출 상한이 잔액이 됩니다. 이 도구는 잔액을 보지 않습니다.
2. API 키를 둘 만듭니다.
   - 빌리기용: misc, instance_read, instance_write. billing·machine·team 은 주지 않습니다.
   - 템플릿용: user_read, user_write. user_write 는 계정 env·키 만들기와 같은 범주라서 빌리기용과 나눠 둡니다. 템플릿용 키가 없으면 빌리기용 키로 템플릿 API 를 부르므로, 그 키에 두 권한이 있어야 합니다.
3. 계정 설정의 환경 변수에 `TS_AUTHKEY` 를 넣습니다(다음 절에서 만듭니다). 이 값은 저장소·템플릿·명령줄 어디에도 들어가지 않습니다. 파드는 다른 프로세스를 띄우기 전에 이 값을 `/dev/shm` 의 600 파일로 옮기고 환경에서 지웁니다.

## 2. Tailscale

1. ACL 정책에 태그 `tag:gpu` 를 만듭니다.
   - 내 기기에서 `tag:gpu` 의 서빙 포트(8000, eval 은 8000~8071 의 차선 포트, 상태 8001)로만 가게 합니다.
   - `tag:gpu` 에서는 아무 데로도 못 가게 합니다. 모델 보관소를 쓰면 그 서버의 받는 포트만 엽니다.
2. 그 태그로 인증 키를 만듭니다. 재사용 가능·임시(ephemeral)·사전 승인으로 만듭니다.
3. 파드의 이름은 역할로 정해집니다.

   | 역할 | 테일넷 이름 | 쓰임 |
   |---|---|---|
   | serve | `stewart-gpu` | 사람이 쓰는 서빙 1장 |
   | eval·eval2 | `stewart-gpu-eval`·`stewart-gpu-eval2` | 평가용 여러 장, 차선 계획 |
   | quant | `stewart-quant` | 4비트 변환 |

   설정 파일에는 `<이름>.<내 테일넷>.ts.net` 꼴로 적습니다.

## 3. gist 두 개

1. 비밀 gist 를 하나 만듭니다. publish 가 여기에 묶음 파일(`stewart-pod-vN.b64`·`stewart-quant-vN.b64`)을 더합니다.
2. 비밀 gist 를 하나 더 만들어 설정 파일(CF)을 둡니다. 파드는 허용 목록의 `KEY=VALUE` 만 읽고, 나머지 줄은 이름만 남기고 버립니다.

   ```
   CTX=long
   SPEC=mtp
   PREFIX_CACHE=1
   QUALITY=auto
   MODEL=<모델 표의 별칭>
   ADAPTER=auto
   ```

   허용 값: `CTX` fast|long, `SPEC` mtp, `PREFIX_CACHE` 0|1, `FAST_VARIANT` 0|1, `QUALITY` auto|0|1, `ADAPTER` auto|hyperqwen-8cf642e|vllm-patched|vllm-stock, `MODEL`·`PLAN` 은 소문자·숫자·`.`·`-` 48자까지.
3. GitHub fine-grained 토큰을 Account permissions → Gists read/write 하나로 만듭니다.

## 4. 설치

설치 폴더는 root 소유여야 합니다. 그래야 같은 사용자로 도는 다른 프로그램(에이전트 포함)이 본체·설정·상태를 고치지 못합니다. 기본 위치는 `/opt/stewart-cloud` 입니다(`stewart_cloud.py` 의 `PREFIX`).

```bash
git clone https://github.com/midopa-dept/stewart-cloud ~/src/stewart-cloud
cd ~/src/stewart-cloud
uv run --no-project --python 3.13 python -m unittest discover -s tests -t .
```

시험이 다 통과하면 설치합니다(관리자 비밀번호를 묻습니다).

```bash
sudo install -d -o root -g wheel -m 755 /opt/stewart-cloud /opt/stewart-cloud/bin
sudo install -o root -g wheel -m 644 stewart_cloud/stewart_cloud.py /opt/stewart-cloud/stewart_cloud.py
sudo install -o root -g wheel -m 755 stewart_cloud/stewart-cloud /opt/stewart-cloud/bin/stewart-cloud
sudo install -o root -g wheel -m 644 stewart_cloud/config.example.json /opt/stewart-cloud/config.json
sudo install -d -o "$USER" -g staff -m 700 /opt/stewart-cloud/state
```

- `config.json` 의 `ts_name` 넷을 내 테일넷 이름으로 고치고, 역할마다 `max_dph`(시간당 값 상한)와 `disk` 를 봅니다. 코드에도 넘을 수 없는 상한이 있습니다(인스턴스 하나 8.0 $/hr, 8장).
- `site.json`(선택): 키체인 계정·홈·기록 폴더·저장소·묶음 폴더를 바꿀 때만 둡니다. 없으면 부른 사용자와 그 홈 아래 기본 경로를 씁니다. 견본은 `stewart_cloud/site.example.json` 입니다.
- verify·publish 의 빌더 실행기: uv 의 CPython 3.13 을 `/opt/stewart-cloud/py313/` 로 복사하고 root 소유·go-w 로 둡니다. 묶음 재현은 3.13 이어야 합니다. 3.9 는 `ast.unparse` 결과가 달라 sha 가 바뀝니다.

## 5. 키 준비

키는 로그인 키체인의 generic password 로 둡니다. 계정은 내 사용자 이름이고, 접근 허용에 `/usr/bin/security` 를 넣습니다. `-w` 를 값 없이 주면 값을 묻습니다(화면에 안 보임).

```bash
security add-generic-password -s stewart-cloud.vast-api -a "$USER" -T /usr/bin/security -w
security add-generic-password -s stewart-cloud.vast-template -a "$USER" -T /usr/bin/security -w
security add-generic-password -s stewart-cloud.github-gist -a "$USER" -T /usr/bin/security -w
```

확인합니다. `stewart- 인스턴스 0개` 가 나오면 키가 읽히고 맞습니다.

```bash
/opt/stewart-cloud/bin/stewart-cloud status
```

- 키는 환경 변수·명령줄·파일로 넘기지 않습니다. 본체가 `security` 를 직접 부르고, 출력은 파이프로만 받고, HTTP Authorization 헤더에만 넣습니다.
- SSH 셸에서는 키체인이 잠겨 있어 종료 코드 78 이 납니다. 먼저 `security unlock-keychain ~/Library/Keychains/login.keychain-db` 를 칩니다.

## 6. 첫 템플릿

publish 는 지금 템플릿(바탕)에서 이미지·디스크·Args 꼴을 베껴 새 템플릿을 만듭니다. 그래서 **첫 템플릿 하나는 콘솔에서 손으로 만듭니다.**

1. 묶음을 만듭니다.

   ```bash
   uv run --no-project --python 3.13 python deploy/vast/make_template_env.py --out ~/.local/share/stewart-cloud/bundles/v1
   ```

   폴더에 `pod-files.b64`·`args.txt`·`boot.sb.txt`·`docker-options.txt` 가 생깁니다. 원본이 같으면 바이트도 같습니다.
2. `pod-files.b64` 를 1번 gist 에 올리고, 리비전을 고정한 raw 주소를 얻습니다.
3. Vast 콘솔에서 비공개 템플릿을 만듭니다.
   - 이미지: `deploy/serving/models.toml` 의 `[images.*].ref`(digest 고정).
   - 실행 방식: Entrypoint, Args 는 `args.txt` 그대로.
   - Docker Options: `docker-options.txt` 의 빈칸 `<gist raw 주소>`·`<설정 gist 최신 raw 주소>` 를 채운 것.
4. 상태 파일을 씨앗으로 넣습니다. 견본은 `stewart_cloud/state-seed.example.json` 입니다. 빈칸은 템플릿 hash_id, `PS`, `boot.sb.txt`·`args.txt` 의 sha256 입니다. 그다음 `stewart-cloud templates` 가 `ok` 를 보이는지 봅니다.

   ```bash
   cp stewart_cloud/state-seed.example.json /opt/stewart-cloud/state/templates.json   # 빈칸을 채운 뒤
   chmod 600 /opt/stewart-cloud/state/templates.json
   /opt/stewart-cloud/bin/stewart-cloud templates
   ```

quant 를 쓰면 `deploy/quant/make_quant_bundle.py` 로 같은 일을 한 번 더 하고 `current.quant` 를 더합니다.

## 7. 묶음 바꾸기: verify·publish

`deploy/` 를 고쳐 커밋한 뒤 새 묶음을 만듭니다(`v2` 처럼 없는 폴더에).

```bash
uv run --no-project --python 3.13 python deploy/vast/make_template_env.py --out ~/.local/share/stewart-cloud/bundles/v2
/opt/stewart-cloud/bin/stewart-cloud verify v2 <커밋 40자> <PS 64자>
/opt/stewart-cloud/bin/stewart-cloud publish v2 <커밋 40자> <PS 64자>
```

- **verify**(키·망 없음): 저장소의 git 객체를 직접 읽습니다. 객체마다 SHA-1 을 다시 계산하고, `.git/config` 의 filter 같은 설정은 거치지 않습니다. `deploy/` 를 깨끗한 임시 폴더에 꺼내 망이 없는 `sandbox-exec` 안에서 빌더를 돌립니다. 결과가 묶음 폴더와 바이트로 같고 PS 가 인자와 같을 때만 통과합니다.
- **publish**: verify 를 다시 합니다. 그다음 gist 에 묶음을 더하고, 리비전 고정 주소를 토큰 없이 받아 PS 와 대조합니다. 새 템플릿을 만들어 다시 읽고 모든 칸이 맞을 때만 상태 파일의 지금 템플릿을 바꿉니다.
- 되돌리기는 옛 묶음을 다시 publish 하는 것입니다.

## 8. 빌리기

```bash
/opt/stewart-cloud/bin/stewart-cloud offers serve
/opt/stewart-cloud/bin/stewart-cloud up serve <offer_id> <시간당 최대 달러>
/opt/stewart-cloud/bin/stewart-cloud status
/opt/stewart-cloud/bin/stewart-cloud logs <instance_id>
```

- `offers` 는 조건에 맞는 offer 를 장당 값 싼 순으로 보이고, 마지막 줄에 그대로 칠 `up` 명령을 보입니다.
- `up` 은 같은 검색을 다시 불러 offer 와 값을 다시 봅니다. 역할마다 하나만 빌리고, 그 역할의 테일넷 이름이 아직 남아 있으면 빌리지 않습니다. 같은 역할의 정지한 인스턴스는 먼저 지웁니다.
- `status` 는 켜진 시간, 시간당 값으로 셈한 지금까지의 값, 쉬다 정지까지 남은 시간, 모델 받기 진행, 실패한 차선·HALT 줄을 보입니다.
- 평가 역할은 `up eval <offer_id> <상한> <계획>` 으로 모델 표의 `plans` 를 고릅니다.

## 9. 모델 올리기

파드는 CF 의 `MODEL=<별칭>` 으로 모델 표의 줄 하나를 고릅니다. 주소·리비전·파일 sha 는 묶음 안의 표와 `pins/` 에서만 옵니다. CF 로는 별칭과 어댑터 이름만 고를 수 있습니다.

- **Hugging Face 에서 받기**(`source = "hf"`)
  1. `deploy/serving/models.toml` 이나 같은 폴더의 `models.local.toml`(공개하지 않을 줄, 있으면 합쳐 읽음)에 줄을 더합니다. `repo`, 40자 `rev`, `pins`, `arch`, `served_name`, `adapters`, `[….serve]` 의 vLLM 인자를 적습니다. 칸 뜻은 `deploy/serving/README.md` 에 있습니다.
  2. `pins/<별칭>.sha256` 에 그 리비전의 파일마다 `sha256  파일 이름` 한 줄씩 적습니다.
  3. 묶음을 다시 만들어 publish 하고, CF 에 `MODEL=<별칭>` 을 적습니다.
- **내 서버에서 밀어 넣기**(`source = "home"`, `deploy/modelstore/`): 파드가 받는 창구를 열고 기다리면, 내 서버의 데몬이 테일넷으로 파일을 보냅니다. 설치와 권한 구조는 `deploy/modelstore/README.md` 에 있습니다.
- **직접 4비트로 바꾸기**(`deploy/quant/`): 표의 `quant_src` 대상을 quant 역할 인스턴스에서 차례로 변환합니다. 끝나면 모델 보관소가 결과를 끌어갑니다. 차례와 검사는 `deploy/quant/README.md` 에 있습니다.

## 10. stewart 붙이기

stewart 의 `backends.toml` 에 백엔드를 하나 더합니다.

```toml
[backends.cloud]
description = "Vast GPU vLLM (테일넷 stewart-gpu)"
base_url = "http://stewart-gpu.<내 테일넷>.ts.net:8000/v1"
model = "<표의 served_name>"
max_context = 150000          # 표의 max_model_len 과 맞춘다
concurrency = 8               # 표의 max_num_seqs 와 맞춘다
enabled = true
timeout_s = 240
```

- vLLM 키는 두지 않습니다. 경계는 Tailscale ACL 입니다.
- `stewart cloud wait` 는 인스턴스가 서빙을 시작할 때까지 기다리고, `stewart ping -b cloud` 로 연결을 확인합니다.
- `stewart chat` 이 이 백엔드를 쓰게 하려면 프로필에 `backend = "cloud"` 를 적거나 `backends.toml` 의 `default` 를 `"cloud"` 로 둡니다.
- 파드가 없으면 연결 실패가 나는 것이 정상입니다.

## 11. 끄기

```bash
/opt/stewart-cloud/bin/stewart-cloud down <instance_id>
```

정지가 아니라 **Destroy** 입니다. 먼저 파드의 `POST :8001/logout` 을 불러 테일넷에서 빼고, 디스크까지 지우고, 목록에서 사라졌는지 확인합니다.

파드가 스스로 정지하는 때는 다음과 같습니다. 정지 뒤에도 디스크 요금이 나가므로 `down` 을 합니다.

| 조건 | 기본값 | 바꾸는 환경 변수 |
|---|---|---|
| 요청·토큰 카운터가 그대로이고 처리 중·대기 요청이 0 | serve 90분, 평가 30분 | `IDLE_MIN` |
| 서버가 아직 안 뜸(첫 부팅). 모델 보관소에서 받는 동안은 세지 않음 | 45분 | `BOOT_GRACE_MIN` |
| 켜진 시간 | 8시간(quant 는 4 × 대상 수 + 2시간) | `MAX_HOURS` |
| 부팅 중 오류(HALT) | 까닭 한 줄 → 900초 → 정지 | — |

## 비용 어림

```
대여비 ≈ offer 의 dph_total($/hr) × 켜진 시간(hr)
정지 중 ≈ 디스크 요금 × 정지해 둔 시간          (down 하면 0)
합계   ≈ Σ 대여비 + 정지 중 + 망 요금(offer 마다 다름)
```

- `dph_total` 은 GPU·CPU·디스크를 합친 시간당 값입니다. `status` 의 usd 칸은 `dph_total × 켜진 시간`으로 셈한 어림이고, 정지 중 디스크 요금과 망 요금은 빠져 있습니다.
- 아래 값은 예시입니다. offer 값은 시각마다 바뀝니다.

| 예 | 계산 | 어림 |
|---|---|---|
| serve, RTX 3090 1장, 저녁 3시간 | 0.30 × 3 | 약 $0.90 |
| eval, RTX 3090 8장, 2시간 | (8 × 0.30) × 2 | 약 $4.80 |
| quant, A100 80GB 1장, 대상 셋(약 7.5시간) | 1.27 × 7.5 | 약 $9.5 |

- quant 행의 시간은 우리 실측(원본 받기부터 양자화 끝까지 대상 하나에 약 1.8시간)에 나머지 단계를 더한 어림입니다.

## 알려진 문제

- **offer 번호가 바뀝니다.** 같은 기계가 짧은 사이에 다른 offer id 로 다시 나옵니다. `offers` 에서 본 id 로 `up` 하면 "목록에 없다"(69)로 거부될 수 있습니다. 이때는 `offers` 를 다시 보고 바로 `up` 합니다.
- **Vast API 시간 초과**(75)가 가끔 납니다. 만들기 요청이 들어갔는지 모르니, 다시 치기 전에 `status` 로 인스턴스가 생겼는지 봅니다.
- **테일넷 이름 겹침**: 이전 노드가 남아 있으면 새 노드가 `stewart-gpu-1` 이 됩니다. 그래서 `up` 은 이름이 풀리는 동안 거부합니다. `down` 없이 사라진 노드는 30~60분 뒤 임시 노드로 지워집니다.
- **Hugging Face 익명 받기**는 비율 제한 경고가 납니다. 파드에는 HF 토큰을 두지 않습니다.
- **첫 템플릿은 콘솔에서** 만들어야 합니다(6절).
- vLLM 의 PPL(prompt_logprobs)은 `SPEC=mtp` 와 같이 쓰면 엔진이 죽거나 값이 깨집니다. 그래서 품질 점검은 이때 GSM8K 만 돕니다.

## 출처와 참조 연구

이 프로젝트는 다른 이들의 작업 위에 지었습니다. 전체 출처는 [NOTICE](NOTICE) 에 있습니다.

- [vLLM](https://github.com/vllm-project/vllm)(Apache-2.0): 추론 서버입니다. 패치 묶음은 v0.30.0 원본 파일에 덧댑니다.
- [HyperQwen](https://github.com/syv-ai/HyperQwen)(syv-ai, Apache-2.0): 커밋 `8cf642e` 의 vLLM 패치 12개를 바꾸지 않고 가져왔습니다. 채팅 틀 준비 스크립트와 품질 점검(한 줄 고침)도 가져왔습니다. 그 컨테이너 이미지는 지원하는 서빙 이미지 둘 중 하나입니다.
- [AutoRound](https://github.com/intel/auto-round)(Intel, Apache-2.0): `deploy/quant` 의 양자화 방법입니다. 논문: W. Cheng 외, "Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs", arXiv:2309.05516.
- [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)(Apache-2.0): 공개 표의 원본 모델과 고정한 채팅 틀입니다.
- [Tailscale](https://tailscale.com) 사용자 공간 모드: 파드는 TUN 장치 없이 테일넷에 붙습니다.

## 이 저장소가 더한 것

- Vast.ai 래퍼: 역할별 값 상한, 빌리기 직전 다시 확인, 역할마다 하나, 정지 인스턴스 먼저 지우기, 키체인에서만 읽는 키, 환경을 비우고 다시 시작하는 실행기.
- 해시로 고정한 파드 묶음: Args → SB → PS 로 이어지는 대조, 커밋에서 다시 만들어 대조하는 verify, 다시 읽어 확인하는 publish.
- 이미지 두 가지를 같은 모델 표로 돌리는 어댑터 층과, 패치를 정확히 덧대거나 멈추는 검사기.
- 한 인스턴스 안에서 여러 모델을 카드별로 띄우는 차선 계획, 차선별 상태와 HALT 까닭.
- 파드가 내 쪽으로 연결을 열지 않는 모델 보관소(밀어 넣기만).
- AutoRound 를 둘러싼 변환 조리법: 0 채우기, MTP 붙이기, 바꾸기 앞뒤 KL, 층별 ΔW 감사, int8 head 세트.

## 시험

```bash
uv run --no-project --offline --python 3.13 python -m unittest discover -s tests -t .
```

- 망·키체인·GPU 없이 돕니다. 가짜 Vast API·가짜 gist·가짜 git 저장소를 씁니다.
- `tests/fixtures/deploy/` 는 시험용 local 표와 pins 입니다(가짜 별칭·저장소 이름, 모의 주소). 시험은 `deploy/` 의 임시 사본 위에 얹어 읽고, 저장소 파일은 바꾸지 않습니다.
- `test_audit.py` 의 수치 시험은 numpy 가 있을 때만 돕니다(`uv run --with numpy …`).

## 라이선스

Apache License 2.0. Copyright 2026 midopa-dept. [LICENSE](LICENSE)·[NOTICE](NOTICE)
