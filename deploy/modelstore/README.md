# deploy/modelstore — 보관 서버 모델 보관소

우리 손으로 만든 모델(변환·미세조정·고친 판)은 보관 서버 `/srv/models` 에 둡니다. 보관 서버가 먼저 연결합니다.

- **push:** 파드에 밀어 넣습니다.
- **pull:** 변환 결과를 끌어옵니다.
- 빌린 기계 → 집 방향 연결은 없습니다. Tailscale ACL 은 `tag:server → tag:gpu tcp:8090` 만이고, gpu → server 는 ACL tests 로 거부를 고정합니다.

## 흐름(E)

1. `stewart-cloud up quant …`(②)
   - 변환 인스턴스가 테일넷 `stewart-quant` 로 들어와 `deploy/quant/run-convert.sh` 0–11단계를 돕니다.
   - 상태는 `:8001/status`, 내보내기는 `:8090` 입니다.
2. (집) 끌어오기: `sudo modelstore pull stewart-quant.tail-example.ts.net` → `/srv/models/incoming/<시각>-stewart-quant/`
   - 세트별 폴더와 `export.json` 이 생깁니다.
   - 다 받으면 `POST /export/done` 을 보내고, 인스턴스는 스스로 정지합니다.
3. (집) 올리기: `sudo modelstore promote /srv/models/incoming/<그 폴더> qwen38-27b-demo2-w4a16-int8h`
   - 검사를 통과하면 `catalog/<별칭>/` + `manifest.json` 이 생깁니다.
   - models.toml 에 붙일 `files` 조각이 화면에 나옵니다.
   - bf16h 도 쓰려면 같은 폴더로 `… qwen38-27b-demo2-w4a16-bf16h` 를 한 번 더 부릅니다.
4. 사람이 확인하고 `stewart-cloud down <변환 인스턴스>` 합니다.
5. `deploy/serving/models.toml` 에 `[models.<별칭>] source = "home"` + 그 `files` 를 넣습니다. 나머지 칸은 모델 A 를 본뜹니다.
   - `[home].server_ip` 가 채워져 있어야 합니다.
6. pod-files 새 판(make_template_env)을 gist 에 올리고, 템플릿 PU·PS 를 바꿉니다.
7. 파드가 그 별칭으로 뜨면 `:8090/need` 를 엽니다. 데몬(`modelpush.service`)이 30초 안에 보고 catalog 의 파일을 PUT 합니다.
   - 파드는 sha 를 대조하고 vLLM 을 띄웁니다. 60분 안에 못 받으면 HALT + 정지합니다.
   - 데몬 로그(`journalctl -u modelpush`, 바뀔 때만 한 줄): `파드 없음(이름 못 풂)` = 테일넷에 그 이름이 없다(빌리지 않음·내림),
     `/need 없음(파드는 있음, …)` = 이름은 풀리나 받는 창구가 아직·이미 닫혀 있다(기동 중·받기 끝).

## promote 검사

- **파일 꼴:**
  - 허용: `*.safetensors`, `*.json`, `tokenizer.model`, `tokenizer.json`, `*.txt`, `*.md`, `chat_template.jinja`
  - 거부: `.py`, `.bin`, `.pt`, `.pth`, `.pkl`, `.pickle`, 그 밖 전부
  - draft 어휘는 정수 목록 JSON(`mtp_draft_vocab_ids.json`)으로 옵니다. 파드가 그것으로 .pt 를 만들어서, pickle 은 집을 지나지 않습니다.
- **목록:** 폴더와 export.json 목록이 같아야 하고, 크기·sha256 을 다시 대조합니다.
- **내용:**
  - safetensors 는 머리를 파싱합니다(dtype·모양·오프셋·파일 크기).
  - json 은 파싱합니다.
  - 색인과 샤드가 맞아야 합니다. 색인의 모든 텐서가 그 샤드 머리에 있어야 합니다.
- **보고 숫자 문턱**(`config.json` 의 thresholds)

  | 항목 | 문턱 |
  |---|---|
  | 0 채우기 KL 평균 | ≤ 1e-4 |
  | 0 채우기 top-1 | ≥ 0.999 |
  | 양자화 KL 평균 | ≤ 0.1 |
  | 양자화 top-1 | ≥ 0.85 |
  | verify·quality | 통과 |

- 같은 별칭은 다시 올리지 않습니다(새 별칭으로).

## 권한 구조

| 무엇 | 계정 | 쓸 수 있는 곳 | 네트워크 |
|---|---|---|---|
| 데몬 `modelpush.service` | modelpush | incoming(catalog 는 ReadOnlyPaths) | 테일넷 + resolved DNS 창구(127.0.0.53) |
| `sudo modelstore pull` | modelpush(일회 유닛) | incoming | 테일넷 + DNS 창구 |
| `sudo modelstore promote` | modelpush(일회 유닛) | **catalog**(incoming 은 읽기) | 없음 |
| `sudo modelstore list` | modelpush(일회 유닛) | 없음 | 없음 |

- catalog 를 쓸 수 있는 길은 promote 하나입니다. 같은 계정이라도 데몬·pull 은 systemd 가 catalog 를 읽기 전용으로 마운트합니다.
- 공통: ProtectSystem=strict, ProtectHome, PrivateTmp, NoNewPrivileges, 능력 없음, RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
- `/srv/models` 는 restic 대상 밖입니다(설치 스크립트는 restic 을 건드리지 않습니다 — 확인만).
- `/srv/models` 는 따로 마운트된 보관소(16TB btrfs `@models`, fstab nofail)입니다. 유닛과 일회 유닛 모두 `RequiresMountsFor=/srv/models` 라,
  마운트가 빠지면 뜨지 않습니다(닫힌 채 실패). 설치 스크립트도 마운트가 없으면 멈춥니다(`MODELSTORE_ALLOW_UNMOUNTED=1` 로만 넘김).

## 설치

`sudo bash deploy/modelstore/install.sh` 입니다. 다시 돌려도 되고, `/etc/modelstore/config.json` 은 덮지 않습니다.
