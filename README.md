English | [한국어](README.ko.md)

# stewart-cloud

stewart-cloud rents a GPU on Vast.ai, serves an open model on it with vLLM, and makes it reachable only from your own Tailscale network.

Use it when you want a large open model for a few hours without running your own GPU and without opening a port to the internet. Everything the instance runs or downloads is pinned by hash, and the instance stops itself when it sits idle or fails to boot, so a forgotten rental does not keep running. It was built for [stewart](https://github.com/midopa-dept/stewart), but any OpenAI-compatible client can use the server.

## What it is

- `stewart_cloud/` — the `stewart-cloud` command. It finds offers, rents, shows status, reads logs, and destroys instances. It also rebuilds the pod bundle from a git commit and checks it (`verify`), then uploads it to a private gist and creates a pinned Vast template (`publish`).
- `deploy/vast/` — the pod bundle that runs when the instance starts. It joins your tailnet, downloads the model, starts vLLM, and watches for idle time.
- `deploy/serving/` — the model table, adapters for two container images (HyperQwen and the official vLLM image), and a vLLM 0.30.0 patch set with a checker that applies the patches exactly or stops.
- `deploy/quant/` (optional) — converts a BF16 checkpoint to AutoRound W4A16 on a rented A100 or H100 and measures KL divergence before and after.
- `deploy/modelstore/` (optional) — keeps models on your own Linux server and pushes them to the pod over Tailscale. The pod cannot open connections to your server.

No model weights are distributed. The public model table has one entry, a W4A16 conversion of Qwen3.8-27B, and it stays unselectable until you convert it yourself (see [Convert models](#convert-models)).

## Requirements

- A Mac. Keys are read from the login keychain with `/usr/bin/security`, and `verify` runs the bundle builder inside `sandbox-exec`.
- `/usr/bin/python3` (Command Line Tools) to run the command, and [uv](https://docs.astral.sh/uv/) with CPython 3.13 for the tests and the bundle builder.
- A Vast.ai account with prepaid credit and two API keys:
  - a rental key with misc, instance_read and instance_write;
  - a template key with user_read and user_write.
- A Tailscale account.
- A GitHub account and a fine-grained token with only Account permissions → Gists (read and write).
- For `deploy/modelstore`: a Linux server with systemd on the same tailnet.

Rentals are billed by the hour while the instance runs. A stopped instance still pays for its disk until you destroy it. An evening on one RTX 3090 costs about a dollar; a three-model conversion on an A100 costs around ten. See [Cost estimates](#cost-estimates).

## Setup

### 1. Vast.ai

1. Add prepaid credit and turn off automatic top-up. Your balance then becomes the spending limit; stewart-cloud does not read your balance.
2. Create the two API keys above. Keep them separate: user_write also covers creating API keys and account environment variables. If you skip the template key, the rental key needs user_read and user_write as well.
3. In the account settings, add the environment variable `TS_AUTHKEY` with the Tailscale key from the next step. The pod moves it to a mode-600 file in `/dev/shm` and removes it from its environment before starting anything else. It never appears in this repository, the template, or a command line.

### 2. Tailscale

1. In the ACL policy, create the tag `tag:gpu`. Let your own devices reach `tag:gpu` only on the serving ports (8000, the lane ports 80k0 for evaluation instances, and status port 8001). Do not let `tag:gpu` reach anything. If you use the model store, allow only your server to reach the pod's receive port.
2. Create an auth key with that tag: reusable, ephemeral, and pre-approved.
3. Each role gets a fixed node name:

   | Role | Node name | Use |
   |---|---|---|
   | serve | `stewart-gpu` | one card for interactive use |
   | eval, eval2 | `stewart-gpu-eval`, `stewart-gpu-eval2` | several cards, one model per lane |
   | quant | `stewart-quant` | model conversion |

   In the config file, write them as `<name>.<your-tailnet>.ts.net`.

### 3. Gists

1. Create a secret gist for bundles. `publish` adds files such as `stewart-pod-v2.b64` to it.
2. Create a second secret gist for the pod settings file. The pod reads only these keys and ignores every other line:

   ```
   CTX=long
   SPEC=mtp
   PREFIX_CACHE=1
   QUALITY=auto
   MODEL=<alias from the model table>
   ADAPTER=auto
   ```

   Allowed values: `CTX` fast or long; `SPEC` mtp; `PREFIX_CACHE` 0 or 1; `FAST_VARIANT` 0 or 1; `QUALITY` auto, 0 or 1; `ADAPTER` auto, hyperqwen-8cf642e, vllm-patched or vllm-stock; `MODEL` and `PLAN` up to 48 lowercase letters, digits, dots and hyphens.
3. Create the GitHub token described in Requirements.

### 4. Install

The install folder must be owned by root, so that other programs running as your user, AI agents included, cannot change the code, config or state. The default is `/opt/stewart-cloud` (`PREFIX` in `stewart_cloud.py`).

```bash
git clone https://github.com/midopa-dept/stewart-cloud ~/src/stewart-cloud
cd ~/src/stewart-cloud
uv run --no-project --offline --python 3.13 python -m unittest discover -s tests -t .
```

The tests need no network, keychain or GPU. When they pass, install:

```bash
sudo install -d -o root -g wheel -m 755 /opt/stewart-cloud /opt/stewart-cloud/bin
sudo install -o root -g wheel -m 644 stewart_cloud/stewart_cloud.py /opt/stewart-cloud/stewart_cloud.py
sudo install -o root -g wheel -m 755 stewart_cloud/stewart-cloud /opt/stewart-cloud/bin/stewart-cloud
sudo install -o root -g wheel -m 644 stewart_cloud/config.example.json /opt/stewart-cloud/config.json
sudo install -d -o "$USER" -g staff -m 700 /opt/stewart-cloud/state
```

- In `config.json`, set the four `ts_name` values to your tailnet and review `max_dph` (hourly price limit) and `disk` for each role. The code also enforces hard limits: 8.0 $/hr and 8 GPUs per instance.
- `site.json` is optional. Add it only to change the keychain account, home folder, record folder, repository or bundle folder. Without it, the command uses the calling user and paths under that user's home. See `stewart_cloud/site.example.json`.
- For `verify` and `publish`, copy uv's CPython 3.13 to `/opt/stewart-cloud/py313/`, owned by root and not writable by group or others. The bundle must be rebuilt with 3.13: other versions produce different bytes and a different hash.

### 5. Keys

Store each key as a generic password in the login keychain, with your user name as the account and `/usr/bin/security` allowed to read it. With `-w` and no value, `security` prompts for the key without echoing it.

```bash
security add-generic-password -s stewart-cloud.vast-api -a "$USER" -T /usr/bin/security -w
security add-generic-password -s stewart-cloud.vast-template -a "$USER" -T /usr/bin/security -w
security add-generic-password -s stewart-cloud.github-gist -a "$USER" -T /usr/bin/security -w
```

Check that the rental key works. If the output reports zero instances (`stewart- 인스턴스 0개`), it does:

```bash
/opt/stewart-cloud/bin/stewart-cloud status
```

Keys never go through environment variables, command lines or files. The command calls `security` itself, reads the output through a pipe, and puts the key only in the HTTP Authorization header. Over SSH the keychain is locked and the command exits with 78; run `security unlock-keychain ~/Library/Keychains/login.keychain-db` first.

### 6. First template

`publish` copies the image, disk size and Args format from the current template, so you create the first template by hand.

1. Build a bundle:

   ```bash
   uv run --no-project --python 3.13 python deploy/vast/make_template_env.py --out ~/.local/share/stewart-cloud/bundles/v1
   ```

   The folder gets `pod-files.b64`, `args.txt`, `boot.sb.txt` and `docker-options.txt`. The same source always produces the same bytes.
2. Upload `pod-files.b64` to the bundle gist and copy the raw URL pinned to that revision.
3. In the Vast console, create a private template:
   - Image: an `[images.*].ref` from `deploy/serving/models.toml` (pinned by digest).
   - Launch mode: Entrypoint, with Args set to the contents of `args.txt`.
   - Docker options: the contents of `docker-options.txt`, with `<gist raw 주소>` replaced by the bundle URL and `<설정 gist 최신 raw 주소>` replaced by the settings gist's latest raw URL.
4. Fill in `stewart_cloud/state-seed.example.json` (template hash_id, the bundle hash `PS`, and the sha256 of `boot.sb.txt` and `args.txt`), install it as the state file, and check it:

   ```bash
   cp stewart_cloud/state-seed.example.json /opt/stewart-cloud/state/templates.json   # after filling it in
   chmod 600 /opt/stewart-cloud/state/templates.json
   /opt/stewart-cloud/bin/stewart-cloud templates
   ```

For conversions, repeat these steps with `deploy/quant/make_quant_bundle.py` and add `current.quant`.

### 7. Updating the bundle

After you change and commit `deploy/`, build a new bundle into a new folder, then verify and publish it:

```bash
uv run --no-project --python 3.13 python deploy/vast/make_template_env.py --out ~/.local/share/stewart-cloud/bundles/v2
/opt/stewart-cloud/bin/stewart-cloud verify v2 <40-character commit> <64-character PS>
/opt/stewart-cloud/bin/stewart-cloud publish v2 <40-character commit> <64-character PS>
```

- `verify` needs no key or network. It reads the git objects directly, recomputes each SHA-1, skips `.git/config` filters, extracts `deploy/` into an empty folder, and runs the builder in `sandbox-exec` with no network. It passes only when the output matches your bundle folder byte for byte and the hash matches `PS`.
- `publish` runs `verify` again and adds the bundle to the gist. It downloads the pinned URL without a token and compares it with `PS`, then creates the new template and reads it back. It switches the current template only when every field matches.
- To roll back, publish an older bundle again.

## Rent and serve

```bash
/opt/stewart-cloud/bin/stewart-cloud offers serve
/opt/stewart-cloud/bin/stewart-cloud up serve <offer_id> <max $/hr>
/opt/stewart-cloud/bin/stewart-cloud status
/opt/stewart-cloud/bin/stewart-cloud logs <instance_id>
```

- `offers` lists matching offers, cheapest per GPU first, and prints the `up` command to run.
- `up` repeats the search and checks the offer and price again. It rents one instance per role, and refuses while the role's node name is still on the tailnet. A stopped instance of the same role is destroyed first.
- `up` reads the current template back from the API and rents only if its bundle hash, boot script hash and Args match the `publish` record.
- `status` shows uptime, cost so far, time left before the idle stop, model download progress, and failed lanes or boot errors.
- For evaluation, `up eval <offer_id> <max $/hr> <plan>` picks a plan from `plans` in the model table.

### Choosing a model

The pod picks one entry of the model table by `MODEL=<alias>` in the settings gist. Repository, revision and file hashes come only from the table and `pins/` inside the bundle; the settings gist can only choose an alias and an adapter.

To serve a model from Hugging Face:

1. Add an entry with `source = "hf"` to `deploy/serving/models.toml`, or to `models.local.toml` in the same folder for entries you keep private (it is merged when present). Fill in `repo`, the 40-character `rev`, `pins`, `arch`, `served_name`, `adapters` and the vLLM arguments under `[<alias>.serve]`. `deploy/serving/README.md` describes each field.
2. Write `pins/<alias>.sha256` with one `sha256  file name` line per file at that revision.
3. Build and publish a new bundle, and set `MODEL=<alias>` in the settings gist.

## Connect stewart

Add a backend to stewart's `backends.toml`:

```toml
[backends.cloud]
description = "Vast GPU vLLM (tailnet stewart-gpu)"
base_url = "http://stewart-gpu.<your-tailnet>.ts.net:8000/v1"
model = "<served_name from the table>"
max_context = 150000          # match max_model_len in the table
concurrency = 8               # match max_num_seqs in the table
enabled = true
timeout_s = 240
```

- There is no vLLM API key; the Tailscale ACL is the boundary.
- `stewart cloud wait` waits until the instance serves, and `stewart ping -b cloud` checks the connection.
- To use it from `stewart chat`, set `backend = "cloud"` in the profile or `default = "cloud"` in `backends.toml`.
- A connection error while no instance is running is expected.

## Convert models

`deploy/quant/` converts the `quant_src` entries of the model table in order, on one quant instance:

1. download the BF16 source pinned by revision and hash;
2. pad the MLP width with zeros when it does not fit 4-bit groups of 128, and check that the padded model gives the same outputs (KL);
3. quantize to W4A16 with AutoRound and re-attach the MTP layers;
4. measure KL against the source;
5. prepare int8 heads and the draft vocabulary, then start vLLM on the result and run a short quality check;
6. optionally, for a pair of source and edited checkpoints, write a per-layer weight-difference summary (`audit_delta.py`).

Rent with `up quant`. The instance serves `/status` on port 8001 and the results on port 8090, which only your model store server may read. It stops itself 120 minutes after the last model is ready, or once the results are collected. The steps and checks are described in `deploy/quant/README.md`.

## Move models to your own storage

`deploy/modelstore/` runs on your Linux server and keeps models under `/srv/models`. Your server always opens the connection; the pod never connects to it.

- `modelstore pull <quant node>` collects conversion results into `incoming/`.
- `modelstore promote <folder> <alias>` checks file types, sizes, hashes, safetensors headers and the KL thresholds, then adds the model to the catalog and prints the `files` list for the model table.
- An entry with `source = "home"` makes the pod open a receive port and wait. The `modelpush` service sees it within 30 seconds and pushes the catalog files; the pod checks every hash before starting vLLM.

Installation and the permission layout are described in `deploy/modelstore/README.md`.

## Shut down

```bash
/opt/stewart-cloud/bin/stewart-cloud down <instance_id>
```

`down` destroys the instance; it never just stops it. It logs the pod out of the tailnet, deletes the instance with its disk, and checks that it is gone from the list.

The pod stops itself in these cases. A stopped instance still pays for its disk, so run `down` afterwards.

| Condition | Default | Variable |
|---|---|---|
| No requests or tokens, nothing running or waiting | 90 minutes for serve, 30 for evaluation | `IDLE_MIN` |
| Server not up yet on first boot (time spent receiving from the model store does not count) | 45 minutes | `BOOT_GRACE_MIN` |
| Total uptime | 8 hours; 4 × targets + 2 hours for quant | `MAX_HOURS` |
| Boot error | one-line reason, then a 900-second wait so you can read `logs`, then stop | — |

## Cost estimates

```
rental     ≈ offer dph_total ($/hr) × hours running
stopped    ≈ disk price × hours stopped        (0 after down)
total      ≈ rental + stopped + network charges (vary by offer)
```

`dph_total` includes GPU, CPU and disk. The `usd` field in `status` is `dph_total × hours running`; it leaves out disk charges while stopped and network charges. Offer prices change all the time, and the price of a running instance can change too, so treat these as examples:

| Example | Calculation | Estimate |
|---|---|---|
| serve, one RTX 3090, 3 hours | 0.30 × 3 | about $0.90 |
| eval, eight RTX 3090, 2 hours | (8 × 0.30) × 2 | about $4.80 |
| quant, one A100 80GB, three models (about 7.5 hours) | 1.30 × 7.5 | about $10 |

## Known issues

- Offer IDs change. The same machine can reappear under a new offer ID within minutes, so `up` with an ID from an earlier `offers` may be refused (exit 69). Run `offers` again and rent right away.
- The Vast API sometimes times out (exit 75). A create request may still have gone through, so run `status` before trying again.
- If an old node with the same name is still on the tailnet, a new node would become `stewart-gpu-1`. `up` therefore refuses until the name is free. Ephemeral nodes that were not logged out disappear after 30 to 60 minutes.
- Model downloads from Hugging Face are anonymous and may be rate-limited. The pod holds no Hugging Face token.
- The first template must be created in the console (see [First template](#6-first-template)).
- With `SPEC=mtp`, vLLM's prompt log-probabilities can crash the engine or return wrong values, so the quality check runs only GSM8K in that case.
- Code comments and command output are in Korean.

## Acknowledgements & references

- [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0) is the inference server. The patch set applies to the v0.30.0 sources.
- [HyperQwen](https://github.com/syv-ai/HyperQwen) by syv-ai (Apache-2.0), commit `8cf642e`: 12 vLLM patches taken unchanged, the chat template preparation scripts, and the quality check (one line changed). Its container image is one of the two supported serving images.
- [AutoRound](https://github.com/intel/auto-round) by Intel (Apache-2.0) is the quantization method in `deploy/quant`. Paper: W. Cheng et al., "Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs", arXiv:2309.05516.
- [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0) is the source model in the public table and the origin of the pinned chat template.
- [Tailscale](https://tailscale.com) userspace networking lets the pod join the tailnet without a TUN device.

Full attributions are in [NOTICE](NOTICE). Tailscale and Vast.ai are trademarks of their owners; this project is not affiliated with either company.

## License

Apache License 2.0. Copyright 2026 midopa-dept. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
