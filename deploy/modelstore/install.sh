#!/bin/bash
# 보관 서버에 modelpush 를 깐다. 사람이 sudo 로 한 번: sudo bash deploy/modelstore/install.sh
# 하는 일: 계정 modelpush(nologin), /srv/models/{catalog,incoming}, /opt/modelstore/modelpush.py, /etc/modelstore/config.json(없을 때만),
#          /usr/local/sbin/modelstore, modelpush.service(enable + restart). 다시 돌려도 된다(설정 파일은 덮지 않는다).
# 하지 않는 일: Tailscale ACL(메인이 서버 설치 때 전체 파일로), restic 설정(/srv/models 는 대상 밖 — 확인만 안내).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ "$(id -u)" = 0 ] || { echo "sudo 로 돌린다" >&2; exit 1; }
command -v python3 >/dev/null && python3 -c 'import sys; assert sys.version_info >= (3, 10)' || { echo "python3 3.10 이상이 필요하다" >&2; exit 1; }
command -v systemd-run >/dev/null || { echo "systemd 가 필요하다" >&2; exit 1; }
# /srv/models 는 따로 마운트된 보관소(16TB btrfs @models)여야 한다. 마운트가 빠진 채 돌리면 루트 디스크에 빈 보관소를
# 만들게 되므로 멈춘다(유닛·감싸개도 RequiresMountsFor=/srv/models). 다시 돌려도 되는 꼴은 그대로다.
if [ "${MODELSTORE_ALLOW_UNMOUNTED:-0}" != 1 ] && ! mountpoint -q /srv/models; then
  echo "/srv/models 가 마운트돼 있지 않다(findmnt /srv/models 로 본다). 마운트한 뒤 다시 돌린다" >&2
  exit 1
fi
id modelpush >/dev/null 2>&1 || useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin modelpush
install -d -o root -g root -m 0755 /srv/models
install -d -o modelpush -g modelpush -m 0750 /srv/models/catalog /srv/models/incoming
install -d -o root -g root -m 0755 /opt/modelstore /etc/modelstore
install -o root -g root -m 0644 "$HERE/modelpush.py" /opt/modelstore/modelpush.py
[ -e /etc/modelstore/config.json ] || install -o root -g root -m 0644 "$HERE/config.example.json" /etc/modelstore/config.json
install -o root -g root -m 0755 "$HERE/modelstore" /usr/local/sbin/modelstore
install -o root -g root -m 0644 "$HERE/modelpush.service" /etc/systemd/system/modelpush.service
systemctl daemon-reload
systemctl enable modelpush.service
systemctl restart modelpush.service   # 다시 돌릴 때 새 modelpush.py·유닛이 바로 먹게(enable --now 는 돌고 있으면 그대로 둔다)
sleep 2
systemctl --no-pager --lines=5 status modelpush.service || true
echo
echo "설치 끝. 확인:"
echo "  systemctl show modelpush -p IPAddressDeny -p IPAddressAllow -p ReadOnlyPaths -p ReadWritePaths -p RequiresMountsFor"
echo "  sha256sum /opt/modelstore/modelpush.py   # 저장소의 deploy/modelstore/modelpush.py 와 같아야 한다"
echo "  restic 설정에 /srv/models 가 대상으로 들어 있지 않은지 본다(대상 밖)"
