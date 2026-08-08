#!/usr/bin/env bash
# Thin identity-free entrypoint; the secure launcher consumes canonical registries.
set -Eeuo pipefail
IFS=$'\n\t'
umask 027

fail() { printf 'cycleback-pose-input-authorization-v1: %s\n' "$*" >&2; exit 2; }
readonly SCRIPT_DIR="$(cd -- "${BASH_SOURCE[0]%/*}" && pwd -P)"
: "${PAMS_ATTEMPT_ID:?set immutable lowercase attempt slug}"
readonly ATTEMPT_ID="$PAMS_ATTEMPT_ID"
[[ "$ATTEMPT_ID" =~ ^[a-z0-9][a-z0-9._-]{0,39}$ && "$ATTEMPT_ID" != *'..'* ]] \
  || fail 'invalid attempt ID'

for forbidden in \
  PAMS_SOURCE_REVISION PAMS_IMAGE_ID PAMS_CYCLEBACK_CONFIG_SHA256 \
  PAMS_CYCLEBACK_LAUNCH_REGISTRY_ROOT PAMS_FULL337_RUN_ROOT \
  PAMS_FULL337_GATE_SHA256 PAMS_FULL337_TRAINING_AUTHORIZATION_SHA256 \
  PAMS_FULL337_RUN_RECEIPT_SHA256 PAMS_FULL337_LEDGER_SHA256 \
  PAMS_CYCLEBACK_REPRESENTATION_ROOT; do
  [[ ! -v "$forbidden" ]] || fail "caller identity override is forbidden: ${forbidden}"
done

exec /usr/bin/env -i \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  /usr/bin/python3 -I "$SCRIPT_DIR/pams_conventional_cycleback_secure_launcher.py" \
  --stage adapter --attempt-id "$ATTEMPT_ID"
