#!/usr/bin/env bash
# Thin identity-free geometry entrypoint; all identities come from canonical registries.
set -Eeuo pipefail
IFS=$'\n\t'
umask 027

fail() { printf 'cycleback-geometry-gate-v1: %s\n' "$*" >&2; exit 2; }
readonly SCRIPT_DIR="$(cd -- "${BASH_SOURCE[0]%/*}" && pwd -P)"
: "${PAMS_ATTEMPT_ID:?set immutable lowercase attempt slug}"
: "${PAMS_CYCLEBACK_CANDIDATE_ID:?set W16_H4, W16_H2, or W24_H4}"
readonly ATTEMPT_ID="$PAMS_ATTEMPT_ID"
readonly CANDIDATE_ID="$PAMS_CYCLEBACK_CANDIDATE_ID"
[[ "$ATTEMPT_ID" =~ ^[a-z0-9][a-z0-9._-]{0,39}$ && "$ATTEMPT_ID" != *'..'* ]] \
  || fail 'invalid attempt ID'
[[ "$CANDIDATE_ID" =~ ^(W16_H4|W16_H2|W24_H4)$ ]] || fail 'invalid candidate ID'
[[ ! -v PAMS_GPU_DEVICE ]] || fail 'caller GPU override is forbidden'

for forbidden in \
  PAMS_SOURCE_REVISION PAMS_IMAGE_ID PAMS_CYCLEBACK_CONFIG_RELATIVE \
  PAMS_CYCLEBACK_CONFIG_SHA256 PAMS_CYCLEBACK_POSE_AUTHORITY_ROOT \
  PAMS_CYCLEBACK_POSE_AUTHORIZATION_SHA256 \
  PAMS_CYCLEBACK_POSE_AUTHORITY_RUN_RECEIPT_SHA256 \
  PAMS_CYCLEBACK_LAUNCH_REGISTRY_ROOT PAMS_ROOT; do
  [[ ! -v "$forbidden" ]] || fail "caller identity override is forbidden: ${forbidden}"
done

exec /usr/bin/env -i \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  /usr/bin/python3 -I "$SCRIPT_DIR/pams_conventional_cycleback_secure_launcher.py" \
  --stage geometry --attempt-id "$ATTEMPT_ID" --candidate-id "$CANDIDATE_ID"
