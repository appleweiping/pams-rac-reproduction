#!/usr/bin/env bash

# Source this file from the server builder/launcher.  The fixed file order and
# path labels make the digest independent of the checkout's absolute path.
pams_environment_fingerprint() {
  local context="$1"
  local relative
  command -v sha256sum >/dev/null 2>&1 || {
    echo "sha256sum is required to fingerprint the container environment" >&2
    return 1
  }
  {
    for relative in Dockerfile constraints.txt pams requirements.txt; do
      [[ -f "${context}/${relative}" ]] || {
        echo "missing container environment input: ${context}/${relative}" >&2
        return 1
      }
      printf '%s\n' "${relative}"
      sha256sum -- "${context}/${relative}" | awk '{print $1}'
    done
  } | sha256sum | awk '{print $1}'
}
