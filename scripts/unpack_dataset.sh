#!/usr/bin/env bash
# unpack_dataset.sh - verify and extract the tars written by pack_dataset.py
# (SPEC v1.1 §10.4, §11.4). Runs on the cloud side (e.g. a RunPod pod with the
# network volume mounted at /workspace).
#
# Usage: unpack_dataset.sh [TAR_DIR] [TARGET_DIR]
#   TAR_DIR     folder with pixel-art-NNN.tar, manifest.json and SHA256SUMS
#               (default: /workspace/dataset-tars)
#   TARGET_DIR  dataset root to extract into, i.e. the captioner's input_dir
#               (default: /workspace/pixel-art); created if missing
#
# Steps:
#   1. Refuses to start unless manifest.json says the pack is complete and
#      SHA256SUMS exists (pack_dataset.py writes it only for a complete pack).
#   2. Verifies the sha256 of every tar listed in SHA256SUMS that is not yet
#      extracted. Any missing or corrupt tar -> nothing is extracted, exit 1.
#   3. Extracts each verified tar into TARGET_DIR (member names are relative
#      POSIX paths, so the source/NNNN/ folder tree is recreated), then records
#      it in TARGET_DIR/.unpacked/<tar>.sha256.
#
# Idempotent: a tar whose marker holds the same sha256 is skipped; re-running
# after an interruption re-extracts only the unfinished tars (overwriting
# their files). Only the tars listed in SHA256SUMS are touched.
#
# Exit codes: 0 done; 1 usage error, missing/incomplete pack, checksum
# mismatch or extraction error.
#
# Needs bash, coreutils (sha256sum) and tar.

set -euo pipefail

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
}

die() {
    printf 'unpack_dataset.sh: error: %s\n' "$*" >&2
    exit 1
}

log() {
    printf '%s  %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    -*) usage >&2; die "unknown option: $1" ;;
esac
[ "$#" -le 2 ] || { usage >&2; die "too many arguments"; }

TAR_DIR="${1:-/workspace/dataset-tars}"
TARGET_DIR="${2:-/workspace/pixel-art}"
MARKER_DIR="$TARGET_DIR/.unpacked"
SUMS="$TAR_DIR/SHA256SUMS"
MANIFEST="$TAR_DIR/manifest.json"

command -v sha256sum >/dev/null 2>&1 || die "sha256sum not found (install coreutils)"
command -v tar >/dev/null 2>&1 || die "tar not found"

[ -d "$TAR_DIR" ] || die "TAR_DIR is not a directory: $TAR_DIR"
[ -f "$MANIFEST" ] || die "missing $MANIFEST"
grep -q '"complete": true' "$MANIFEST" \
    || die "$MANIFEST does not mark the pack complete; finish pack_dataset.py first"
[ -f "$SUMS" ] || die "missing $SUMS (pack incomplete?)"

# Read and validate SHA256SUMS: "<64 hex>  pixel-art-<digits>.tar" per line.
names=()
hashes=()
while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    [ -n "$line" ] || continue
    if [[ "$line" =~ ^([0-9a-f]{64})\ \ (pixel-art-[0-9]+\.tar)$ ]]; then
        hashes+=("${BASH_REMATCH[1]}")
        names+=("${BASH_REMATCH[2]}")
    else
        die "malformed line in $SUMS: $line"
    fi
done < "$SUMS"
[ "${#names[@]}" -gt 0 ] || die "$SUMS lists no tars"

# 1. Verify every tar that still has to be extracted (nothing is extracted on failure).
pending=()
skipped=0
for i in "${!names[@]}"; do
    name="${names[$i]}"
    hash="${hashes[$i]}"
    marker="$MARKER_DIR/$name.sha256"
    if [ -f "$marker" ] && [ "$(cat "$marker")" = "$hash" ]; then
        skipped=$((skipped + 1))
        continue
    fi
    [ -f "$TAR_DIR/$name" ] || die "missing tar: $TAR_DIR/$name"
    log "verifying $name"
    actual="$(sha256sum "$TAR_DIR/$name" | cut -d' ' -f1)"
    [ "$actual" = "$hash" ] || die "checksum mismatch for $name: expected $hash, got $actual"
    pending+=("$i")
done

# 2. Extract the verified tars; a marker is written only after a full extraction.
mkdir -p "$TARGET_DIR" "$MARKER_DIR"
for i in "${pending[@]+"${pending[@]}"}"; do
    name="${names[$i]}"
    log "extracting $name into $TARGET_DIR"
    tar -xf "$TAR_DIR/$name" -C "$TARGET_DIR" --no-same-owner \
        || die "extraction failed: $name"
    printf '%s\n' "${hashes[$i]}" > "$MARKER_DIR/$name.sha256.tmp"
    mv -f "$MARKER_DIR/$name.sha256.tmp" "$MARKER_DIR/$name.sha256"
done

log "done: ${#pending[@]} tars extracted, $skipped already extracted, ${#names[@]} in total; dataset in $TARGET_DIR"
