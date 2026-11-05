#!/usr/bin/env bash
# Make the benchmark input (bench.tpr) ONCE. Every configuration then runs this same
# file, so only the build differs between measurements (like model.gguf for llama).
#
# Usage: gromacs/prepare_input.sh <path/to/gmx> [size] [output.tpr]
#   size = water box of the official GROMACS benchmark set, in thousands of atoms:
#          0024, 0048, 0096 (default), 0192, ...  (water_GMX50_bare, PME)
# Writes gromacs/bench.tpr (or output.tpr) and its .sha256 beside it.
set -euo pipefail

GMX="$1"
SIZE="${2:-0096}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$(realpath -m "${3:-$HERE/bench.tpr}")"
DL="$HERE/download"
SRC="$DL/water-cut1.0_GMX50_bare/$SIZE"
WATER_SHA256="2219c10acb97787f80f6638132bad3ff2ca1e68600eef1bc8b89d9560e74c66a"

mkdir -p "$DL"
if [ ! -f "$DL/water_GMX50_bare.tar.gz" ]; then
    curl -sSfL -o "$DL/water_GMX50_bare.tar.gz" https://ftp.gromacs.org/pub/benchmarks/water_GMX50_bare.tar.gz
fi
echo "$WATER_SHA256  $DL/water_GMX50_bare.tar.gz" | sha256sum -c -
[ -d "$SRC" ] || tar -xzf "$DL/water_GMX50_bare.tar.gz" -C "$DL"

"$GMX" grompp -f "$HERE/bench.mdp" -c "$SRC/conf.gro" -p "$SRC/topol.top" \
    -o "$OUT" -po "$DL/mdout_$SIZE.mdp" -maxwarn 0 > "$DL/grompp_$SIZE.log" 2>&1
echo "water_GMX50_bare/$SIZE" > "$OUT.info"
(cd "$(dirname "$OUT")" && sha256sum "$(basename "$OUT")") | tee "$OUT.sha256"
