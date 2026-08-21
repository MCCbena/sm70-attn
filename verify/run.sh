#!/usr/bin/env bash
# sm70 d256 attention — standalone cross-check (V100 / WSL2)
#
# Usage:
#   bash verify/run.sh           # ① small battery (all-rows cases + sampled 3000)
#   bash verify/run.sh oob       # ② OOB-guard: K/V exact-size, any past-kv_len read is illegal
#   bash verify/run.sh full      # ③ + 32k sampled case (CPU ref slower, a few minutes)
#   bash verify/run.sh oobfull   # oob + full
#   RUN_SANITIZER=1 bash verify/run.sh oob   # + compute-sanitizer memcheck (10-50x slower)
#
# The CUDA binary is compiled ONCE (sm_70, V100). CPU reference runs on host.
# Does NOT need llama.cpp, a model, or the 8080 server — standalone.

set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
cuda="$root/ggml/src/ggml-cuda"
inc="$cuda/sm70-vendor"
bin="$here/sm70_verify"

mode="${1:-n}"

# ---- source fingerprint guard -----------------------------------------------
# The harness is valid ONLY against the exact kernel that was static-audited.
# If the repo drifted, fail loudly instead of producing unverifiable numbers.
kmd5="$(md5sum "$cuda/fattn-sm70-d256-kernel.cuh" | cut -d' ' -f1)"
expect="7f32188b7786adb43179f750396a8774"   # fattn-sm70-d256-kernel.cuh @ main bfc1e99 (2026-08-21)
if [ "$kmd5" != "$expect" ]; then
    echo "kernel md5 drift: $kmd5 != $expect" >&2
    echo "— audit the diff / update the harness before trusting any numbers" >&2
    exit 2
fi
echo "kernel md5 ok: $kmd5"

if command -v nvcc >/dev/null 2>&1; then :; else
    echo "nvcc not found on PATH — is the CUDA toolkit installed in WSL2?" >&2
    exit 3
fi

echo "== build =="
nvcc -O2 -std=c++17 -arch=sm_70 -I "$inc" -I "$cuda" "$here/sm70_verify.cu" -o "$bin" || {
    echo "nvcc build failed" >&2; exit 4; }

# ---- mode envs ----------------------------------------------------------------
case "$mode" in *o*) export SM70_VERIFY_OOB=1 ;;  *) unset SM70_VERIFY_OOB ;; esac
case "$mode" in *f*) export SM70_VERIFY_FULL=1 ;; *) unset SM70_VERIFY_FULL ;; esac

echo "== run ($mode) =="
"$bin"
rc=$?
echo "exit = $rc"

# ---- optional sanitizer pass (OOB only) ----------------------------------------
# SLOWS the kernel 10-50x; keep it to the non-full battery.
if [[ "$mode" == *o* && "${RUN_SANITIZER:-0}" == "1" ]]; then
    echo "== compute-sanitizer memcheck =="
    compute-sanitizer --tool memcheck --error-exitcode 9 "$bin"
    echo "sanitizer exit = $?"
fi

exit $rc
