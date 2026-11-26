"""Filter the MILC flag catalog for this CPU. The FIRST value of each option is its baseline."""

import platform
import shutil
import subprocess

AVX512 = {"avx512f", "avx512cd", "avx512vl", "avx512dq", "avx512bw"}
V2 = {"pni", "ssse3", "sse4_1", "sse4_2", "popcnt", "cx16", "lahf_lm"}
V3 = V2 | {"avx", "avx2", "fma", "f16c", "bmi1", "bmi2", "movbe", "xsave"}
NEEDS = {"-march=x86-64-v2": V2, "-march=x86-64-v3": V3, "-march=x86-64-v4": V3 | AVX512,
         "-mprefer-vector-width=256": {"avx"}, "-mprefer-vector-width=512": {"avx512f"}}

# Combinations in which one setting cannot reach the binary, so two labels would mean one build.
INERT = [
    # Without AVX every vector is 128-bit already (verified: identical object code).
    *({"-mprefer-vector-width": f"-mprefer-vector-width={w}", "-march": m}
      for w in (128, 256) for m in ("-march=x86-64", "-march=x86-64-v2")),
    # 256 bits is already the AVX2 default (verified: identical object code).
    {"-mprefer-vector-width": "-mprefer-vector-width=256", "-march": "-march=x86-64-v3"},
]
# The same holds for -march=native on a CPU without AVX-512.
INERT_NO_AVX512 = [{"-mprefer-vector-width": "-mprefer-vector-width=256", "-march": "-march=native"}]
# Values that cannot change the code at the pinned commit (verified: identical object code).
NO_EFFECT = {
    # ks_multicg_field() has no case for these: they run the HYBRID code.
    "KS_MULTICG": {"OFFSET", "REVERSE", "REVHYB"},
    # Only declare macros; no source of su3_rhmd_hisq uses them unless *_GLOBAL is set too.
    "INLINE_C": {"ON"},
    "INLINE_SSE": {"ON"},
}


def cpu_info():
    features = set()
    for line in subprocess.check_output(["lscpu"], text=True).splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() == "flags":
            features.update(value.lower().split())
    if not features:
        raise RuntimeError("lscpu did not report CPU flags")
    return platform.machine().lower(), features


def compiles(flag):
    return subprocess.run(["clang++", "-Werror", "-x", "c++", "-c", "-", flag, "-o", "/dev/null"],
                          input="int main() { return 0; }", text=True,
                          capture_output=True).returncode == 0


def filter_catalog(catalog, architecture, features):
    if architecture not in {"x86_64", "amd64"} or not shutil.which("clang++"):
        raise ValueError("an x86-64 host with clang++ is required")
    compiler = {}
    for name, values in catalog["compiler"].items():
        kept = [v for v in values if v == "default" or (NEEDS.get(v, set()) <= features and compiles(v))]
        if not kept or kept[0] != values[0]:
            raise ValueError(f"baseline of {name} is not supported here")
        compiler[name] = kept
    cmake = {**catalog["device"]["CPU"], **catalog["cpu"]}
    # Keep the option, so it stays recorded, but without the values that do nothing.
    cmake = {n: [x for x in v if x not in NO_EFFECT.get(n, ())] for n, v in cmake.items()}
    names = cmake.keys() | compiler.keys()
    rules = [r for r in catalog.get("invalid_combinations", []) if r.keys() <= names]
    return {"cmake": cmake, "compiler": compiler, "invalid_combinations": rules + INERT + ([] if "avx512f" in features else INERT_NO_AVX512)}
