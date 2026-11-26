"""Image build: create the one thermalized lattice that every benchmarked binary starts from."""
import os, subprocess
from pathlib import Path
src, build, data = "/opt/src/milc", "/tmp/ref", Path("/opt/data")
data.mkdir(parents=True)
subprocess.run(["cmake", "-G", "Ninja", "-S", src, "-B", build, "-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++",
                "-DCMAKE_C_FLAGS=-O2 -march=x86-64", "-DSUBDIR=ks_imp_rhmc", "-DOMP=OFF"], check=True)
subprocess.run(["cmake", "--build", build, "--target", "su3_rhmd_hisq"], check=True)
text = Path("/workspace/milc/inputs_bench.template").read_text()
for a, b in {"@WARMS@": "3", "@STEPS@": "8", "@LOAD@": "fresh", "@SAVE@": "save_serial /opt/data/lattice.l8888"}.items():
    text = text.replace(a, b)
Path("/tmp/gen.in").write_text(text)
out = subprocess.run([f"{build}/ks_imp_rhmc/su3_rhmd_hisq", "/tmp/gen.in"], capture_output=True, text=True, check=True).stdout
assert "RUNNING COMPLETED" in out and (data / "lattice.l8888").stat().st_size > 1_000_000, out[-2000:]
