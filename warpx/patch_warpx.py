"""Make two catalog flags compile (upstream bugs at the pinned commit). Applied to every build alike.

WarpX_UNITY_BUILD=ON: 'Direction' is ambiguous once all files share one unit -> qualify the name.
WarpX_PRECISION=SINGLE: std::max(float, double) in EffectivePotentialES.cpp -> name the type.
  Likewise AddNoRet(float*, double) in MassMatricesDeposition.H.
Neither changes behaviour in double precision.
"""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "Source"
changed = 0
for path in root.rglob("*.cpp"):
    text = path.read_text()
    if "ablastr" in str(path.relative_to(root)).split("/")[0]:
        continue
    new = re.sub(r"(?<![\w:])Direction\{", "ablastr::fields::Direction{", text)
    if path.name == "EffectivePotentialES.cpp":
        old = "std::max(\n                    density_floor*q,"
        if old not in new:
            sys.exit("EffectivePotentialES.cpp changed; update patch_warpx.py")
        new = new.replace(old, old.replace("std::max(", "std::max<amrex::Real>("))
    if new != text:
        path.write_text(new)
        changed += 1
# Mass matrices (implicit solvers only): float fields with double particles pass mixed types.
path = root / "Particles/Deposition/MassMatricesDeposition.H"
text = path.read_text()
path.write_text(text.replace("Atomic::AddNoRet(", "Atomic::AddNoRet<amrex::Real>("))
print(f"patched {changed} files")
