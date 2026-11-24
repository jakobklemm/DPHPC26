# Flag catalog contract

This document is the complete catalog contract needed by this skill. It captures
the GROMACS/llama.cpp catalog conventions without requiring their source,
preconfigured flag collections, training data, or ML implementation. The bundled
validator implements the structural checks described here using only Python's
standard library.

## Shape

Illustration only; replace these fictional application options with source-backed
ones. The example has a CPU-only reference (`APP_GPU` omitted), and CUDA builds
set `APP_GPU=CUDA` through the selected backend.

```json
{
  "device": {
    "CUDA": {
      "APP_GPU": ["CUDA"],
      "APP_GPU_FUSION": ["ON", "OFF"]
    }
  },
  "cuda_compiler": {
    "-Xptxas -O": ["-Xptxas=-O3", "-Xptxas=-O2"]
  },
  "cpu": {
    "APP_OPENMP": ["ON", "OFF"],
    "APP_KERNEL": ["portable", "vectorized"]
  },
  "compiler": {
    "-O": ["-O3", "-O2"],
    "-funroll-loops": ["-fno-unroll-loops", "-funroll-loops"]
  },
  "invalid_combinations": [
    {"APP_GPU": "CUDA", "APP_OPENMP": "OFF"}
  ]
}
```

| Key | Contents |
| --- | --- |
| `device` | Object of uppercase backend names (`CUDA`, `VULKAN`, etc.) to option maps. `{}` for CPU-only. No extra `CPU` nesting. |
| `cpu` | CPU and backend-independent build options mapped to arrays of strings. May be `{}`. |
| `compiler` | Compiler flag **group** to array of complete command-line alternatives. May be `{}`. |
| `cuda_compiler` | Optional CUDA compiler flag groups. Omit when unused. |
| `invalid_combinations` | Array of forbidden assignments. Use `[]` when none are known. |

These are the only top-level keys. Require `device`, `cpu`, `compiler`, and
`invalid_combinations` in new catalogs. Each option domain must be a nonempty
array of unique strings. Booleans are `"ON"` / `"OFF"`, not JSON booleans; numeric
values are strings. A one-element domain is valid for a fixed backend selector.

Keep option names unique within each effective configuration (CPU/shared + one
selected backend + host/CUDA compiler groups). Different backend maps may reuse
an option name. Avoid collisions because consumers flatten selected maps.

Compiler group names identify a feature, not the complete invocation: `-O` maps
to `-O3`, `-O2`, etc.; `-ffp-contract` maps to `-ffp-contract=on`, etc.; the
historical CUDA group `-Xptxas -O` maps to complete `-Xptxas=-O3` alternatives.
No additional shell quoting belongs inside the values.

## Defaults and delivery are separate from the JSON shape

For **new** catalogs, put the intended reference choice first.
The intended reference configuration is the first value from each shared option
and each option in the selected backend; CUDA compiler groups apply only to CUDA.
Do not merge options from multiple backends. Backend selectors and host-dependent
settings still require an adapter; absent options follow documented application
defaults, not an invented universal `OFF` value.

The shape does not tell a generic consumer whether an option is a CMake cache
variable or an environment variable. A backend's environment controls may live
inside its `device` map, but must not be passed as CMake options. Do not introduce
a top-level `runtime` section or use filtered output
(`cmake`, a flattened `device`, etc.) as the new catalog format.

## Exclusion semantics

For an effective configuration `config`, the rule

```json
{"APP_FUSION": "ON", "APP_GRAPHS": "OFF"}
```

rejects it only when **both** assignments match. Multiple rule objects are ORed.
Thus `APP_FUSION=ON` requiring `APP_GRAPHS=ON` can be expressed this way if the
latter's domain is `ON/OFF`. Enumerate all disallowed alternatives for larger
domains. Rule keys must name catalog options, and values must be strings. The
equivalent matching logic, after applying documented application defaults, is:

```python
def is_valid(effective_config, rules):
    return not any(
        all(effective_config.get(name) == value for name, value in rule.items())
        for rule in rules
    )
```

Matching is exact string equality. Missing options do not match. Consumers may
normalize boolean aliases before comparison, but new catalogs should consistently
use the option's native spelling. Rules cannot express arbitrary predicates,
OR within one object, hardware properties, or external dependency discovery.

For compatibility with historical policy exclusions or implicit defaults, the
checker warns rather than fails for out-of-domain rule values (for example,
excluding `-O0` when the selectable domain is `-O3`/`-O2`). This is not permission
to invent implicit defaults for new applications. Similarly,
`invalid_combinations` can encode experimental/correctness policy, not just
compiler errors; only include rules supported by evidence or the user's policy.

## Compatibility boundary

The shared shape preserves catalog tooling conventions. It is not a universal
application plugin interface. Existing host filters, compiler-group parsers,
option schemas, CLI choices, and loaders contain application-specific logic.
New options, backends, and build systems need their own adapters or extensions.
Do not extend the catalog schema with provenance, dependency requirements, or
workload definitions.
