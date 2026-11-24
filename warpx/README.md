# WarpX CPU benchmark

Builds and benchmarks WarpX for random flag configurations. x86-64 Linux with Docker; the sources are
pinned inside the image. Run from the repository root:

```sh
docker pull ghcr.io/jakobklemm/dphpc26:warpx-cpu
docker tag ghcr.io/jakobklemm/dphpc26:warpx-cpu dphpc26:warpx-cpu
# or build it: docker build -t dphpc26:warpx-cpu -f warpx/Dockerfile.cpu warpx

mkdir -p ~/dphpc-data/warpx
docker run -d --name warpx-random -v ~/dphpc-data/warpx:/data dphpc26:warpx-cpu random \
  --count 500 --workers 16 --threads 8 --results-dir /data/results/random500
```

`--workers`: parallel compile jobs (about 1 GB RAM each). `--threads`: OpenMP threads of the benchmark;
use your number of physical cores. `docker logs -f warpx-random` shows progress, `docker stop` pauses
and `docker start` resumes. Values your CPU cannot run are filtered out, so plans differ between CPUs
with different features. `random --count 500 --dry-run --results-dir /tmp/x` prints the plan.

Sampling matches the llama run: seed 20261021, harmonic toggle counts, invalid configurations resampled.
Every build uses a fresh directory without ccache and is checked against the requested flags
(`build.json`), including a hash of the compiled code.

`results/Davide_ryzen5800x_cpu_random_harmonic211.zip`: the first 211 of 500 configurations, all
completed (run stopped early), 8 threads, 20 steps, 3 repetitions.
