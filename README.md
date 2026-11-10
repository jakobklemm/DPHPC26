# DPHPC

```sh
docker pull ghcr.io/jakobklemm/dphpc26:cpu

# optional?
docker tag ghcr.io/jakobklemm/dphpc26:cpu dphpc26:cpu

mkdir -p ~/dphpc-data
docker run --rm -v ~/dphpc-data:/data dphpc26:cpu setup --skip-train

docker run --rm -v ~/dphpc-data:/data dphpc26:cpu sweep \
  --workers 4 --prompt 32 --generate 32 --repetitions 1

docker run --rm -v ~/dphpc-data:/data dphpc26:cpu random \
  --count 50 --flags-per-build 7
```

## Train & active sampling

```sh
mkdir -p ~/dphpc-data/llama ~/dphpc-data/gromacs
cp llama/flags.json ~/dphpc-data/llama/
docker run --rm -v ~/dphpc-data:/data dphpc26:cpu train --model hardware
docker run --rm -v ~/dphpc-data:/data dphpc26:cpu active-sampling --steps 10
```
