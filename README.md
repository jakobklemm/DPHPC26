# DPHPC

```sh
docker pull ghcr.io/jakobklemm/dphpc26:cuda

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
cp gromacs/flags.json ~/dphpc-data/gromacs/
docker run --rm -v ~/dphpc-data:/data dphpc26:cpu train --model hardware
docker run --rm -v ~/dphpc-data:/data dphpc26:cpu active-sampling --steps 10
```
