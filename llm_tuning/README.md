# LLM based tuning 

```bash
mkdir -p /tmp/llm-run
set -o pipefail
docker run --rm --network host \
  --user "$(id -u):$(id -g)" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e OPENAI_API_KEY -e OPENAI_BASE_URL -e OPENAI_MODEL \
  -v "$PWD:/workspace:ro" \
  -v /tmp/llm-run:/data \
  -w /workspace \
  --entrypoint bash ghcr.io/jakobklemm/dphpc26:cpu -ec '
    python -m ml.system --output /data/system.json
    exec python -m llm_tuning \
      --system /data/system.json \
      --application /workspace/llm_tuning/examples/llama-cpu.json \
      --count 7 \
      --output /data/result.json
  ' 2>&1 | tee /tmp/llm-run/progress.log
```

```bash
tail -f /tmp/llm-run/progress.log
jq '.candidates' /tmp/llm-run/result.json
jq '.recommendation' /tmp/llm-run/result.json
```

- `system.json`: discovered hardware and compilers.
- `trace.json`: array of API requests/responses and call timings; updated per call.
- `result.json`: inputs, candidates and selected build parameters.
- `progress.log`: terminal output, including tracebacks on failure.
