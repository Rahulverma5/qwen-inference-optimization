| Experiment | Status | Change | GPUs | Exec acc | Weights GiB | KV tokens | Ready s | Peak tok/s | Max c @SLO | tok/s @SLO | req/s @SLO | $/1M out tok | $/1k queries | Spec accept |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp16 | ok | - | 1 | 64.7% | 3.22 | 86,400 | 103 | 481 | 4 | 142 | 4.51 | 0.686 | 0.0216 | - |
| awq_w4a16 | missing_model | - | 1 | - | - | - | - | - | 0 | - | - | - | - | - |
| gptq_w4a16 | ok | - | 1 | 62.4% | 1.27 | 104,640 | 106 | 505 | 16 | 354 | 11.90 | 0.275 | 0.0082 | - |
| w8a8_int8 | ok | - | 1 | 64.0% | 1.90 | 98,640 | 91 | 513 | 4 | 167 | 5.39 | 0.582 | 0.0180 | - |
| attn_flex | ok | attention-backend=FLEX_ATTENTION | 1 | 64.7% | 3.22 | 86,912 | 76 | 275 | 1 | 44 | 1.40 | 2.209 | 0.0694 | - |
| eager | ok | enforce-eager=True | 1 | 64.7% | 3.22 | 86,992 | 46 | 458 | 1 | 30 | 0.95 | 3.260 | 0.1024 | - |
| kv_fp8_e5m2 | server_failed | kv-cache-dtype=fp8_e5m2 | 1 | - | 3.22 | - | 46 | - | 0 | - | - | - | - | - |
| kv_fp8_e4m3 | server_failed | kv-cache-dtype=fp8 | 1 | 64.7% | 3.22 | - | 61 | - | 0 | - | - | - | - | - |
| util_0.5 | ok | gpu-memory-utilization=0.5 | 1 | 64.7% | 3.22 | 32,304 | 61 | - | 0 | - | - | - | - | - |
| capacity_awq_kv_fp8 | missing_model | kv-cache-dtype=fp8_e5m2 | 1 | - | - | - | - | - | 0 | - | - | - | - | - |
| prefix_cache_off | ok | enable-prefix-caching=False | 1 | 64.7% | 3.22 | 86,832 | 64 | 230 | 4 | 119 | 3.78 | 0.819 | 0.0257 | - |
| max_seqs_16 | ok | max-num-seqs=16 | 1 | 64.7% | 3.22 | 91,024 | 64 | 324 | 4 | 136 | 4.34 | 0.714 | 0.0224 | - |
| chunk_512 | ok | max-num-batched-tokens=512 | 1 | 64.7% | 3.22 | 86,496 | 73 | - | 0 | - | - | - | - | - |
| chunk_8192 | ok | max-num-batched-tokens=8192 | 1 | 64.7% | 3.22 | 85,952 | 76 | - | 0 | - | - | - | - | - |
| chunked_prefill_off | ok | enable-chunked-prefill=False, max-num-batched-tokens=4096 | 1 | 64.7% | 3.22 | 86,240 | 76 | - | 0 | - | - | - | - | - |
| spec_ngram | ok | speculative-config={'method': 'ngram', 'num_speculative_tokens': 4, 'prompt_lookup_max': 4, 'prompt_lookup_min': 2} | 1 | 64.7% | 3.22 | 81,408 | 73 | 480 | 4 | 144 | 4.60 | 0.673 | 0.0211 | 25% |
| spec_draft_0.6b | ok | speculative-config={'method': 'draft_model', 'model': 'Qwen/Qwen3-0.6B', 'num_speculative_tokens': 4} | 1 | 64.7% | 4.33 | 35,504 | 106 | 264 | 0 | - | - | - | - | 43% |
| tp2 | ok | - | 2 | 64.7% | 1.61 | 203,520 | 122 | 829 | 16 | 432 | 14.53 | 0.450 | 0.0134 | - |
| replicas2 | ok | - | 2 | 64.7% | 3.22 | 173,664 | 58 | 836 | 16 | 329 | 11.06 | 0.591 | 0.0176 | - |
