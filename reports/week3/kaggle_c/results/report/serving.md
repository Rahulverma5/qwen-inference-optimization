| Experiment | Status | Change | Units | Exec acc | Weights GiB | KV tokens | Ready s | Peak tok/s | Max c @SLO | tok/s @SLO | req/s @SLO | $/1M out tok | $/1k queries | Spec accept |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q4_k_m | ok | - | 1 | 64.3% | - | - | 6 | 6 | 0 | - | - | - | - | - |
| q8_0 | ok | - | 1 | 64.5% | - | - | 3 | 5 | 0 | - | - | - | - | - |
| threads_2 | ok | threads=2 | 1 | - | - | - | 3 | 6 | 1 | 6 | 0.19 | 18.764 | 0.5590 | - |
| parallel_1 | ok | parallel=1, ctx-size=2048 | 1 | - | - | - | 3 | 6 | 0 | - | - | - | - | - |
| parallel_8 | ok | parallel=8, ctx-size=16384 | 1 | - | - | - | 6 | 7 | 0 | - | - | - | - | - |
| no_prompt_cache | ok | cache-prompt=False | 1 | - | - | - | 3 | 5 | 0 | - | - | - | - | - |
| host_cache_1g | ok | cache-ram=1024 | 1 | - | - | - | 3 | 9 | 0 | - | - | - | - | - |
| spec_ngram | server_failed | spec-type=ngram-simple, draft-max=4 | 1 | - | - | - | 3 | - | 0 | - | - | - | - | - |
| flash_attn_off | ok | flash-attn=off | 1 | - | - | - | 3 | 7 | 1 | 6 | 0.21 | 16.800 | 0.4991 | - |
| kv_q8_0 | ok | cache-type-k=q8_0, cache-type-v=q8_0 | 1 | - | - | - | 3 | 6 | 0 | - | - | - | - | - |
