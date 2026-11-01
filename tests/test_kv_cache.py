from bench.kv_cache import GIB, AttnShape, kv_budget_bytes, kv_bytes_per_token, max_concurrent_seqs

# Qwen3-1.7B config.json values
QWEN3_1_7B = {"num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
              "head_dim": 128, "hidden_size": 2048}


def test_bytes_per_token_matches_hand_math():
    shape = AttnShape.from_hf_config(QWEN3_1_7B)
    # 2 (K,V) * 28 layers * 8 kv heads * 128 dim * 2 bytes = 114,688 B = 112 KiB
    assert kv_bytes_per_token(shape, "float16") == 114_688
    assert kv_bytes_per_token(shape, "fp8") == 57_344


def test_head_dim_fallback_when_missing():
    cfg = {k: v for k, v in QWEN3_1_7B.items() if k != "head_dim"}
    assert AttnShape.from_hf_config(cfg).head_dim == 2048 // 16


def test_capacity_on_t4():
    shape = AttnShape.from_hf_config(QWEN3_1_7B)
    budget = kv_budget_bytes(gpu_mem_gib=15.0, weight_gib=3.4, gpu_memory_utilization=0.9,
                             activation_overhead_gib=1.0)
    assert abs(budget / GIB - 9.1) < 1e-9
    bpt = kv_bytes_per_token(shape)
    # 4k seq = 448 MiB -> floor(9.1 GiB / 0.4375 GiB) = 20
    assert max_concurrent_seqs(budget, 4096, bpt) == 20
    assert max_concurrent_seqs(budget, 16384, bpt) == 5


def test_budget_never_negative():
    assert kv_budget_bytes(4.0, 10.0) == 0.0
