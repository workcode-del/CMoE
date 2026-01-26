import re


attn_sizes = {"llama7b": 4096 * 4096 * 4, "dpsk-moe-16b": 2048 * 2048 * 2 + 2048 * 2048 / 16 * 2}
shared_sizes = {"llama7b": 0, "dpsk-moe-16b": 10944 * 2048 * 3 * 2}
ffn_sizes = {"llama7b": 4096 * 11008 * 3, "dpsk-moe-16b": 2048 * 1408 * 3 * 64}
layer_nums = {"llama7b": 32, "dpsk-moe-16b": 27}
# first_layer_size = {"llama7b": 0, "dpsk-moe-16b": 10944 * 2048 * 3 + attn_sizes["dpsk-moe-16b"]}
vol_sizes = {"llama7b": 32000 * 4096, "dpsk-moe-16b": 102400 * 2048}


modeltype = "llama7b"
# qscheme_strs = ["a8s0m8888", "a8s0m4222", "a8s0m3333", "a8s0m3322", "a8s0m3222", "a8s0m3221", "a8s0m2222", "a8s0m44444444", "a8s0m44332211", "a8s0m44322221", "a8s0m44222222", "a8s0m42222222", "a8s0m33333333", "a8s0m22222222", "a8s0m33222222", "a8s0m32222221", "a8s0m32222211"]
# qscheme_strs = ["a8s0m8", "a8s0m4", "a8s0m3", "a8s0m2", "a4s0m4", "a4s0m3", "a4s0m2", "a3s0m3", "a3s0m2", "a2s0m2"]

modeltype = "dpsk-moe-16b"
qscheme_strs = [
    "a8s8m8"
    "a8s4m2",
    "a8s2m2",
    "a2s2m2",
    "a8s4m22",
    "a8s2m22",
    "a8s4m3221",
    "a8s2m3222",
    "a8s2m2222",
    "a8s2m3221"]

for qscheme_str in qscheme_strs:
    if qscheme_str is not None:
        match = re.search(r'a(\d)s(\d)m(\d+)', qscheme_str)
        qscheme_attn = [int(match.group(1))]
        qscheme_share = [int(match.group(2))]
        ee = match.group(3)
        qscheme_expert = [int(e) for e in ee]

        attn_size = attn_sizes[modeltype]
        ffn_size = ffn_sizes[modeltype]

        final_size = vol_sizes[modeltype]
        # final_size += first_layer_size[modeltype] 
        final_size += attn_size * (qscheme_attn[0] + 0.25)
        final_size += shared_sizes[modeltype] * (qscheme_share[0] + 0.25)

        expert_num = len(qscheme_expert)
        print(qscheme_attn[0], qscheme_share[0], qscheme_expert)
        for e in qscheme_expert:
            final_size += ffn_size / expert_num * (e + 0.25)

        final_size *= layer_nums[modeltype]
        final_size_gb = final_size / 8 / 1024 / 1024 / 1024
        print(f"{qscheme_str}\t {final_size}b\t {final_size_gb:.4f} GB")
        # print(f"{final_size_gb :.4f}")
