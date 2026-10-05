"""
ReDeEP (Sun et al., ICLR 2025) Appendix B copying-head score, implemented for
Mistral-7B (GQA: 32 query heads, 8 KV heads) without ever materialising M.

Per head, M = W_E @ W_V @ W_O @ W_U  (V x V) = A @ B with
    A = W_E @ W_V   [V, d_head]      (shared by the 4 heads of a KV group)
    B = W_O @ W_U   [d_head, V]

Steps (as in the paper):
  1. trace            tr(M) = sum_i a_ii,  a_ii = sum_k A[i,k] * B[k,i]
  2. Gershgorin       R_i = sum_{j != i} |M_ij|   (computed in row chunks)
  3. IQR outliers     boundary points a_ii +/- R_i, count points outside
                      [Q1 - 1.5 IQR, Q3 + 1.5 IQR]
  4. rank sum         rank(#outliers, ascending) + rank(|trace|, descending),
                      ranked over ALL heads in the model.
                      Lower rank sum = more copy-like (under the default reading).

The paper does not state the direction of the final ranking, so the defaults
below are an interpretation. Flags let you flip them. For reference, the script
also computes the exact eigenvalue statistic from the small d_head x d_head matrix
K = W_O (W_U W_E) W_V, and reports its Spearman correlation with the heuristic.

RMSNorm gains are ignored (as in Elhage et al., 2021 and the paper's formulas).
A GPU is strongly recommended: ~1024 heads x a 32000x32000 product of rank 128.
"""
import argparse

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from transformers import AutoModelForCausalLM


def gershgorin_stats(A, B, chunk):
    """Return (trace, #IQR outliers among Gershgorin boundary points) for M = A @ B."""
    diag = (A * B.T).sum(1)                                  # a_ii
    row_abs = torch.empty_like(diag)
    for i0 in range(0, A.shape[0], chunk):
        row_abs[i0:i0 + chunk] = (A[i0:i0 + chunk] @ B).abs().sum(1)
    R = (row_abs - diag.abs()).clamp_min(0)                  # sum over j != i of |M_ij|
    pts = torch.cat([diag - R, diag + R])
    q = torch.quantile(pts, torch.tensor([0.25, 0.75], device=pts.device))
    iqr = q[1] - q[0]
    n_out = ((pts < q[0] - 1.5 * iqr) | (pts > q[1] + 1.5 * iqr)).sum().item()
    return diag.sum().item(), n_out


def exact_stats(W_O, G, W_V):
    """Exact spectrum of the head's circuit via the d_head x d_head matrix K."""
    K = W_O @ G @ W_V
    eig = torch.linalg.eigvals(K.cpu())
    return (eig.real.sum() / eig.abs().sum()).item(), torch.trace(K).item()


def rank_sum_score(trace, n_out, signed_trace=False, outliers_high_is_copy=False):
    t = trace if signed_trace else np.abs(trace)
    r_out = rankdata(-n_out if outliers_high_is_copy else n_out, method="min")  # ascending
    r_trace = rankdata(-t, method="min")                                         # descending
    return r_out + r_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--chunk", type=int, default=2048, help="rows per chunk for row sums")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--tf32", action="store_true", help="faster matmuls, slightly less precise")
    ap.add_argument("--signed-trace", action="store_true",
                    help="rank by signed trace instead of |trace| (paper uses |trace|)")
    ap.add_argument("--outliers-high-is-copy", action="store_true",
                    help="flip the outlier-count ranking direction")
    ap.add_argument("--out", default="redeep_copy_scores.npz")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("WARNING: running on CPU will be extremely slow.")
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    cfg = model.config
    L, H, KV = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    d_head = cfg.hidden_size // H
    group = H // KV

    W_E = model.get_input_embeddings().weight.to(dev, torch.float32)     # [V, d]
    W_U = model.get_output_embeddings().weight.to(dev, torch.float32).T  # [d, V]
    G = W_U @ W_E                                                        # [d, d]

    trace = np.zeros((L, H))
    n_out = np.zeros((L, H))
    exact_ratio = np.zeros((L, H))
    trace_exact = np.zeros((L, H))

    with torch.no_grad():
        for l, layer in enumerate(model.model.layers):
            Wv = layer.self_attn.v_proj.weight.to(dev, torch.float32)    # [KV*d_head, d]
            Wo = layer.self_attn.o_proj.weight.to(dev, torch.float32)    # [d, H*d_head]
            for g in range(KV):
                W_V = Wv[g * d_head:(g + 1) * d_head].T                  # [d, d_head]
                A = W_E @ W_V                                            # [V, d_head]
                for h in range(g * group, (g + 1) * group):
                    W_O = Wo[:, h * d_head:(h + 1) * d_head].T           # [d_head, d]
                    B = W_O @ W_U                                        # [d_head, V]
                    trace[l, h], n_out[l, h] = gershgorin_stats(A, B, args.chunk)
                    exact_ratio[l, h], trace_exact[l, h] = exact_stats(W_O, G, W_V)
            print(f"layer {l + 1}/{L} done", flush=True)

    # sanity check: tr(M) from the big matrix must equal tr(K) from the small one
    rel = np.abs(trace - trace_exact) / (np.abs(trace_exact) + 1e-8)
    print(f"max relative trace mismatch (big vs small matrix): {rel.max():.2e}")

    score = rank_sum_score(trace.ravel(), n_out.ravel(),
                           args.signed_trace, args.outliers_high_is_copy).reshape(L, H)

    rho = spearmanr(-score.ravel(), exact_ratio.ravel()).correlation
    print(f"Spearman(heuristic copy-ness, exact eigenvalue ratio) = {rho:.3f}")

    np.savez(args.out, trace=trace, n_outliers=n_out, score=score,
             exact_ratio=exact_ratio, trace_exact=trace_exact)

    order = np.argsort(score.ravel(), kind="stable")[:args.topk]
    print(f"\ntop {args.topk} heads by ReDeEP-style score (lower = more copy-like):")
    for idx in order:
        l, h = divmod(idx, H)
        print(f"L{l:02d}H{h:02d}  score={score[l, h]:.0f}  trace={trace[l, h]:+.3e}  "
              f"outliers={int(n_out[l, h])}  exact_ratio={exact_ratio[l, h]:+.3f}")


if __name__ == "__main__":
    main()
