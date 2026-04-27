"""Logit-lens sweep over (layer, shared-suffix position).

Two contexts:
  ctx1: <user>Say a sentence without 'lovely.'</user> <asst>Lovely. The sky is clear today.</asst>
  ctx2: <user>Say a sentence without 'lovely.'</user> <asst>             The sky is clear today.</asst>

For each layer 0..N, for each position in the shared suffix " The sky is clear today.",
take the residual stream diff (h1 - h2) at that suffix position, project through the
final norm + lm_head, and record top-K tokens. Also record the rank/logit of
'lovely'/'Lovely' tokens specifically so the hypothesis can be checked numerically.
"""
from __future__ import annotations
import argparse, json, os, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='allenai/Olmo-3-7B-Instruct')
    ap.add_argument('--gpu', type=int, default=2)
    ap.add_argument('--inject', default='Lovely.')
    ap.add_argument('--user', default="Say a sentence without 'lovely.'")
    ap.add_argument('--asst-tail', default='The sky is clear today.')
    ap.add_argument('--track', default='lovely,Lovely',
                    help='comma-separated word variants to track (rank/logit)')
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--out', default='logit_lens_sweep.json')
    args = ap.parse_args()

    device = f'cuda:{args.gpu}'
    torch.set_grad_enabled(False)

    print(f'loading {args.model} on {device}…', flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map={'': device})
    model.eval()
    print('loaded', flush=True)

    msgs = [{'role': 'user', 'content': args.user}]
    base = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # Match a leading space before the tail in ctx2 so both contexts tokenize
    # 'The' the same way (' The'); without this, ctx1 has ' The' and ctx2 has
    # 'The' as different tokens, and the shared suffix is shorter by one.
    text1 = base + f'{args.inject} {args.asst_tail}'
    text2 = base + f' {args.asst_tail}'

    ids1 = tok(text1, return_tensors='pt').input_ids.to(device)
    ids2 = tok(text2, return_tensors='pt').input_ids.to(device)
    seq1, seq2 = ids1.shape[1], ids2.shape[1]
    print(f'tokens: ctx1={seq1}, ctx2={seq2}', flush=True)

    # Identify the shared suffix: largest K such that ids1[-K:] == ids2[-K:]
    K = 0
    while K < seq1 and K < seq2 and int(ids1[0, -1-K]) == int(ids2[0, -1-K]):
        K += 1
    print(f'shared suffix length: {K} tokens', flush=True)
    suffix_tokens = ids1[0, -K:].tolist()
    suffix_decoded = [tok.decode([t]) for t in suffix_tokens]
    print('suffix tokens:', suffix_decoded, flush=True)

    out1 = model(ids1, output_hidden_states=True)
    out2 = model(ids2, output_hidden_states=True)
    h1 = out1.hidden_states  # tuple of len n_layers+1, each [1, seq, hidden]
    h2 = out2.hidden_states
    n_layers = len(h1) - 1
    print(f'layers: {n_layers}', flush=True)

    norm = model.model.norm                  # final RMSNorm
    lm_w = model.lm_head.weight              # [vocab, hidden]
    lm_w_f32 = lm_w.T.to(torch.float32)

    # token ids for tracked variants (rank/logit)
    base_variants = [v.strip() for v in args.track.split(',') if v.strip()]
    tracked = []
    for v in base_variants:
        for prefix in ('', ' '):
            tracked.append(prefix + v)
    lovely_ids = {}
    for v in tracked:
        enc = tok(v, add_special_tokens=False).input_ids
        if len(enc) == 1:
            lovely_ids[v] = enc[0]
    print('tracked tokens:', lovely_ids, flush=True)

    # full sweep: for each layer 0..n_layers, for each suffix position 0..K-1,
    # compute diff at that position, logit lens, record top-k + lovely stats
    grid = []
    for L in range(n_layers + 1):
        for off in range(K):
            i1 = seq1 - K + off  # absolute index in ctx1
            i2 = seq2 - K + off  # absolute index in ctx2
            d = (h1[L][0, i1] - h2[L][0, i2]).to(torch.float32)
            d_n = norm(d.to(torch.bfloat16)).to(torch.float32)
            logits = d_n @ lm_w_f32
            top = torch.topk(logits, args.topk)
            sorted_logits = torch.argsort(logits, descending=True)
            entry = {
                'layer': L,
                'pos': off,                       # 0 = first shared token, K-1 = last (final '.')
                'token_at_pos': suffix_decoded[off],
                'topk': [
                    {'tok': tok.decode([int(i)]), 'logit': float(v)}
                    for v, i in zip(top.values.tolist(), top.indices.tolist())
                ],
                'lovely_rank': {},
                'lovely_logit': {},
            }
            # rank of lovely variants
            ranks_lookup = {int(t): r for r, t in enumerate(sorted_logits.tolist())}
            for v, tid in lovely_ids.items():
                entry['lovely_rank'][v] = ranks_lookup.get(int(tid))
                entry['lovely_logit'][v] = float(logits[int(tid)])
            grid.append(entry)
        print(f'layer {L}/{n_layers} done', flush=True)

    payload = {
        'model': args.model,
        'inject': args.inject,
        'user': args.user,
        'asst_tail': args.asst_tail,
        'n_layers': n_layers,
        'shared_suffix_tokens': suffix_decoded,
        'K': K,
        'tokens_ctx1': seq1,
        'tokens_ctx2': seq2,
        'lovely_token_ids': lovely_ids,
        'topk': args.topk,
        'grid': grid,
    }
    with open(args.out, 'w') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'\nsaved {args.out}', flush=True)


if __name__ == '__main__':
    sys.exit(main() or 0)
