"""Logit-lens demo: does the residual-stream diff between (with prefill) and
(without prefill) project to 'lovely' / related tokens?

Two contexts:
  user: Say a sentence without 'lovely.'
  assistant (1): Lovely. The sky is clear today.
  assistant (2):              The sky is clear today.

We forward both, take the residual stream at the LAST token (the period after
"today") at each layer, subtract, apply final norm + lm_head (logit lens), and
print top-k tokens.
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
    ap.add_argument('--topk', type=int, default=15)
    ap.add_argument('--out', default='logit_lens_demo.json')
    args = ap.parse_args()

    device = f'cuda:{args.gpu}'
    torch.set_grad_enabled(False)

    print(f'loading {args.model} on {device}…', flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map={'': device})
    model.eval()

    msgs = [{'role': 'user', 'content': args.user}]
    base = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    text1 = base + f'{args.inject} {args.asst_tail}'
    text2 = base + f'{args.asst_tail}'

    ids1 = tok(text1, return_tensors='pt').input_ids.to(device)
    ids2 = tok(text2, return_tensors='pt').input_ids.to(device)
    print(f'tokens: ctx1={ids1.shape[1]}, ctx2={ids2.shape[1]}')
    print(f'last token: ctx1 = {tok.decode(ids1[0, -1:])!r}, ctx2 = {tok.decode(ids2[0, -1:])!r}')

    out1 = model(ids1, output_hidden_states=True)
    out2 = model(ids2, output_hidden_states=True)
    h1 = out1.hidden_states  # tuple of len n_layers+1, each [1, seq, hidden]
    h2 = out2.hidden_states
    n_layers = len(h1) - 1
    print(f'layers: {n_layers}')

    norm = model.model.norm                     # final RMSNorm
    lm_w = model.lm_head.weight                  # [vocab, hidden]

    results = []
    for L in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]:
        d = (h1[L][0, -1] - h2[L][0, -1]).to(torch.float32)
        # logit lens: pretend `d` is a residual at the model's exit, project to vocab
        d_n = norm(d.to(torch.bfloat16)).to(torch.float32)
        logits = d_n @ lm_w.T.to(torch.float32)
        top = torch.topk(logits, args.topk)
        rows = [{'tok': tok.decode([int(i)]), 'logit': float(v)}
                for v, i in zip(top.values.tolist(), top.indices.tolist())]
        results.append({'layer': L, 'topk': rows})
        print(f'\n--- layer {L}/{n_layers} ---')
        for r in rows:
            print(f'  {r["logit"]:+8.3f}  {r["tok"]!r}')

    payload = {
        'model': args.model,
        'inject': args.inject,
        'user': args.user,
        'asst_tail': args.asst_tail,
        'n_layers': n_layers,
        'last_token_ctx1': tok.decode(ids1[0, -1:]),
        'last_token_ctx2': tok.decode(ids2[0, -1:]),
        'tokens_ctx1': ids1.shape[1],
        'tokens_ctx2': ids2.shape[1],
        'results': results,
    }
    with open(args.out, 'w') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'\nsaved {args.out}')


if __name__ == '__main__':
    sys.exit(main() or 0)
