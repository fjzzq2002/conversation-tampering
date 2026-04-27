"""Concept-injection vs output-injection comparison.

Recipe (following safety-research/introspection-mechanisms):
  1. Build a 'lovely' steering vector =
       activation_at_last_token('Tell me about Lovely', layer L)
       − mean over 100 baseline words of the same.
  2. Three forward passes on the same target prompt:
       A: assistant message PREFILLED with 'Lovely.' (output-level injection)
       B: no prefill, but `concept_vec * strength` added at layer L for every token (activation injection)
       C: no prefill, no steering (baseline)
  3. At each (layer, shared-suffix position), compute
       diff_A = h_A - h_C   diff_B = h_B - h_C   cos(diff_A, diff_B).
     If the hypothesis "activation ≈ output injection" holds, cos > 0 at downstream layers.

The shared suffix is the assistant's tail tokens (e.g. ' The sky is clear today.')
that appear in both A and C; we align their token positions before subtracting.
"""
from __future__ import annotations
import argparse, json, os, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



# 100 baseline words from safety-research/introspection-mechanisms (vector_utils.py)
BASELINE_WORDS = [
    "Desks","Jackets","Gondolas","Laughter","Intelligence","Bicycles","Chairs",
    "Orchestras","Sand","Pottery","Arrowheads","Jewelry","Daffodils","Plateaus",
    "Estuaries","Quilts","Moments","Bamboo","Ravines","Archives","Hieroglyphs",
    "Stars","Clay","Fossils","Wildlife","Flour","Traffic","Bubbles","Honey",
    "Geodes","Magnets","Ribbons","Zigzags","Puzzles","Tornadoes","Anthills",
    "Galaxies","Poverty","Diamonds","Universes","Vinegar","Nebulae","Knowledge",
    "Marble","Fog","Rivers","Scrolls","Silhouettes","Marbles","Cakes","Valleys",
    "Whispers","Pendulums","Towers","Tables","Glaciers","Whirlpools","Jungles",
    "Wool","Anger","Ramparts","Flowers","Research","Hammers","Clouds","Justice",
    "Dogs","Butterflies","Needles","Fortresses","Bonfires","Skyscrapers","Caravans",
    "Patience","Bacon","Velocities","Smoke","Electricity","Sunsets","Anchors",
    "Parchments","Courage","Statues","Oxygen","Time","Butterflies","Fabric","Pasta",
    "Snowflakes","Mountains","Echoes","Pianos","Sanctuaries","Abysses","Air",
    "Dewdrops","Gardens","Literature","Rice","Enigmas",
]


def chat_prompt(tok, content: str) -> str:
    msgs = [{'role': 'user', 'content': content}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def last_token_activation(model, tok, text: str, layer: int, device: str) -> torch.Tensor:
    ids = tok(text, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return out.hidden_states[layer + 1][0, -1].detach().clone()  # +1 because hidden_states[0] = embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='allenai/Olmo-3-7B-Instruct')
    ap.add_argument('--gpu', type=int, default=2)
    ap.add_argument('--concept', nargs='+', default=['Lovely'],
                    help='one or more concepts to sweep')
    ap.add_argument('--inject-token', default='Lovely.')
    ap.add_argument('--user', default="Say a sentence without 'lovely.'")
    ap.add_argument('--asst-tail', default='The sky is clear today.')
    ap.add_argument('--inject-layer', type=int, default=22,
                    help='which transformer layer to add the steering vector after (0-indexed)')
    ap.add_argument('--strength', type=float, nargs='+', default=[8.0],
                    help='one or more strengths to sweep')
    ap.add_argument('--inject-at', choices=['all', 'pre-suffix'], nargs='+',
                    default=['all', 'pre-suffix'],
                    help='where in the sequence to add the concept vector. '
                         '"all" = every position (paper-style steering); '
                         '"pre-suffix" = the single position right before the shared suffix starts.')
    ap.add_argument('--n-baseline', type=int, default=100)
    ap.add_argument('--out', default='concept_inj_compare.json')
    args = ap.parse_args()

    device = f'cuda:{args.gpu}'
    torch.set_grad_enabled(False)

    print(f'loading {args.model} on {device}…', flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={'': device})
    model.eval()
    n_layers = model.config.num_hidden_layers
    L_inject = args.inject_layer
    assert 0 <= L_inject < n_layers
    print(f'loaded ({n_layers} layers); inject layer = {L_inject}', flush=True)

    # ---- 1) Build concept vectors for every concept ----
    print('computing baseline activations…', flush=True)
    bl_words = BASELINE_WORDS[:args.n_baseline]
    baseline_acts = []
    for i, w in enumerate(bl_words):
        if i % 20 == 0: print(f'  baseline {i}/{len(bl_words)}', flush=True)
        text = chat_prompt(tok, f'Tell me about {w}')
        baseline_acts.append(last_token_activation(model, tok, text, L_inject, device))
    baseline_mean = torch.stack(baseline_acts).mean(dim=0)

    concept_vecs = {}
    for c in args.concept:
        text = chat_prompt(tok, f'Tell me about {c}')
        a = last_token_activation(model, tok, text, L_inject, device)
        v = a - baseline_mean
        concept_vecs[c] = v
        print(f'concept_vec[{c!r}] norm = {v.norm().item():.3f}', flush=True)

    # ---- 2) Build target contexts A and C (independent of concept/strength) ----
    base = chat_prompt(tok, args.user)
    text_A = base + f'{args.inject_token} {args.asst_tail}'
    text_C = base + f' {args.asst_tail}'   # leading space so 'The' tokenizes the same as in A
    ids_A = tok(text_A, return_tensors='pt').input_ids.to(device)
    ids_C = tok(text_C, return_tensors='pt').input_ids.to(device)
    seq_A, seq_C = ids_A.shape[1], ids_C.shape[1]
    K = 0
    while K < seq_A and K < seq_C and int(ids_A[0, -1 - K]) == int(ids_C[0, -1 - K]):
        K += 1
    suffix_decoded = [tok.decode([t]) for t in ids_A[0, -K:].tolist()]
    print(f'shared suffix (K={K}): {suffix_decoded}', flush=True)
    pre_suffix_pos = seq_C - K - 1   # last token before the shared suffix in ctx C

    # A and C only need to be computed once
    print('forward A (prefill output injection)', flush=True)
    out_A = model(ids_A, output_hidden_states=True)
    print('forward C (baseline, no steering)', flush=True)
    out_C = model(ids_C, output_hidden_states=True)

    target_layer = model.model.layers[L_inject]

    # ---- 3) Sweep concept × inject_at × strength ----
    runs = []  # list of {concept, inject_at, strength, grid}
    for concept in args.concept:
        v = concept_vecs[concept]
        for mode in args.inject_at:
            for strength in args.strength:
                # build delta (broadcast over positions or a single position mask)
                full_delta = (strength * v)
                def make_hook(mode=mode):
                    if mode == 'all':
                        def h_all(module, inputs, output):
                            if isinstance(output, tuple):
                                h = output[0]
                                return (h + full_delta.view(1,1,-1).to(h.dtype),) + output[1:]
                            return output + full_delta.view(1,1,-1).to(output.dtype)
                        return h_all
                    else:  # pre-suffix
                        def h_pre(module, inputs, output):
                            if isinstance(output, tuple):
                                h = output[0]
                                h = h.clone()
                                h[0, pre_suffix_pos] = h[0, pre_suffix_pos] + full_delta.to(h.dtype)
                                return (h,) + output[1:]
                            else:
                                out = output.clone()
                                out[0, pre_suffix_pos] = out[0, pre_suffix_pos] + full_delta.to(out.dtype)
                                return out
                        return h_pre
                handle = target_layer.register_forward_hook(make_hook())
                try:
                    out_B = model(ids_C, output_hidden_states=True)
                finally:
                    handle.remove()

                grid = []
                for L in range(n_layers + 1):
                    for off in range(K):
                        iA = seq_A - K + off
                        iC = seq_C - K + off
                        hA = out_A.hidden_states[L][0, iA].float()
                        hB = out_B.hidden_states[L][0, iC].float()
                        hC = out_C.hidden_states[L][0, iC].float()
                        dA = hA - hC
                        dB = hB - hC
                        nA, nB = float(dA.norm()), float(dB.norm())
                        cos = (
                            float(torch.nn.functional.cosine_similarity(dA, dB, dim=0))
                            if nA > 1e-4 and nB > 1e-4 else None
                        )
                        grid.append({'layer': L, 'pos': off, 'cos': cos,
                                     'norm_diffA': nA, 'norm_diffB': nB})

                runs.append({
                    'concept': concept,
                    'inject_at': mode,
                    'strength': strength,
                    'concept_vec_norm': float(v.norm().item()),
                    'grid': grid,
                })

                # Print summary line: cos at last layer, last position
                last = next(x for x in grid if x['layer'] == n_layers and x['pos'] == K - 1)
                print(f'  {concept:>10}  inject_at={mode:>10}  strength={strength:>5}  '
                      f'cos@(L{n_layers}, last_pos)={last["cos"]:+.3f}', flush=True)

    payload = {
        'model': args.model,
        'inject_token': args.inject_token,
        'inject_layer': L_inject,
        'n_layers': n_layers,
        'shared_suffix': suffix_decoded,
        'pre_suffix_pos': pre_suffix_pos,
        'runs': runs,
    }
    with open(args.out, 'w') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'\nsaved {args.out}', flush=True)

    # Print sweep table at last layer
    print(f'\n=== cos at (last layer L{n_layers}, last position {suffix_decoded[-1]!r}) ===')
    print(f'{"concept":>10}  {"mode":>10}  ' + ''.join(f'{f"s={s}":>10}' for s in args.strength))
    for c in args.concept:
        for m in args.inject_at:
            row = [f'{c:>10}', f'{m:>10}']
            for s in args.strength:
                r = next((r for r in runs
                          if r['concept']==c and r['inject_at']==m and r['strength']==s), None)
                if r is None:
                    row.append(f'{"-":>10}'); continue
                last = next(x for x in r['grid'] if x['layer'] == n_layers and x['pos'] == K - 1)
                row.append(f'{last["cos"]:+.3f}'.rjust(10))
            print('  '.join(row))


if __name__ == '__main__':
    sys.exit(main() or 0)
