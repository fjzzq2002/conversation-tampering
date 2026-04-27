"""35x35 cosine-similarity grid: prefill X (rows) × steering Y (cols).

For each prefill concept X:
  - run A_X: assistant prefilled with 'X.' (user msg: 'Say a sentence without ...')
  - run C_X: baseline (no prefill, no steering)
  - for each steering concept Y:
      run B_X,Y: same baseline but with Y's steering vector at L=16
      cell[X][Y] = average over (layer L_inj+1..n_layers, position) of
                   cos(h_A_X - h_C_X, h_B_X,Y - h_C_X)

Saves grid + per-row/col ranks of the diagonal.
"""
from __future__ import annotations
import argparse, json, os, sys
import torch
import statistics
from transformers import AutoTokenizer, AutoModelForCausalLM



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

# 35 concepts: 5 test + 30 controls
ALL_CONCEPTS = [
    'Lovely','Sure','Assistant','Bread','Justice',
    'Cat','Blue','Tree','Run','Apple','Light','Stone','River','Book','Cloud',
    'Luminous','Gravel','Whisper','Alloy','Mirage','Tundra','Velvet','Orbit',
    'Lantern','Mosaic','Entropy','Liminality','Resonance','Friction','Emergence',
    'Decay','Symmetry','Drift','Abstraction','Saturation',
]

# Prefill token + user prompt template; treat all uniformly (Word + period)
def prefill_token(c: str) -> str:
    return f'{c}:' if c == 'Assistant' else f'{c}.'

def user_prompt(c: str) -> str:
    # Matches gaslight.make_word_config: always 'Say a sentence without '<word>.''
    # regardless of decoration choice for the prefill token.
    return f"Say a sentence without '{c.lower()}.'"


def chat_prompt(tok, content: str) -> str:
    msgs = [{'role': 'user', 'content': content}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def last_token_activation(model, tok, text: str, layer: int, device: str) -> torch.Tensor:
    ids = tok(text, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return out.hidden_states[layer + 1][0, -1].detach().clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='allenai/Olmo-3-7B-Instruct')
    ap.add_argument('--gpu', type=int, default=2)
    ap.add_argument('--inject-layer', type=int, default=16)
    ap.add_argument('--strength', type=float, default=0.05)
    ap.add_argument('--asst-tail', default='The sky is clear today.')
    ap.add_argument('--n-baseline', type=int, default=100)
    ap.add_argument('--out', default='concept_grid_s05.json')
    args = ap.parse_args()

    device = f'cuda:{args.gpu}'
    torch.set_grad_enabled(False)

    print(f'loading {args.model} on {device}…', flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={'': device})
    model.eval()
    n_layers = model.config.num_hidden_layers
    L_inj = args.inject_layer

    # baseline mean
    print('computing baseline activations…', flush=True)
    baseline_acts = []
    for i, w in enumerate(BASELINE_WORDS[:args.n_baseline]):
        if i % 20 == 0: print(f'  baseline {i}/{args.n_baseline}', flush=True)
        text = chat_prompt(tok, f'Tell me about {w}')
        baseline_acts.append(last_token_activation(model, tok, text, L_inj, device))
    baseline_mean = torch.stack(baseline_acts).mean(dim=0)

    # concept vectors for all 35
    print('computing concept vectors…', flush=True)
    concept_vecs = {}
    for c in ALL_CONCEPTS:
        text = chat_prompt(tok, f'Tell me about {c}')
        a = last_token_activation(model, tok, text, L_inj, device)
        concept_vecs[c] = a - baseline_mean

    target_layer = model.model.layers[L_inj]

    # main grid loop
    grid = {}  # X -> {Y: cos_value}
    suffix_decoded = None
    for ix, X in enumerate(ALL_CONCEPTS):
        print(f'\n--- prefill X={X} ({ix+1}/{len(ALL_CONCEPTS)}) ---', flush=True)
        base = chat_prompt(tok, user_prompt(X))
        text_A = base + f'{prefill_token(X)} {args.asst_tail}'
        text_C = base + f' {args.asst_tail}'   # leading space hack
        ids_A = tok(text_A, return_tensors='pt').input_ids.to(device)
        ids_C = tok(text_C, return_tensors='pt').input_ids.to(device)
        seq_A, seq_C = ids_A.shape[1], ids_C.shape[1]

        # shared suffix
        K = 0
        while K < seq_A and K < seq_C and int(ids_A[0, -1 - K]) == int(ids_C[0, -1 - K]):
            K += 1
        if suffix_decoded is None:
            suffix_decoded = [tok.decode([t]) for t in ids_A[0, -K:].tolist()]
            print(f'shared suffix (K={K}): {suffix_decoded}', flush=True)

        out_A = model(ids_A, output_hidden_states=True)
        out_C = model(ids_C, output_hidden_states=True)

        row = {}
        for Y in ALL_CONCEPTS:
            v = concept_vecs[Y]
            full_delta = (args.strength * v).view(1, 1, -1)
            def hook(module, inputs, output, _d=full_delta):
                if isinstance(output, tuple):
                    h = output[0]
                    return (h + _d.to(h.dtype),) + output[1:]
                return output + _d.to(output.dtype)
            handle = target_layer.register_forward_hook(hook)
            try:
                out_B = model(ids_C, output_hidden_states=True)
            finally:
                handle.remove()

            # aggregate cos over (layer L_inj+1..n_layers, all suffix positions)
            cos_vals = []
            for L in range(L_inj + 1, n_layers + 1):
                for off in range(K):
                    iA = seq_A - K + off
                    iC = seq_C - K + off
                    hA = out_A.hidden_states[L][0, iA].float()
                    hB = out_B.hidden_states[L][0, iC].float()
                    hC = out_C.hidden_states[L][0, iC].float()
                    dA, dB = hA - hC, hB - hC
                    nA, nB = float(dA.norm()), float(dB.norm())
                    if nA > 1e-4 and nB > 1e-4:
                        cos_vals.append(float(torch.nn.functional.cosine_similarity(dA, dB, dim=0)))
            row[Y] = statistics.mean(cos_vals) if cos_vals else None
            del out_B
        grid[X] = row

        # quick row summary
        diag = row[X]
        rk = sum(1 for c in row.values() if c is not None and c >= diag) if diag is not None else None
        print(f'  diag cos={diag:+.3f}, rank={rk}/{len(row)}' if diag is not None else '  diag missing')

    payload = {
        'model': args.model,
        'inject_layer': L_inj,
        'strength': args.strength,
        'concepts': ALL_CONCEPTS,
        'shared_suffix_first': suffix_decoded,
        'grid': grid,
    }
    with open(args.out, 'w') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'\nsaved {args.out}', flush=True)


if __name__ == '__main__':
    sys.exit(main() or 0)
