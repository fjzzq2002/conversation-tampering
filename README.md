# Conversation Tampering

Do LLMs notice when their conversation history has been silently tampered with?

This repo contains the experiment scripts and a static transcript viewer for a study
on whether assistants flag inserted/prefilled tokens that they did not actually
output. We test prepend, prefill, and control conditions across Claude Opus 4.5 / 4.6,
Sonnet 4.5, GPT‑5.4, and OLMo‑3.1 (32B Instruct/SFT/DPO) and OLMo‑3 7B Instruct.

## Live viewer

GitHub Pages is enabled on `main` at the repo root. Once published the viewer is at:

  https://fjzzq2002.github.io/conversation-tampering/

Each session has a permalink of the form
`?id=<model>-<setup>-<word>-<run>` (e.g. `?id=op45-pf-l-06`),
or `?id=<...>-tNN` to jump to a specific turn.

## Repo layout

```
.
├── index.html          # transcript viewer (single-page app)
├── data/               # 740 sanitized session JSONs + manifest.json
├── scripts/            # experiment + analysis pipeline (see scripts/README.md)
└── .gitignore
```

The `data/` directory is **derived** from sanitized logs (no thinking traces or
encrypted reasoning blobs). The `scripts/build_viewer_data.py` script that
generates `data/` hard-fails if it ever encounters a `thinking` or
`encrypted_content` field.

## Citation

If you use this dataset or the experiment design, please cite the [accompanying
LessWrong post](https://www.lesswrong.com/posts/yAR6uMdSaBjkbJ4u9/spontaneous-introspection-in-conversation-tampering).
