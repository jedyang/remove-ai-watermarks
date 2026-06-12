# Doubao clean-reverse-alpha distillation (re-investigated 2026-05-29)

> Relocated verbatim from `CLAUDE.md` on 2026-06-11 to keep the always-loaded
> context small. Long single-line entries were reformatted into paragraphs;
> no content was changed or summarized.

**RESOLVED 2026-05-29: black+gray Doubao captures were obtained and a reverse-alpha is built** (`doubao_engine.remove_watermark_reverse_alpha`, `assets/doubao_alpha.png`; see the `doubao_engine.py` section in `docs/module-internals.md`). The captures (`data/doubao_capture/captures/`, now committed) confirmed the alpha-composite model: on black `captured = a*logo`, logo pure white.

**UPDATE 2026-05-31 (issue #13 follow-up): the first build was NOT "exact"** — it left a readable "豆包AI生成" outline on the real sample (the detector was fooled, conf 0.0). The alpha is now rebuilt by `scripts/visible_alpha_solve.py` (the careful gray-self solve shared with Jimeng), removal always-aligns + thin-inpaints, and the locate box was widened; see the `doubao_engine.py` section in `docs/module-internals.md`. The notes below (the failed content-image distillation) are retained as the record of why controlled captures were necessary.

**Conclusion (historical): pure reverse-alpha distilled from content images does NOT work, and the blocker is the WRONG kind of data, not too little of it.**

The earlier framing ("need ~5-8 PRISTINE same-resolution originals") is obsolete -- `data/spaces/originals/` holds plenty. Curate them with `DoubaoEngine.detect` + an NCC filter against a clean glyph template, keeping only marks at offset ≈ (0,0): that yields e.g. **15 pixel-aligned 2048² marks** (sub-pixel drift, not the ±50 px the old lossy/mixed-res scrapes had), plus 1086x1448 / 1792x2400 clusters. With those, LaMa-clean `O` + weighted-LS (and per-pixel I-on-O regression) for `α` (+ logo colour) was tried end-to-end and **still leaves a persistent ghost outline.**

Diagnosed why, empirically (cached stacks, `/tmp/doubao_distill`): (1) the mark is a clean white overlay with **no dark halo** -- over glyph pixels ~54% are brighter than the clean bg, only ~4% darker -- so the white-logo model `I=(1-α)O+α·255` is correct; (2) but content backgrounds are almost never dark *under* the mark (median darkest available bg over glyph pixels = **58/255**; only ~13% of mark pixels are ever observed on a bg < 40), so on bright backgrounds the equation is ill-conditioned and `α` is unidentifiable; (3) LaMa's `O` is a plausible **hallucination**, not the true pre-mark background, which compounds the error, and per-pixel regression on ~15 obs overfits into colour noise.

**Why Gemini's engine is clean (verified in GeminiWatermarkTool `src/core/watermark_engine.cpp`): its alpha map is the watermark stamped on a PURE-BLACK background**, where `watermarked = α·255 + (1-α)·0 = α·255`, so `alpha = capture/255` exactly -- no estimation. (`gemini_bg_*.png` is literally the sparkle in grey on black.) So the real Doubao unlock is the same controlled capture, **not more content images**. Black/white/gray seeds exist (`data/doubao_capture/seeds/seed_*_1x1_2048x2048.png`); a capture run (feed a black seed through doubao.com edit mode, download the *original*) was requested from the #13 reporter 2026-05-29. With ~2-3 black captures we get `α = capture/255` for free, Gemini-quality.

**Until black captures arrive, the shipped direction is precise canonical glyph mask + inpaint (cv2 default, lama optional), NOT reverse-alpha.**

The consensus glyph silhouette across the aligned marks distills cleanly (proto: a tight "豆包AI生成" strip, width ≈ 0.156 × image-width) and is good both as an exact inpaint mask and as an NCC localiser -- the latter also fixes the #23 detector false-positives (match the real glyph shape, not any bright low-saturation corner). Do **not** retry content-image reverse-alpha: it is data-limited by physics (no dark-background observations), not by effort.
