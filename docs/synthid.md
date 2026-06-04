# SynthID-Image: technical reference

This document covers how Google SynthID for images works mechanically, what it
survives, what removes it, and the current deployment landscape. It is written
for engineers working on watermark detection and removal -- specifically to
inform decisions about strength settings, test methodology, and what oracle
results mean.

Primary sources are cited inline. Marketing-only claims are flagged separately
from independently-verified results.

---

## 1. Mechanism

### 1.1 Post-hoc, model-independent design

SynthID-Image is **not** baked into a diffusion model's weights. It is a
post-hoc, model-independent system: a separate encoder `f` is applied to an
already-generated image, and a separate decoder `g` reads it back.

> "We deliberately designed SynthID-Image as a post-hoc, model-independent
> approach, a choice largely based on deployment considerations."
> -- Gowal et al., arXiv:2510.09263

The formal definition from the paper:

> "A post-hoc watermarking scheme is a pair f, g consisting of an encoder
> function f: X -> X, which adds an identification mark, and a decoder
> function g: X -> {+-1}, which tries to detect if the mark is present."

This is the key architectural fact: **the generative model (Imagen, Gemini's
image model) is not modified**. The watermark is stamped onto the pixel output
after generation, by a separate neural network. This means:

- The watermark is in **pixel space**, not in the model's latent activations.
- Replacing the generative model does not remove the watermarking capability.
- The encoder/decoder pair can be updated independently of the generative model.

The paper does not disclose the internal architecture of the encoder/decoder
networks (layer types, capacity). The external variant SynthID-O is available
to partners; the production internal variant is not published.

### 1.2 How it differs from classical DWT-DCT watermarks

The open watermarks used by Stable Diffusion / SDXL / FLUX (via the
`imwatermark` library) use classical **DWT-DCT** frequency-domain embedding: a
fixed bit pattern is added to specific frequency coefficients of the image's
wavelet transform. This is fast, key-free, and locally detectable with a public
decoder.

SynthID-Image uses **jointly-trained deep learning models**:

> "SynthID uses two deep learning models -- for watermarking and identifying --
> that have been trained together on a diverse set of images. The combined model
> is optimised on a range of objectives, including correctly identifying
> watermarked content and improving imperceptibility by visually aligning the
> watermark to the original content."
> -- Google DeepMind blog, 2023

The practical difference for robustness: the deep learning encoder learns to
spread the signal across the image in a way that is optimized to survive a
specific perturbation distribution seen during training. Classical DWT-DCT
embeds in fixed, predictable frequency bins, making it brittle to any
operation that hits those bins (e.g., JPEG re-quantization wipes it cleanly at
quality <= 90).

### 1.3 Payload capacity

SynthID-O (the external/partnership variant) encodes:

- **136 bits** within a **512x512 pixel image**

For comparison (from the same paper):

| Method      | Bits | Resolution |
|-------------|------|------------|
| SynthID-O   | 136  | 512x512    |
| StegaStamp  | 100  | 400x400    |
| TrustMark   | 100  | 256x256    |
| WAM         | 32   | 256x256    |

The payload carries an identification mark (not a user-readable secret). The
paper separates watermark **detection** (is this watermarked?) from payload
**recovery** (what does the payload say?): the detection path is what oracles
like the Gemini app's "Verify with SynthID" exercise.

### 1.4 Where in the pipeline it lives

```
[Diffusion model]
       |
  raw pixel output
       |
  [SynthID encoder f]   <-- separate neural net, stamps the watermark
       |
  watermarked image
       |
  [served / downloaded]
       |
  [SynthID decoder g]   <-- separate neural net, run by Google's verifier only
       |
  present / not present
```

The VAE decoder of the diffusion model is **not** involved in watermarking.
Some in-generation watermark approaches (like the research method "Tree Ring")
inject the signal into the initial noise latent so it propagates through the
diffusion process and appears in the final image; SynthID-Image does not do
this -- it is applied after the VAE has already decoded latents to pixels.

---

## 2. Robustness

### 2.1 What the paper claims it survives (primary-source verified)

The SynthID-Image paper (arXiv:2510.09263) evaluates SynthID-O against **30
image transformations** grouped into 6 categories:

| Category    | Examples                                      |
|-------------|-----------------------------------------------|
| Color       | brightness, contrast, saturation, hue shifts  |
| Combination | combinations of multiple transforms           |
| Noise       | Gaussian noise, impulse noise, median filter  |
| Overlay     | text overlays, logos, stickers                |
| Quality     | JPEG compression, WebP, format conversion     |
| Spatial     | crop, resize, rotate, flip, padding           |

**TPR at 0.1% FPR -- SynthID-O vs. baselines (resized to 512x512):**

| Category         | SynthID-O | Best baseline (WAM) | Worst baseline (StegaStamp spatial) |
|------------------|-----------|---------------------|--------------------------------------|
| Identity (none)  | 100.00%   | 100.00%             | 100.00%                              |
| Aggregated       | 99.98%    | 90.62%              | ~70%                                 |
| Color            | 100.00%   | 81.29%              | ~75%                                 |
| Combination      | 99.96%    | 96.08%              | ~22%                                 |
| Noise            | 99.98%    | 100.00%             | ~92%                                 |
| Overlay          | 100.00%   | 100.00%             | 100.00%                              |
| Quality          | 99.99%    | --                  | ~89%                                 |
| Spatial (worst)  | 99.97%    | 76.04%              | 15.25%                               |

The "Spatial worst" row is the hardest case (aggressive crop + resize).
SynthID-O retains 99.97% TPR; StegaStamp collapses to 15.25%. This is where
the deep-learning approach gains the most over classical methods.

Google's marketing page states the watermark is:

> "designed to stand up to modifications like cropping, adding filters, changing
> frame rates, or lossy compression."
> -- deepmind.google/models/synthid/

The marketing claim is broadly consistent with the paper's numbers for these
specific categories.

**JPEG and format conversion specifically** fall under the "Quality" category,
where SynthID-O achieves 99.99% TPR. This is the empirical basis for the fact
that **GitHub-recompressed JPEGs from issue attachments are valid SynthID test
subjects**: the re-encoding does not remove the pixel watermark.

### 2.2 Stated limits (vendor claim, not independently verified)

> "SynthID isn't foolproof against extreme image manipulations."
> -- Google DeepMind blog, 2023

This is the only public failure-mode statement Google has made. No specific
perturbation type, threshold, or quantitative boundary is named. The
Limitations section of the paper (Section 10) was not recoverable from the
public HTML version of arXiv:2510.09263v1 due to a rendering failure in the
conversion (the body text of Section 10 is absent from the HTML).

**What is known empirically from our own oracle-verified testing.**

A controlled study (June 2026, clean v0.8.6 with text/face protection OFF,
native resolution on this repo's default SDXL pipeline) measured the minimum
img2img strength that removes the SynthID pixel watermark, verified per image on
the vendor's own oracle (openai.com/verify for OpenAI, the Gemini app "Verify
with SynthID" for Google). Each subject is archived in `data/synthid_corpus/` as a
pos original plus its minimum-clearing cleaned output (manifest `verified_via` =
`openai-verify` / `gemini-app`), EXCEPT one third-party image from issue #14, which
was oracle-verified but is not committed (third-party content stays out of the
public corpus).

| Vendor | Images | Resolution(s) | Pipeline | Removed at |
|--------|--------|---------------|----------|------------|
| OpenAI (gpt-image) | n=4 (3 archived + 1 external-only) | 1024x1536 .. 1600x1600 | native | **0.05** |
| Google (Gemini)    | n=4 | 2816x1536 -> capped 1536 | `--max-resolution 1536` | **0.15** (0.05 and 0.10 do NOT clear) |

**Two findings, both oracle-verified:**

1. **Vendor is the dominant factor, not resolution.** Google's SynthID is
   roughly 3x more robust than OpenAI's: at a comparable (small) working
   resolution, OpenAI clears at 0.05 while Google needs 0.15. This matches
   Google having hardened SynthID more aggressively over time.

2. **OpenAI SynthID removal is resolution-independent in the tested range.**
   All four OpenAI images (including a 1600x1600) cleared at 0.05.

**CORRECTION (supersedes the earlier "resolution dependence" claim).** A prior
version of this doc and CLAUDE.md stated that strength 0.30 failed to remove
SynthID on 1600x1600 gpt-image and that removal was resolution-dependent. That
was a **measurement artifact of a since-removed per-region re-scrub step** (issue
#14): on the dense-text infographics tested, that step could reconstitute SynthID
in text regions. Re-running the *same* 1600x1600 image on the clean current
pipeline removes SynthID at **0.05**. The "large images resist removal" conclusion
was false; the resistance was that region-rescrub shielding, since removed.

**Open / not locally testable:**

- **Native large Gemini (2816x1536, ~4.3 MP).** The Gemini floor of 0.15 was
  measured on the *capped* (`--max-resolution 1536`) path, which is the
  practical local route on Apple-Silicon (native 2816 OOMs / falls back to slow
  CPU on a 32 GB M-series). Native large Gemini was not measured here; the
  vendor and resolution effects would stack, so it plausibly needs >= 0.30 or a
  discrete GPU. Confirm on a CUDA box if needed.
- **Heavy JPEG compression** (quality < ~50-60): not oracle-tested; the DL
  approach is more robust than DWT-DCT but Google acknowledges limits at
  "extreme" manipulation.

### 2.3 Removal attacks and forensic detectability

The paper arXiv:2605.09203 ("Removing the Watermark Is Not Enough",
Goonatilake & Ateniese, 2026) evaluates 6 removal attacks against a ResNet-50
forensic detector. All attacks defeat the watermark verifier but are detected
by the forensic classifier:

| Attack          | Family           | AUROC  | TPR @ 1% FPR | TPR @ 0.1% FPR |
|-----------------|------------------|--------|--------------|----------------|
| UnMarker        | Distortion       | 0.9994 | 99.81%       | 98.28%         |
| WatermarkAttacker| Regeneration    | 0.9997 | 99.95%       | 99.38%         |
| CtrlRegen+      | Regeneration     | 0.9999 | 99.97%       | 99.64%         |
| NFPA            | Inversion/Pert.  | 0.9984 | 99.24%       | 62.10%         |
| Boundary Leak.  | Inversion/Pert.  | 0.9991 | 99.24%       | 88.34%         |
| WiTS            | Erosion          | 0.9999 | 99.80%       | 99.55%         |

The forensic detector is a standard ResNet-50 fine-tuned end-to-end; no exotic
architecture needed. The key finding:

> "These removers do not return images to a clean forensic state. They often
> trade an explicit watermark for an implicit watermark: a detectable artifact
> introduced by the removal process itself."

This means: even when our SDXL img2img pass defeats the SynthID pixel
watermark (oracle reads negative), the output may still be classifiable as
"an image that went through a removal pipeline" by an independent detector --
even if that detector is not trained on SynthID specifically. **Defeating the
verifier does not restore forensic deniability.**

CtrlRegen+ is the most detectable removal method (AUROC 0.9999), which is
notable because it is also the most powerful removal attack. The paper notes
that diffusion regeneration "leaves a strong reconstruction signature from the
diffusion prior."

---

## 3. Detectability and verifier access

### 3.1 No public local detector

The SynthID decoder is proprietary and not released:

> "SynthID-Image has been used to watermark over ten billion images and video
> frames across Google's services and its corresponding verification service is
> available to trusted testers."
> -- Gowal et al., arXiv:2510.09263

There is no public API, no released decoder weights, and no reproducible
algorithm for local detection. The verification service (SynthID Detector) is:

> "a verification portal" in early testing with "journalists and media
> professionals" on a waitlist
> -- deepmind.google/models/synthid/

The external variant SynthID-O is available "through partnerships" only. Our
tool cannot locally detect SynthID presence or absence -- this is by design,
not a gap we can fill.

### 3.2 How our tool detects SynthID (metadata proxy)

We detect SynthID indirectly: if the image's C2PA manifest is signed by a
known SynthID-using issuer (Google, OpenAI), we infer SynthID is present. This
is a **metadata proxy**, not a pixel watermark decode. It works while the C2PA
manifest is intact, and is silent once the manifest is stripped or the image
is re-encoded without C2PA (e.g., a screenshot, a social-media re-upload, or
after `metadata --remove`).

This is why:
- `identify` on a GitHub-recompressed issue attachment returns Unknown (C2PA is
  gone) even though the pixel SynthID is still present and detectable by
  openai.com/verify.
- A quiet `identify` output is not proof that SynthID was removed -- it only
  means the metadata signal is gone.

### 3.3 Oracle scope: each vendor detects only their own

From openai.com/research/verify (verbatim, verified 2026-05-31):

> "OpenAI generation signals will only be detected if the image was generated
> with our tools."
> "Content could also still be AI-generated by another company's model, which
> the tool currently does not detect."

SynthID technology is used by multiple vendors, but each verifier is keyed to
its own payload:

| Oracle                        | Detects          | Does NOT detect         |
|-------------------------------|------------------|-------------------------|
| Gemini app "Verify with SynthID" | Google SynthID | OpenAI SynthID          |
| openai.com/research/verify    | OpenAI SynthID   | Google SynthID          |

A Google-SynthID image reads clean on openai.com/verify. An OpenAI image reads
clean in the Gemini oracle. They are different payloads within the same
framework.

---

## 4. Adoption and current state (as of June 2026)

### 4.1 Google products

Google has watermarked **over 10 billion** images and video frames. The
deployment split by surface matters for our tool:

| Surface                              | SynthID pixel | C2PA metadata | Visible sparkle |
|--------------------------------------|---------------|---------------|-----------------|
| Gemini app (generated images)        | YES           | YES (Google)  | YES             |
| Gemini API / AI Studio / Nano Banana | YES           | NO            | YES             |

The Gemini API surface is a key blind spot: it embeds the pixel watermark and
the visible sparkle but **no C2PA or IPTC at all**. Our `identify` returns
Unknown on API-generated images unless the visible sparkle is detected (via
`check_visible=True`) or the user runs the Gemini app oracle.

### 4.2 OpenAI

OpenAI confirmed SynthID adoption (Help Center, updated 2026-05-21):

> "ChatGPT images include both C2PA metadata and SynthID watermarks."

This is time-gated: pre-rollout ChatGPT/gpt-image images carry C2PA without
SynthID. Our C2PA proxy therefore over-reports SynthID presence on old images
(hence the `_OPENAI_CAVEAT` hedging flag in the codebase).

### 4.3 Other vendors

- **Kakao** (South Korea): SynthID adopter as of May 2026 (Google announcement)
- **NVIDIA Cosmos**: SynthID for video (not still images; different pipeline)
- **Meta AI**: does NOT use SynthID; uses IPTC `digitalSourceType` marker instead

### 4.4 Version evolution (v1 vs v2 hardening)

Google has not publicly documented version numbers for the SynthID image
watermark in a way that maps to our testing observations. What is known
empirically from oracle tests:

- **Before May 2026 (Gemini)**: strength 0.05 removed the watermark
- **May 2026 (Gemini)**: strength 0.05 insufficient; 0.10 required
- **Current (Gemini, June 2026)**: on the capped 1536 path, 0.05 and 0.10 do
  NOT clear; 0.15 clears (n=4, Gemini app oracle). See section 2.2.
- **OpenAI (June 2026)**: clears at 0.05 across 1024-1600 (n=4, clean v0.8.6).
  The earlier "0.30 still detected on 1600x1600" report (issue #14) was the
  text-protection bug, not a hardening of the watermark -- see the correction in
  section 2.2.

Google has hardened SynthID relative to OpenAI's (vendor gap measured at ~3x
strength), but the year-over-year "0.05 -> 0.10 -> 0.30" progression above
conflates a real hardening trend with the now-debunked region-rescrub artifact;
treat only the section 2.2 controlled numbers as authoritative.

---

## 5. Practical implications for this tool

### 5.1 Preserving content means regenerating it, never copying it

**Core rule:** SynthID is a pixel-amplitude pattern, so any approach that FREEZES
or RESTORES original pixels in a region re-introduces the watermark there. Early
region-based text/face "protection" (since removed) proved this: restoring the
original face pixels guaranteed SynthID survived in faces, and even a per-region
high-resolution re-scrub from an upscaled crop could be insufficient to destroy
the payload, reconstituting SynthID in text. The lesson held and shaped the
current design: **content is preserved by REGENERATING it under structural
conditioning, never by copying original pixels.**

Both preservation features below are **EXPERIMENTAL and opt-in (off by default)**;
the plain `default` SDXL img2img pass is the shippable path.

- **Text + structure:** `--pipeline controlnet` (SDXL img2img + a canny ControlNet,
  experimental/opt-in) conditions the regeneration on the edge map, so text and
  structure stay sharp
  while every pixel is still regenerated -- SynthID is removed everywhere. Verified
  better than plain img2img at the same strength (text stays legible where plain
  garbles it), and the controlnet background scrub reads clean on the oracle.
- **Face identity:** canny holds face *structure* but not *identity*. Shipped as the
  optional `--restore-faces` GFPGAN post-pass (`face_restore.py`, the `restore`
  extra, experimental/opt-in, off by default). It runs GFPGAN on the ORIGINAL
  faces and feather-composites the restored face REGIONS into the cleaned image:
  GFPGAN RE-SYNTHESIZES each face from a StyleGAN2 prior (GAN pixels, not original
  -> scrubs SynthID) at a low fidelity weight (`--restore-faces-weight`, default
  `0.5`). Oracle-confirmed clean in face regions with identity preserved. Commercial-
  safe (GFPGAN Apache-2.0 + RetinaFace MIT); the CodeFormer alternative is
  NON-COMMERCIAL and is not shipped. (An IP-Adapter FaceID approach was tried and
  REMOVED -- it needs high denoise strength and corrupts faces at removal strength;
  see `docs/controlnet-removal-pipeline-research.md`.)

### 5.2 Strength setting

There is no single permanent correct strength, but the controlled June 2026
study (section 2.2) gives empirical floors:

- **OpenAI**: 0.05 clears across 1024-1600 (n=4). 0.30 is large overkill here.
- **Google (capped 1536)**: 0.15 (n=4); 0.05 and 0.10 do not clear.
- **Google native 2816**: not locally measured; likely needs >= 0.30 (vendor +
  resolution stack). Use a GPU or `--max-resolution 1536`.

The default is **vendor-adaptive** (`watermark_profiles.resolve_strength` +
`vendor_for_strength`): the tool reads the C2PA issuer on the original input and
picks `OPENAI_STRENGTH` 0.10 / `GEMINI_STRENGTH` 0.15 / `UNKNOWN_STRENGTH` 0.15.
This uses the vendor signal we DO have locally (the C2PA SynthID proxy) to avoid
the overkill of a single high default on OpenAI images, without needing a local
pixel detector. An explicit `--strength` always wins. If the watermark still
survives (e.g. a large native Gemini beyond the capped-1536 validation), raise
toward 0.30 then 0.35-0.40 (0.40 visibly corrupts dense text), using the lowest
value that reads clean on the oracle.

### 5.3 Test methodology

- **GitHub-recompressed JPEGs from issue attachments are valid SynthID test
  subjects.** JPEG re-encoding removes C2PA metadata but does NOT remove the
  SynthID pixel watermark (verified June 2026 on issue #14 pic3). Do not
  dismiss these as "not faithful originals" for SynthID-removal tests.
- **The correct oracle for OpenAI images is openai.com/verify**, not the Gemini
  app. The two oracles detect different payloads.
- **A quiet `identify` output after processing is not proof of removal.** It
  means the metadata proxy is gone. The pixel watermark state is unknown without
  an oracle check.
- **After removal, the output may carry forensic artifacts** detectable by an
  independent classifier even if the vendor oracle reads negative. Defeating the
  verifier is not the same as being forensically indistinguishable from clean
  content (arXiv:2605.09203).

### 5.4 Strength vs forensic detectability: the tradeoff

Higher img2img strength removes the watermark but introduces detectable
regeneration artifacts. The Goonatilake & Ateniese paper shows the strongest
diffusion-based removers are simultaneously the most forensically detectable
(AUROC up to 0.9999). The tradeoff is unavoidable with current diffusion-based
approaches: defeating the vendor's verifier is not the same as being clean.

---

## References

1. Gowal et al. (2025). **SynthID-Image: Image watermarking at internet scale.**
   arXiv:2510.09263. https://arxiv.org/abs/2510.09263

2. Google DeepMind. **Identifying AI-generated images with SynthID.** Blog post,
   2023. https://deepmind.google/blog/identifying-ai-generated-images-with-synthid/

3. Google DeepMind. **SynthID.** Product page.
   https://deepmind.google/models/synthid/

4. Goonatilake & Ateniese (2026). **Removing the Watermark Is Not Enough:
   Forensic Stealth in Generative-AI Watermark Removal.** arXiv:2605.09203.
   https://arxiv.org/abs/2605.09203

5. OpenAI. **Verify tool for AI-generated images.** openai.com/research/verify.
   Accessed 2026-05-31.
