# Token X-ray design system

The story is a radiology reading room. The page is the lightbox; every exhibit is a film of one training
step; anomalies are circled in red grease pencil, the way radiologists mark films. Spend boldness only on
the films and the pencil. Everything around them stays quiet.

## Tokens

| Name          | Hex       | Use                                                    |
| ------------- | --------- | ------------------------------------------------------ |
| Lightbox      | `#E8EDF0` | page background                                        |
| Diffuser      | `#F6F9FA` | the frame around a film, colophon                      |
| Ink           | `#13202A` | text                                                   |
| Muted         | `#5A6873` | captions, secondary text                               |
| Rule          | `#C7D1D7` | chapter dividers, frame outline                        |
| Film          | `#0D1922` | film base, code blocks                                 |
| Bone          | `#EEF3F1` | brightest exposure                                     |
| Grease pencil | `#D7263D` | annotations, focus ring, link underline. Nothing else. |

## Type

- Atkinson Hyperlegible Next (variable, self-hosted): headlines 750–800 with tight tracking, body 400.
- Atkinson Hyperlegible Mono: code only. Never for labels.
- Reenie Beanie: grease-pencil notes on films only.
- KaTeX: all mathematics.
- Sentence case everywhere. No all-caps labels, no eyebrows, no middle-dot meta strings.

## Films

- One cell per token (or per document in leak films). Brightness is the measured value; never decoration.
- Weight films show `w·N`; a correct step is an even mid grey. Contrast curve: `0.42 · v^1.5`.
- Film edge print: `Plate n   Real run | Emulation | Computed   what it is`.
- Notes live in the film's margin (wide) or a strip under it (narrow), never on the cells. A loop marks
  rows only when the effect is local; film-wide effects get a note without a loop.
- Hover shows the exact value and the intended value.

## Motion

One orchestrated moment: the hero lightbox strikes on (fluorescent flicker), then the pencil draws.
Everything else answers the reader: scroll steps swap films and redraw the pencil, objective tabs morph
the formula term by term. `prefers-reduced-motion` removes all of it.

## Layout

Left-aligned prose at 64ch. Films up to 1180 px. The accumulation chapter is a scrolly: steps on the left,
sticky film on the right; on phones the film sticks to the top and steps scroll beneath it.
