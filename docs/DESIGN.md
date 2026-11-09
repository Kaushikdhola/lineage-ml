# Token X-ray design system

The story is a radiology reading room. The page is the dim room; every exhibit is a film of one training
step, lit from behind, and the films are the only light sources on the page. Anomalies are measured, not
doodled: bracketed regions of interest with numeric readouts, the way a scanner display marks an image.
Spend boldness only on the films. Everything around them stays quiet.

## Tokens

| Name        | Hex       | Use                                                                                       |
| ----------- | --------- | ----------------------------------------------------------------------------------------- |
| Room        | `#10191F` | page background                                                                           |
| Room raised | `#142028` | colophon                                                                                  |
| Film        | `#05090D` | film base, code blocks                                                                    |
| Ink         | `#E3E9EC` | headings, text                                                                            |
| Ink soft    | `#C3CDD2` | body prose                                                                                |
| Muted       | `#8A99A3` | captions, secondary text                                                                  |
| Rule        | `#24323C` | chapter dividers, hairlines                                                               |
| Bone        | `#F6F9F7` | brightest exposure, formulas, corner print                                                |
| Caliper     | `#F0B544` | regions of interest, window/level readout, focus ring, selected tab, links. Nothing else. |

## Type

- Newsreader (variable, optical size, self-hosted): headlines light (330–380) and large; body 400 with
  old-style figures. The essay voice.
- IBM Plex Sans Condensed 400/500/600: everything the instrument prints (corner print, row labels, ROI
  notes), captions, tabs, the slider. The scanner's voice.
- IBM Plex Mono: code only.
- KaTeX: all mathematics.
- Sentence case everywhere. No all-caps labels, no eyebrows, no middle-dot meta strings.

## Films

- One cell per token (or per document in leak films). Brightness is the measured value; never decoration.
- Exposure ramp: fog `#050A0E` → silver `#687C86` at 0.5 → bone. A blurred copy of the exposure adds
  halation; grain and a vignette make it a film, not a chart. None of these change which cell is brighter.
- Weight films use a window and level (`luminance = (w·N − level) / width + ½`), default level 1, width
  1.2: a correct step is an even mid grey. The reader can drag across the film (horizontal: width,
  vertical: level), use the arrow keys, and double-click or press Home to reset. Other films use a fixed
  scale.
- Corner print: top-left `Plate n`, the run kind (Real run | Emulation | Computed | Reference) and what it
  is; top-right the window and level in caliper; bottom-left the sample (sequences, targets, micro-batches,
  wide films only); bottom-right the value under the pointer (exact and intended), or a one-line hint.
- Regions of interest are four corner brackets around the rows they measure, with a leader to the note in
  the margin (wide) or a numbered strip under the film (narrow). Film-wide effects get a note with no
  bracket. Never on top of the cells.

## Motion

One orchestrated moment: the hero film is exposed by a scanning bar sweeping top to bottom, then the
regions of interest are measured (brackets draw, readouts appear). Everything else answers the reader:
scroll steps swap films and redraw the brackets, objective tabs morph the formula term by term, dragging
changes the window. `prefers-reduced-motion` removes all of it.

## Layout

- Hero: the title (display serif) on the left, the lead, install line and link on the right, aligned to
  the title's baseline; the hero film at full content width below. On phones everything stacks.
- Prose left-aligned at 62ch. Films and rows of films up to 1240 px, straight on the room (no frames or
  cards), caption below in the condensed face.
- The accumulation chapter is a scrolly: steps on the left with a hairline that turns caliper when active,
  sticky film on the right; on phones the film sticks to the top and steps scroll beneath it.
- Must work at 390 px and honour `prefers-reduced-motion`.
