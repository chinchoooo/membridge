# After Effects: where it adds something, and what I prepared

**Honest scope.** I cannot open or operate After Effects. Everything in the assembled video (cuts, captions, transitions, blur, zooms, title and outro cards, speed-ramps) is produced with ffmpeg and is complete without it.
After Effects is optional polish. This is where it is genuinely worth the time, and the layered assets I generate for it.

## Assets (generated into `submission/ae/`, 1920x1080, transparent PNG unless noted)
| File | Use |
|---|---|
| `arch_base.svg`, `arch_layers/*.png` | the architecture diagram as separate layers (boxes, arrows, labels, FalkorDB badge) for animating |
| `hop_nodes.svg`, `hop_layers/*.png` | the "no shared words" path: message → conversation → project → decision, each node and edge separate |
| `lower_third_demo.png`, `lower_third_live.png` | "Live demo" / "My real ChatGPT account" lower thirds |
| `callout_idontknow.png` | the amber "Nothing stored, it will not guess" card for the key moment |
| `badge_falkordb.png`, `badge_private.png` | small badges |
| `membridge_demo.srt` | captions (import as text via a caption plugin, or leave burned in) |

Composition settings: **1920 x 1080, 30 fps, square pixels**, colour space sRGB. Background `#0b1020`, text `#e8ecf8`, accent teal `#35d0ba`, amber `#ffb454`.

## Four places it is worth it (about 20 minutes total)
1. **The "I don't know" moment (scene 2 and scene 3).** Slide `callout_idontknow.png` in with a 6-frame overshoot (Position + Scale, ease out), hold 2 s, fade. This is the point of the whole project; give it a beat.
2. **The graph hop (scene 4).** Import `hop_layers/*.png`. Reveal node 1, then draw the edge (Trim Paths 0 → 100 % on the SVG shape layer) to node 2, 3, 4 over about 5 s, while the voice says "hops from the message, to its conversation, to the project, to the decision". Pulse the final node.
3. **Architecture build (scene 4).** Reveal the layers left to right: Exports → SQLite → FalkorDB → Retrieval → Assistants. Highlight the FalkorDB box with a soft glow while "FalkorDB" is said.
4. **The 12 / 0 counter (scene 4).** Count the text-only number staying at 0 and the graph number rolling up to 12 (Slider Control + expression `Math.round(slider)` on a text layer).

## Workflow with my cut
1. I give you `submission/clips/scene_1.mp4 … scene_6.mp4` (silent visuals) and `submission/final/membridge_demo.mp4`.
2. Drop the scene clips in a 1920x1080 / 30 fps comp in order, lay the voice-over (`voiceover.wav`) underneath, add the four elements above.
3. Render with Media Encoder: H.264, VBR 2 pass, target 12 Mbps, AAC 192 kbps, 1920x1080.
4. If you do this, tell me: I will not re-cut over your render.
