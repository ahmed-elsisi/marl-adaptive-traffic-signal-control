# Defense Talk — Timing & Speaker Script

**Total target: ~20 minutes of speaking** (Q&A separate). 23 slides.
Per-slide talking points also live in each slide's notes pane — press **N** in the
deck to show/hide them while presenting.

## Opening the deck

- Open `index.html` in a browser, press **F** for fullscreen.
- Navigation: **← / →** (or click), **Esc** = thumbnail overview, **N** = notes,
  type a number + wait = jump to that slide.
- Backup: browser **Print → Save as PDF** gives one slide per page.

## Timing budget (≈20 min)

| # | Slide | Section | Target | Running |
|---|-------|---------|-------:|--------:|
| 1 | Title | Open | 0:30 | 0:30 |
| 2 | Overview | Open | 0:20 | 0:50 |
| 3 | Introduction | Intro | 1:30 | 2:20 |
| 4 | Problem Statement | Problem | 1:30 | 3:50 |
| 5 | Research Objectives | Objectives | 1:30 | 5:20 |
| 6 | Theory — RL | Framework | 1:00 | 6:20 |
| 7 | Theory — MARL | Framework | 1:00 | 7:20 |
| 8 | Theory — Social Dilemmas + spectrum | Framework | 1:00 | 8:20 |
| 9 | Methodology — Environment | Methods | 1:00 | 9:20 |
| 10 | Methodology — Intersection & Actions | Methods | 1:00 | 10:20 |
| 11 | Methodology — MAPPO & Why | Methods | 1:30 | 11:50 |
| 12 | Methodology — Observations & Reward | Methods | 1:00 | 12:50 |
| 13 | Methodology — Comparators & Fair Protocol | Methods | 1:00 | 13:50 |
| 14 | Results — Training | Results | 0:50 | 14:40 |
| 15 | Results — Fair baselines (Q1) | Results | 1:30 | 16:10 |
| 16 | Results — Not a strawman (Q4) | Results | 0:45 | 16:55 |
| 17 | Results — The Twist (Q2) | Results | 1:30 | 18:25 |
| 18 | Results — Live demo (GIF) | Results | 0:45 | 19:10 |
| 19 | Discussion — Why ≈ 0 (Q3) | Discussion | (fold) | ~19:1 |
| 20 | Challenges | Wrap | 0:30 | — |
| 21 | Future Work | Wrap | 0:30 | — |
| 22 | Conclusion | Wrap | 0:30 | — |
| 23 | Thank You / Q&A | Close | — | ~20:30 |

> If running long, the compressible slides are 6–7 (theory), 16 (fidelity), and
> 20 (challenges). **Never rush slide 17 — that's the result.** Discussion (19) can
> be folded into the twist if time is tight.

## The narrative spine (say it as a story)

1. **Setup** — MARL is hard (non-stationarity); MAPPO's centralized critic + neighbour
   reward *should* fix it. Traffic = pure-coordination testbed.
2. **Rising action** — MAPPO converges; under a *fair* 3 s-clearance test it beats the
   heuristics decisively. (Methodological fix: the heuristics had been flattered by a
   zero-clearance evaluation.)
3. **Credibility** — our compact MAPPO ≈ the published recipe, so it's no strawman.
4. **The twist** — against an identical IPPO, MAPPO wins *nothing*. Coordination
   value ≈ 0 on 2×2.
5. **Resolution** — not because coordination is useless, but because the network is
   small enough that incentives are already aligned. ⇒ value of coordination is a
   *function of scale*. Future work tests it.

## Objectives → answers (tick these off as you go)

- **Q1** does MARL beat heuristics fairly? → **slide 15** ✓ yes (2×–5×)
- **Q4** is compact MAPPO enough? → **slide 16** ✓ yes (≈ paper, −45% params)
- **Q2** does independent IPPO degrade? → **slide 17** ✓ no (n.s. on every metric)
- **Q3** is coordination value a function of scale? → **slide 19** → hypothesis

## Key numbers (single source of truth — `report/README.md` §4)

- Training: reward −2757 → **−49.76 ± 0.41**; expl-var **0.907**; KL **0.0034**.
- Fair avg wait (3 s clearance, seeds 42–46): MAPPO **9.92 ± 0.65**, IPPO **10.34 ± 0.31**,
  Paper **10.21 ± 0.28**, Max-Pressure **21.63 ± 1.00**, Fixed-Time **47.03 ± 0.59**.
- Confound: Max-Pressure **8.15** (0 s) → **21.63** (3 s). Fixed **38.02 → 47.03**.
- MAPPO vs IPPO p-values: wait **0.236**, halting **0.534**, max halt **0.745**,
  trips **0.545**, phase switches **0.015** (MAPPO 1391 vs IPPO 1370).
- MAPPO vs Paper: wait p = **0.39**.

> ⚠ Only ever cite the **matched-3 s** figures. Never quote the old 0 s numbers
> (8.05 / 38.25) without the clearance caveat.

## Anticipated Q&A

- *"Isn't ≈ 0 just low statistical power?"* — 5 seeds, and we lead with effect sizes,
  not just p: the point estimates are nearly identical and demand is deterministic, so
  the only variation is car-following micro-behaviour.
- *"Is IPPO a fair comparator?"* — Only two variables differ (centralized vs
  decentralized critic; neighbour reward on/off). Same actor, same hyperparameters,
  same parameter sharing.
- *"Why not just run 5×5?"* — Single GPU; ~2 days/run at 2×2; 25 agents was beyond the
  compute budget. Skeleton was prototyped and reverted.
- *"Does the +21 phase switches matter?"* — No measurable performance effect; if
  anything a mild inefficiency, consistent with a critic that's active but not pivotal.

## Before presenting — checklist

- [ ] Drop `assets/mappo_vs_fixed.gif` (MAPPO vs fixed-time SUMO render) — slide 18.
- [ ] Drop `assets/city_5x5.png` (5×5 prototype) — slide 20.
- [ ] Open in the browser you'll present with; press **F**; arrow through all 23.
- [ ] Test on the projector resolution (deck auto-scales to any 16:9-ish window).
