# Decisions — nfl-fourth-down-ev-lab

Human-owned product and modeling judgment. AI accelerated scaffolding and iteration; metrics and cut calls are owned here.

## 1. Metric definition (EV convention)

**Unit:** expected points (EP) from the **offense’s perspective** at the current snap, using a coarse **EP-by-yardline** curve averaged from nflverse pre-play `ep` on regular-season plays. That curve is a **published reference scale**, not a feature of the conversion/FG models.

| Action | Formula (frozen) |
| --- | --- |
| **Go** | `p_conv * EP(yardline_100 − ydstogo)` + `(1 − p_conv) * (−EP_opp at 100 − yardline_100)` |
| **FG** | **Points-added proxy:** `p_make * (+3)` + `(1 − p_make) * (−EP_opp at miss spot)`. Miss spot heuristic: opponent `yardline_100 ≈ 100 − (yardline_100 + 8)`. |
| **Punt** | Empirical mean offense-perspective EP after punt by yardline bucket: net yards → opponent field position → `−EP_opp`. |

**Recommendation** = argmax EV. **ΔEV** = best − second best.

**Not claimed:** win-probability optimality, betting edge, or causal “coaches are wrong.”

## 2. Rejected leaky features

| Column / family | Why rejected |
| --- | --- |
| `epa`, `ep` as **model inputs** | Post-snap / circular with valuation scale; `ep` used only for EP tables |
| `wpa`, `wp`, `vegas_wp` | Outcome-adjacent / market; WP cut from MVP |
| `success`, `yards_gained`, `air_yards`, `yards_after_catch`, `complete_pass` | Realized after the play |
| `interception`, `fumble_*` | Outcomes |
| `passer_player_id`, `receiver_player_id`, `rusher_player_id` | Player IDs → team/QB stretch cut |
| `desc`, `pass_length`, `run_location` | Post-decision / play design after snap |
| `first_down`, `touchdown`, `field_goal_result` | **Labels**, not features |
| `kick_distance` as live feature | Known after kick; used only to *fit* distance→make curve |
| `return_yards`, `penalty` | Post-play |
| `cp`, `cpoe`, `xyac_*` | Model-on-model / post-throw |
| nflfastR `fg_prob`, `td_prob`, `safety_prob` as features | Circular if we claim originality |

Enforced in `src/nfl_ev/features.py` + `tests/test_features_banned.py`.

## 3. Bucket / min-N choices

- Conversion cells: `yardline_100` in 10-yard bins × `ydstogo` ∈ {1, 2, 3–5, 6+}.
- **Min cell N = 30.** Below that, back off to empirical rate by ydstogo bucket, then global rate; Streamlit flags low confidence.
- FG: kick-distance bins `[0,30,35,40,45,50,55,60,70,90)` with same min-N backoff.
- Train seasons: all available except holdout = last season in the downloaded set (prefer 2021–2024 train / 2025 test).

## 4. FG distance heuristic

At decision time: **`kick_distance ≈ yardline_100 + 17`** (LOS to uprights rule of thumb: 7 end zone + ~10 snap/holder). Historical `kick_distance` is used only when fitting the empirical make curve, never as a coach slider input.

**Long-kick guardrail:** attempts with proxy distance ≥65 / ≥70 yards are hard-capped to ~8% / ~1% make probability. Sparse 60+ yard empirical cells back off toward near-zero — **not** the league-wide make rate (~86%), which is dominated by short kicks and would wrongly recommend FG from own territory.

**FG logistic calibration (Review fix):** do **not** train FG (or convert) with `class_weight='balanced'`. On ~86% make rate that setting collapsed `p_make` toward ~0.5 and systematically undervalued FG in the EV console. Product also blends toward the empirical distance table when a thick cell’s model p is >15pp low.

## 5. What AI suggested that was rejected

| Suggestion | Rejected because |
| --- | --- |
| Win-probability / `nfl4th`-style WP console | Spec cut list; EP story is enough for hire bar |
| Team- or QB-specific hierarchical models | Sample + polish risk before deadline |
| Monte Carlo full possession trees | Stretch; heuristics freeze |
| Shipping raw nflfastR `fg_prob` as “our” FG model | Circular / not original |
| Multi-page Streamlit product with auth | Polish distraction |
| Paid odds / live scrapers | $0 constraint |

## 6. Known limitations + next measurements

- EP ≠ win value in late-game / large score-differential situations (caveat always shown).
- League-average only; no weather beyond optional `roof`.
- FG make and punt net use field-position heuristics.
- Holdout is one season — sensitive to that year’s kick/go mix.
- **Next:** proper WP layer (still free data), opponent-adjusted conversion, weather, and a labeled “ΔEV vs historical coach choice” audit chart (not causal proof).

## Convert label rule

Go attempt (`play_type` ∈ {pass, run}) converts if `first_down == 1` **or** `touchdown == 1`.
