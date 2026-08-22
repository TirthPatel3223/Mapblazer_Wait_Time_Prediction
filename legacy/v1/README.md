# v1 — superseded, kept for reference

These are the original scripts. They are no longer executed by anything and are retained
only so the two defects documented in [`KNOWN_ISSUES.md`](../../KNOWN_ISSUES.md) can be
read in their original form.

**Do not run these.** They produce numbers that do not hold up:

| File | Defect |
|---|---|
| `data_utils.py` | `PARK_CONSTRAINTS` declares park-local operating hours (`Disneyland: 8–24`) and applies them to **UTC** timestamps. Since the parks are UTC−8/−7, this kept 00:00–15:00 local and discarded the entire evening peak — 53% of all wait-minute mass. Calendar features were derived from UTC too, so Sunday evening was labelled Monday. |
| `evaluate_test_data.py`, `evaluate_high_traffic_test_data.py` | Look up model artefacts with `ride.replace(' ','_')` while training wrote them with `re.sub(r'[^\w\s-]','',ride)`. Any name containing an apostrophe, colon, comma, `!` or `&` missed, hit a bare `except: pass`, and was dropped — 43 of 120 rides, disproportionately the highest-wait ones. |
| `train_xgboost_models.py` | `RandomizedSearchCV(cv=3)` is random K-Fold on a time series: future rows land in the validation folds, so the selected hyperparameters are optimistic. |
| `train_xgboost_global.py` | Computes no test metric at all — the v1 evaluation code was dropped during a refactor. |

The published v1 headline of **3.27 min MAE (Prophet)** is a consequence of the first two
rows above: it was measured on a 77-ride low-wait subsample, with the busiest hours of
every day removed. The current pipeline reports **6.84 min** measured on all 109
attractions against genuinely unseen future data.

Both defects are pinned by regression tests in [`tests/`](../../tests) — `test_naming.py`
reproduces both sanitizers verbatim and asserts they now agree, and `test_timezone.py`
asserts the evening peak survives the operating-hours filter.
