# Feature Timing Audit

This document catalogs all features used in the IPL Impact Intelligence model and specifies when each feature becomes available during a match. This audit is critical for preventing temporal leakage and ensuring the model can be deployed in real-world scenarios.

## Feature Availability Timeline

Features are classified into three temporal categories:

1. **Pre-Match** (SAFE): Available before the match starts (historical data, team info)
2. **Post-Toss** (CAUTION): Available after toss but before first ball
3. **In-Play** (UNSAFE for pre-match prediction): Available during or after the match

## Feature Catalog

### Batting Features

| Feature Name | Type | Availability | Notes | Use Case |
|--------------|------|--------------|-------|----------|
| `batting_strike_rate` | Derived | Pre-Match | Historical career/season average | Pre-match prediction ✓ |
| `batting_average` | Derived | Pre-Match | Historical career/season average | Pre-match prediction ✓ |
| `boundary_percentage` | Derived | Pre-Match | Historical metric | Pre-match prediction ✓ |
| `dot_percentage` | Derived | Pre-Match | Historical metric | Pre-match prediction ✓ |
| `runs_scored` | Raw | In-Play | Match-specific performance | Post-match analysis only ✗ |
| `balls_faced` | Raw | In-Play | Match-specific count | Post-match analysis only ✗ |
| `fours` | Raw | In-Play | Match-specific count | Post-match analysis only ✗ |
| `sixes` | Raw | In-Play | Match-specific count | Post-match analysis only ✗ |
| `dismissals` | Raw | In-Play | Match-specific event | Post-match analysis only ✗ |
| `batting_index` | Composite | Pre-Match | Derived from historical metrics | Pre-match prediction ✓ |

### Bowling Features

| Feature Name | Type | Availability | Notes | Use Case |
|--------------|------|--------------|-------|----------|
| `bowling_economy` | Derived | Pre-Match | Historical career/season average | Pre-match prediction ✓ |
| `bowling_strike_rate` | Derived | Pre-Match | Historical metric | Pre-match prediction ✓ |
| `wickets_per_match` | Derived | Pre-Match | Historical average | Pre-match prediction ✓ |
| `phase_efficacy` | Derived | Pre-Match | Historical phase performance vs league | Pre-match prediction ✓ |
| `runs_conceded` | Raw | In-Play | Match-specific performance | Post-match analysis only ✗ |
| `balls_bowled` | Raw | In-Play | Match-specific count | Post-match analysis only ✗ |
| `wickets` | Raw | In-Play | Match-specific count | Post-match analysis only ✗ |
| `overs_bowled` | Derived | In-Play | Calculated during match | Post-match analysis only ✗ |
| `bowling_index` | Composite | Pre-Match | Derived from historical metrics | Pre-match prediction ✓ |

### Composite Features

| Feature Name | Type | Availability | Notes | Use Case |
|--------------|------|--------------|-------|----------|
| `overall_index` | Composite | Pre-Match | Sum of batting_index + bowling_index | Pre-match prediction ✓ |
| `impact_rating` | Derived | Pre-Match | SHAP-weighted historical performance | Pre-match prediction ✓ |

### Match Context (Post-Toss)

| Feature Name | Type | Availability | Notes | Use Case |
|--------------|------|--------------|-------|----------|
| `venue` | Categorical | Pre-Match/Post-Toss | Known after schedule | Pre-match prediction ✓ |
| `city` | Categorical | Pre-Match/Post-Toss | Known after schedule | Pre-match prediction ✓ |
| `batting_team` | Categorical | Post-Toss | Known after toss | Post-toss prediction ✓ |
| `bowling_team` | Categorical | Post-Toss | Known after toss | Post-toss prediction ✓ |
| `date` | Temporal | Pre-Match | Known from schedule | Pre-match prediction ✓ |
| `season` | Temporal | Pre-Match | Known from schedule | Pre-match prediction ✓ |

## Leakage Prevention Guidelines

### For Pre-Match Prediction Use Case

**ALLOWED FEATURES:**
- All historical averages and aggregated metrics (batting/bowling indices, strike rates, averages, economy)
- Venue and team information
- Season and temporal features
- Player clustering assignments (based on historical data)

**PROHIBITED FEATURES:**
- Match-specific raw counts (runs_scored, balls_faced, wickets in THIS match)
- In-play phase performance for THIS match
- Outcomes from THIS match (team_won for prediction)

### Implementation Guards

The `feature_engineering.py` module implements the following guards:

1. **Feature Calculation Separation**: Historical metrics are calculated using data from PREVIOUS matches only
2. **Temporal Filtering**: The training pipeline ensures no future information leaks into historical aggregates
3. **Match-Level Isolation**: Raw match statistics are only used for post-match analysis and impact rating calculation

### Code Integration

In `feature_engineering.py`, features are computed at the player-match level using historical data:

```python
# SAFE: Historical batting metrics aggregated from past matches
batting_metrics = compute_batting_metrics(player_stats)

# SAFE: Historical bowling metrics with phase efficacy
bowling_metrics = compute_bowling_metrics(player_stats, phase_stats)

# SAFE: Composite indices from historical metrics
features, metric_columns = compute_composite_indices(batting_metrics, bowling_metrics)
```

The model training in `model_training.py` uses time-aware splits:

```python
# Time-aware splits prevent leakage
from splits.time_splits import create_time_splits

split = create_time_splits(model_df, config=SplitConfig(
    train_seasons_max=2018,
    val_seasons=[2019],
    test_seasons=[2020]
))
```

## Validation Checklist

- [x] All features documented with availability timing
- [x] Pre-match vs in-play distinction clear
- [x] Leakage prevention guidelines established
- [x] Time-aware splits implemented
- [ ] Automated tests for feature timing (recommended future work)
- [ ] Feature timing validation in CI/CD pipeline (recommended future work)

## Updates and Maintenance

This document should be updated whenever:
1. New features are added to the model
2. Feature calculation logic changes
3. New use cases with different timing requirements are introduced
4. Deployment scenarios change (e.g., real-time prediction vs batch analysis)

**Last Updated**: 2025-11-04
**Version**: 1.0
**Maintainer**: IPL Impact Intelligence Team
