# Business KPIs and Success Metrics

This document defines measurable business objectives for the IPL Impact Intelligence system and establishes performance targets aligned with real-world use cases.

## Primary Use Cases

### 1. Pre-Match Win Probability Estimation
**Business Objective**: Provide accurate and well-calibrated win probability estimates before matches for:
- Broadcast graphics and commentary
- Fantasy cricket decision support
- Betting market analysis
- Team strategy planning

### 2. Player Impact Assessment
**Business Objective**: Quantify individual player contributions to team success for:
- Player auctions and team building
- Performance bonuses and incentives
- Talent scouting and development
- Squad selection decisions

### 3. Upset Detection and Value Identification
**Business Objective**: Identify matches where underdogs have higher-than-expected win probability for:
- Fantasy picks (high risk/high reward)
- Strategic insights for analysts
- Market inefficiency identification

## Key Performance Indicators (KPIs)

### KPI 1: Calibrated Win Probability Accuracy

**Metric**: Mean Absolute Error (MAE) between predicted probabilities and actual outcomes

**Target**: MAE ≤ 0.07 on held-out test data

**Rationale**: 
- A well-calibrated model where predicted probabilities match empirical frequencies is essential for all use cases
- MAE of 0.07 means predictions are typically within 7 percentage points of actual outcomes
- This level of accuracy provides actionable insights while acknowledging inherent uncertainty in sports

**Measurement**:
```python
mae = np.mean(np.abs(y_true - y_pred_proba))
```

**Current Performance**: *To be measured after calibration pipeline*

### KPI 2: Expected Calibration Error (ECE)

**Metric**: Expected Calibration Error across probability bins

**Target**: ECE ≤ 0.05

**Rationale**:
- ECE measures calibration quality by comparing predicted probabilities to actual frequencies in bins
- Lower ECE indicates better probability calibration
- Essential for use cases requiring probabilistic outputs (e.g., broadcast graphics showing "65% win probability")

**Current Performance**: *To be measured after calibration pipeline*

### KPI 3: Top-K Upset Detection Precision

**Metric**: Precision@K for identifying upsets in top K predictions

**Definition**: An "upset" is when a team with <40% predicted win probability actually wins

**Target**: Precision@20 ≥ 0.60

**Rationale**:
- For fantasy cricket and value betting, identifying potential upsets is high-value
- Precision@20 means that among the top 20 predicted upsets, at least 60% should materialize
- This KPI directly supports fantasy picks and strategic insights use cases

**Measurement**:
```python
# Identify predicted upsets
upset_threshold = 0.40
predicted_upsets = (y_pred_proba < upset_threshold)

# Actual upsets
actual_wins = (y_true == 1)

# Precision at K
top_k_indices = np.argsort(y_pred_proba)[:K]
precision_at_k = (predicted_upsets[top_k_indices] & actual_wins[top_k_indices]).sum() / K
```

**Current Performance**: *To be measured after threshold tuning*

### KPI 4: Discrimination (ROC AUC)

**Metric**: Area Under the ROC Curve

**Target**: ROC AUC ≥ 0.70 on test data

**Rationale**:
- ROC AUC measures the model's ability to discriminate between wins and losses
- 0.70 represents substantially better than random performance (0.50)
- Relevant for ranking and selection use cases (e.g., player selection, team building)

**Current Performance**: *To be measured on test split*

## Utility Curves

### Win Probability Utility for Fantasy Picks

Different probability ranges have different utility for fantasy cricket decisions:

| Probability Range | Recommended Action | Expected Value |
|-------------------|-------------------|----------------|
| 0.00 - 0.20 | **High Risk Pick** | High reward if correct |
| 0.20 - 0.40 | **Value Pick** | Good risk/reward ratio |
| 0.40 - 0.60 | **Toss-up** | Avoid or use tiebreaker |
| 0.60 - 0.80 | **Safe Pick** | Expected outcome |
| 0.80 - 1.00 | **Very Safe Pick** | Low variance |

*Note: These thresholds are defined in `configs/policy.yaml` and can be tuned based on user risk preference.*

### Player Impact Threshold for Selection

Player impact ratings guide selection decisions:

| Impact Rating | Interpretation | Selection Priority |
|--------------|----------------|-------------------|
| 90 - 100 | Elite performer | **Must select** |
| 75 - 90 | Strong performer | High priority |
| 60 - 75 | Above average | Consider for balance |
| 40 - 60 | Average | Situational |
| < 40 | Below average | Avoid unless specialized role |

## Business Value Calculation

### Example: Fantasy Cricket Value

For a fantasy league with:
- Entry fee: $10
- Payout: $100 for top 3 finish
- Team size: 11 players

**Value from Improved Selection**:
- Baseline win rate (random): ~10%
- Model-informed win rate: ~15% (estimated)
- Expected value increase: (0.15 - 0.10) × $100 - $10 = **$5 per entry**
- With 1000 users: **$5,000 total value created**

### Example: Player Auction Insight

For team building with $10M budget:
- Overvalued player (by market): Impact rating 65, market price $2M
- Model-recommended alternative: Impact rating 75, market price $1.5M
- **Value capture**: $500K savings + better on-field performance

## Monitoring and Alerts

### Real-time KPI Monitoring

KPIs should be monitored on a rolling basis:

1. **Weekly Performance Review**: Check MAE and ECE on recent matches
2. **Monthly Calibration Check**: Re-calibrate if drift detected (ECE > 0.07)
3. **Seasonal Model Refresh**: Retrain model with new season data

### Alert Thresholds

Trigger model review if:
- MAE > 0.10 for 3 consecutive weeks
- ECE > 0.08
- ROC AUC drops below 0.65
- Precision@20 for upsets < 0.50

## Continuous Improvement

### Quarterly Goals

**Q1 2025**: Establish baseline performance on all KPIs
**Q2 2025**: Implement real-time calibration monitoring
**Q3 2025**: Enhance upset detection (target Precision@20 = 0.65)
**Q4 2025**: Integrate player form trends for improved MAE (target: 0.065)

## Stakeholder Communication

### Executive Summary Format

For each model version, provide:
1. **Calibration Quality**: "Win probabilities accurate within ±7%"
2. **Discrimination Power**: "70% better than random at predicting outcomes"
3. **Upset Detection**: "Successfully identifies 60% of top-20 predicted upsets"
4. **Business Impact**: "$5 expected value per fantasy entry"

## Data Requirements for KPI Tracking

To compute these KPIs, ensure:
- Historical match outcomes with player-level data (min 3 seasons)
- Validation data from most recent season for calibration
- Test data from held-out season for unbiased evaluation
- Regular updates as new match data becomes available

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-04  
**Next Review**: End of 2025 IPL Season  
**Owner**: Analytics Team
