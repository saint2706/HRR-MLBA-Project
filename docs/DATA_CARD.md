# IPL Dataset Data Card

This document provides comprehensive information about the IPL ball-by-ball dataset used in the Impact Intelligence project, following best practices for dataset documentation.

## Dataset Identification

**Name**: Indian Premier League (IPL) Ball-by-Ball Dataset (2008-2025)

**Version**: Kaggle Dataset - `chaitu20/ipl-dataset2008-2025`

**Format**: CSV (Comma-Separated Values)

**Size**: ~296 MB (managed via Git LFS)

**Last Updated**: 2025 (covers matches through latest available season)

## Provenance

### Source

- **Origin**: Kaggle public dataset aggregating IPL match data
- **Collection Method**: Aggregated from official IPL scorecards and match reports
- **Temporal Coverage**: IPL seasons from 2008 to 2025
- **Geographic Coverage**: Matches played in India and occasionally international venues

### Licensing and Usage Rights

- **License**: Publicly available on Kaggle (check Kaggle dataset page for specific terms)
- **Attribution**: Dataset compiled by Kaggle user `chaitu20`
- **Commercial Use**: Allowed for educational and analytical purposes
- **Restrictions**: No resale of raw data; derivative works (models, insights) are permitted

### Data Collection Process

1. **Official Sources**: Data derived from official IPL match scorecards
2. **Granularity**: Ball-by-ball records including runs, wickets, extras, and contextual information
3. **Curation**: Data cleaned and standardized by dataset maintainer
4. **Updates**: Dataset refreshed periodically after IPL seasons

## Schema and Structure

### Core Columns

| Column Name | Type | Description | Availability |
|-------------|------|-------------|--------------|
| `match_id` | Integer | Unique identifier for each match | Pre-match |
| `season` | Integer | IPL season year | Pre-match |
| `date` | Date | Match date | Pre-match |
| `venue` | String | Stadium/ground name | Pre-match |
| `city` | String | City where match was played | Pre-match |
| `innings` | Integer | Innings number (1 or 2) | In-play |
| `over` | Integer | Over number (1-20) | In-play |
| `ball_in_over` | Integer | Ball number within the over | In-play |
| `batter` / `striker` | String | Name of batsman facing the ball | In-play |
| `non_striker` | String | Name of non-striker batsman | In-play |
| `bowler` | String | Name of bowler | In-play |
| `batting_team` | String | Team currently batting | Post-toss |
| `bowling_team` | String | Team currently bowling | Post-toss |
| `runs_batter` / `batsman_runs` | Integer | Runs scored by batsman off this ball | In-play |
| `runs_total` / `total_runs` | Integer | Total runs off this ball (including extras) | In-play |
| `runs_extras` / `extras` | Integer | Extra runs (wides, no-balls, etc.) | In-play |
| `extra_type` | String | Type of extra (wide, noball, bye, etc.) | In-play |
| `player_out` / `player_dismissed` | String | Name of dismissed player (if wicket) | In-play |
| `wicket_kind` / `dismissal_kind` | String | Type of dismissal | In-play |
| `is_wicket` | Boolean | Whether a wicket fell on this ball | In-play |
| `win_team` / `winner` | String | Team that won the match | Post-match |

**Note**: Column names vary across dataset versions. The pipeline handles multiple naming conventions via the `COLUMN_ALIASES` mapping in `data_preprocessing.py`.

## Data Statistics

### Sample Size by Season

| Season | Approx. Matches | Approx. Ball Records |
|--------|----------------|---------------------|
| 2008-2010 | ~50-60/season | ~30,000-40,000/season |
| 2011-2015 | ~60-76/season | ~40,000-50,000/season |
| 2016-2020 | ~56-60/season | ~35,000-45,000/season |
| 2021-2025 | ~70-74/season | ~45,000-55,000/season |

**Total**: ~900-1000 matches, ~600,000-700,000 ball records

*Note: Exact counts depend on dataset version and season length variations.*

### Class Balance

For match-level predictions (team win/loss):
- **Balanced by design**: Each match has one winner and one loser
- **Home advantage**: Varies by venue (typically 50-55% win rate)
- **Team imbalance**: Top teams may have 55-65% win rates; weaker teams 35-45%

For player-level analysis:
- **Batting opportunities**: Not all players bat in every match
- **Bowling distribution**: Bowlers bowl 1-4 overs typically
- **Class imbalance**: Star players have more data than fringe players

## Known Biases and Limitations

### Temporal Biases

1. **Rule Changes**: IPL rules have evolved (e.g., strategic timeouts, player substitutions)
2. **Format Evolution**: Early seasons had different tournament structures
3. **Team Dynamics**: Franchise stability varies; some teams ceased to exist or were renamed

### Data Quality Issues

1. **Missing Data**: 
   - Some early seasons have incomplete ball-by-ball records
   - Player names may have variations (e.g., "AB de Villiers" vs "de Villiers")
   
2. **Inconsistent Naming**:
   - Venue names changed over time
   - Team franchises renamed (e.g., "Delhi Daredevils" → "Delhi Capitals")

3. **Edge Cases**:
   - Rain-affected matches (DLS adjustments not explicitly labeled)
   - Super overs not always clearly marked
   - Retired hurt dismissals may be inconsistently recorded

### Representation Biases

1. **Venue Bias**: Certain venues host more matches (e.g., Mumbai, Bangalore)
2. **Team Imbalance**: Top teams (MI, CSK) have more data due to playoff appearances
3. **Player Sampling**: Star players overrepresented; bench players underrepresented

### Leakage Risks

**Critical for ML Applications**:

1. **Temporal Leakage**: Using future data to predict past outcomes
   - **Mitigation**: Time-aware splits (train on past seasons, test on future)
   
2. **In-Match Leakage**: Using ball-by-ball data for pre-match predictions
   - **Mitigation**: Feature timing audit (see `docs/FEATURE_TIMING_AUDIT.md`)
   
3. **Target Leakage**: Features derived from match outcome
   - **Mitigation**: Calculate player metrics from historical matches only

## Data Processing Pipeline

### Preprocessing Steps

1. **Canonicalization**: Standardize column names using alias mapping
2. **Derived Features**: Add game phase (Powerplay, Middle, Death)
3. **Boolean Flags**: Create is_dot_ball, is_boundary, is_wicket
4. **Aggregation**: Roll up to player-match level statistics
5. **Feature Engineering**: Calculate strike rates, averages, economy, phase efficacy

### Quality Checks

- **Completeness**: Drop rows with missing critical fields (match_id, batter, bowler)
- **Validity**: Ensure numeric fields (runs, balls) are non-negative
- **Consistency**: Verify total_runs = batter_runs + extras

## Ethical Considerations

### Consent and Privacy

- **Public Figures**: Players are public figures; performance data is publicly available
- **No Personal Information**: Dataset contains only on-field performance, no personal details
- **Aggregation**: Individual ball records aggregated for analysis

### Fair Use

- **Non-Discriminatory**: Model does not discriminate based on protected attributes
- **Transparency**: Model predictions explainable via SHAP analysis
- **Limitations**: Model should not be sole basis for player selection or contracts

### Potential Harms

1. **Over-reliance**: Model predictions should augment, not replace, expert judgment
2. **Unfair Comparison**: Players from different eras or roles should not be directly compared
3. **Gaming**: Model should not be exploited for betting with inside information

## Maintenance and Updates

### Update Frequency

- **During Season**: Dataset updated weekly or after each match day
- **Post-Season**: Comprehensive update after IPL season concludes
- **Version Control**: Dataset versioned in repository via Git LFS

### Data Quality Monitoring

- **Automated Checks**: Pipeline validates data schema and distributions
- **Manual Review**: Periodic spot-checks for anomalies or errors
- **Community Feedback**: Report issues on Kaggle dataset page or project repository

### Deprecation Policy

- **Backward Compatibility**: New dataset versions maintain core schema
- **Breaking Changes**: Announced with migration guide if column names change
- **Archive**: Old dataset versions retained for reproducibility

## Links and References

- **Kaggle Dataset**: [https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025](https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025)
- **Feature Timing Audit**: See `docs/FEATURE_TIMING_AUDIT.md`
- **Preprocessing Code**: See `data_preprocessing.py`
- **Git LFS Guide**: See `GIT_LFS_GUIDE.md`

## Contact and Feedback

For questions or issues related to this dataset:
- **Project Issues**: Open an issue on the GitHub repository
- **Dataset Issues**: Report on the Kaggle dataset page
- **Data Card Updates**: Submit PR to update this document

---

**Data Card Version**: 1.0  
**Last Updated**: 2025-11-04  
**Next Review**: After 2026 IPL Season  
**Maintainer**: IPL Impact Intelligence Team
