# IPL Impact Intelligence

This project is a data-driven analysis of the Indian Premier League (IPL), designed to uncover player archetypes, measure their on-field impact, and select a "Best XI" team based on a robust, data-informed methodology. It leverages machine learning and advanced statistical analysis to go beyond traditional cricket metrics and provide a deeper understanding of player performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Streamlit Dashboard](#running-the-streamlit-dashboard)
  - [Command-Line Pipeline](#command-line-pipeline)
- [Project Structure](#project-structure)

## Project Overview

The goal of this project is to analyze ball-by-ball IPL data to identify different types of players and quantify their impact on match outcomes. The key features of the project include:

- **Player Archetyping**: Using K-Means clustering, players are grouped into distinct roles (e.g., "Power Hitter," "Death Specialist") based on their performance metrics.
- **Impact Rating**: A machine learning model (XGBoost) is trained to predict match outcomes based on player performance. The model's feature importances, derived from SHAP analysis, are used to create a weighted "impact rating" for each player.
- **Best XI Selection**: A data-driven algorithm selects a balanced and high-performing team of 11 players, considering both their impact ratings and their assigned roles.
- **Interactive Dashboard**: A Streamlit app consolidates the analysis into an interactive workspace with filters, charts, and tables tailored for analysts and coaches.
- **High-Quality Visualizations**: The project generates insightful and aesthetically pleasing plots for feature importance and player clusters.

## Methodology

The project follows a multi-step data pipeline to process the raw data and generate the final insights. Here is a step-by-step breakdown of the methodology:

1.  **Data Preprocessing**: The raw ball-by-ball IPL dataset is loaded and cleaned. Column names are standardized to ensure consistency, and new features are derived, such as identifying dot balls, boundaries, and wickets. Each delivery is also assigned to a specific phase of the game (Powerplay, Middle, Death).

2.  **Feature Engineering**: The preprocessed data is aggregated to the player-match level. Advanced metrics are then calculated for both batting and bowling, including strike rates, averages, economy rates, and a unique "phase efficacy" score for bowlers, which measures their performance against the league average in different stages of an innings.

3.  **Composite Indices**: The engineered metrics are combined into composite `batting_index` and `bowling_index` scores. The weights for these indices are determined by the correlation of each metric with the match outcome (`team_won`), providing a data-driven way to value different aspects of performance.

4.  **Model Training**: An XGBoost classifier is trained on the engineered features to predict whether a player's team won the match. The model learns the complex relationships between player performance and match outcomes.

5.  **SHAP Analysis**: SHAP (SHapley Additive exPlanations) is used to interpret the trained model. It calculates the importance of each feature in driving the model's predictions. These feature importances are then used as weights to calculate the final player impact ratings.

6.  **Player Clustering**: K-Means clustering is applied to the performance metrics to group batters and bowlers into distinct archetypes. Each cluster is then assigned a human-readable label (e.g., "Power Hitter") by analyzing the characteristics of its center.

7.  **Team Selection**: A "Best XI" team is selected using a greedy algorithm that prioritizes players with the highest impact ratings while ensuring a balanced team composition of batters, bowlers, and all-rounders.

8.  **Rigorous Evaluation**: The model is evaluated with time-aware train/validation/test splits, calibrated probabilities, confidence intervals on metrics, baseline comparisons, and drift analysis to ensure reliable real-world performance.

## Reproducibility & Evaluation

This project follows best practices for ML reproducibility and rigorous evaluation:

### Data Splits (Time-Aware)
- **Training**: Seasons ≤ 2018
- **Validation**: Season 2019
- **Test**: Season 2020

**Why time-aware?** To prevent temporal leakage and simulate real-world deployment where we predict future matches using historical data.

### Random Seed
All random operations use seed `42` by default for reproducibility. Set via:
```bash
python main.py --seed 42
```

### Model Evaluation Metrics
- **Calibration Quality**: Expected Calibration Error (ECE), Brier Score
- **Discrimination**: ROC-AUC, Precision-Recall AUC
- **Accuracy Metrics**: Accuracy, F1 Score, with 95% Confidence Intervals (via bootstrap)
- **Baseline Comparisons**: Performance deltas vs. venue heuristic, recent form, and simple logistic baselines

### Feature Timing Audit
All features are documented with their availability timeline (pre-match, post-toss, in-play) to prevent leakage. See [FEATURE_TIMING_AUDIT.md](docs/FEATURE_TIMING_AUDIT.md).

### Business KPIs
The model targets measurable business objectives:
- Calibrated win probability MAE ≤ 0.07
- Expected Calibration Error (ECE) ≤ 0.05
- ROC AUC ≥ 0.70

See [BUSINESS_KPIS.md](docs/BUSINESS_KPIS.md) for details.

### How to Reproduce Results

**One-command reproduction:**
```bash
make reproduce
```

This runs the complete pipeline:
1. Trains the model with time-aware splits
2. Calibrates probabilities on validation data
3. Evaluates with comprehensive metrics and CIs
4. Generates all reports and visualizations

**Manual step-by-step:**
```bash
# Setup environment
make setup

# Run tests
make test

# Train model
make train

# Evaluate model
make evaluate

# Generate calibration and reports
make calibrate
make report

# Launch dashboard
make app
```

### Generated Artifacts
After running `make reproduce`, find outputs in `reports/`:
- `metrics_with_ci.json` - Metrics with 95% confidence intervals
- `baseline_deltas.csv` - Performance vs. baselines
- `splits.csv` / `split_summary.json` - Train/val/test split details
- `figures/` - ROC, PR, calibration, and drift plots
- `DRIFT.md` - Analysis of performance over seasons
- `FAILURES.md` - Top false positives/negatives analysis

## Installation

To set up the project and run it on your local machine, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/ipl-impact-intelligence.git
    cd ipl-impact-intelligence
    ```

2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies**:
    The project's dependencies are listed in the `requirements.txt` file. You can install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Dataset with Git LFS**:
    This project uses Git LFS to manage the large `IPL.csv` dataset. To download the dataset, please follow the instructions in the [**GIT_LFS_GUIDE.md**](./GIT_LFS_GUIDE.md).

## Usage

### Running the Streamlit Dashboard

Launch the interactive dashboard to explore the project outputs visually:

```bash
streamlit run streamlit_app.py
```

By default the app expects the Kaggle dataset at `IPL.csv` in the repository root. You can supply an alternative path using the sidebar configuration panel, which will trigger the cached pipeline to refresh automatically.

### Command-Line Pipeline

To run the full end-to-end analysis pipeline, execute the `main.py` script from the root directory:
```bash
python main.py
```
This will process the data, train the model, and print the model metrics and the selected "Best XI" team to the console. You can also export the results to a JSON file:
```bash
python main.py --export-json results.json
```
The pipeline will also generate and save all visualizations in the `outputs/` directory.

## Project Structure

The project is organized into several Python modules, each with a specific responsibility:

-   `main.py`: The main script that orchestrates the entire data analysis pipeline.
-   `data_preprocessing.py`: Handles loading, cleaning, and preprocessing the raw data.
-   `feature_engineering.py`: Calculates advanced metrics and composite indices.
-   `model_training.py`: Trains the XGBoost model and evaluates its performance.
-   `shap_analysis.py`: Performs SHAP analysis to interpret the model's predictions.
-   `clustering_roles.py`: Groups players into archetypes using K-Means clustering.
-   `team_selection.py`: Selects the "Best XI" team based on impact and roles.
-   `visualization.py`: Generates and saves high-quality plots for feature importance and clusters.
-   `streamlit_app.py`: Hosts the interactive dashboard that wraps the entire analytics pipeline.
-   `requirements.txt`: Lists all the Python packages required to run the project.
-   `IPL.csv`: The raw dataset file (managed by Git LFS).
-   `GIT_LFS_GUIDE.md`: A guide on how to set up and use Git LFS for this project.
-   `outputs/`: The directory where all generated plots are saved.
-   `NON_PROGRAMMER_GUIDE.md`: A guide to the project for a non-technical audience.
### New Module Structure

**Core Utilities** (`src/common/`):
-   `seeding.py`: Global random seed management for reproducibility
-   `calibration.py`: Probability calibration (Platt scaling, isotonic regression)
-   `metrics.py`: Metrics computation with bootstrap confidence intervals
-   `baselines.py`: Baseline models for comparison (venue heuristic, recent form, simple logistic)
-   `visualization.py`: Evaluation plots (ROC, PR, calibration curves)

**Analysis Modules** (`analysis/`):
-   `error_slices.py`: Slice metrics by season, venue, and matchup
-   `failures.py`: Identify and analyze top prediction errors

**Time-Aware Splits** (`splits/`):
-   `time_splits.py`: Chronological data splitting to prevent temporal leakage

**CLI Tools** (`cli/`):
-   `score_oos.py`: Out-of-sample scoring script
-   `calibrate.py`: Standalone calibration tool

**Configuration** (`configs/`):
-   `policy.yaml`: Probability band mappings for different use cases (fantasy picks, broadcast, strategy)

**Documentation** (`docs/`):
-   `DATA_CARD.md`: Comprehensive dataset documentation
-   `FEATURE_TIMING_AUDIT.md`: Feature availability timeline audit
-   `BUSINESS_KPIS.md`: Business metrics and success criteria

## How to Read the Outputs

### Calibration Plot
The calibration plot (reliability diagram) shows how well predicted probabilities match actual outcomes:
- **Perfect calibration**: Points lie on the diagonal line
- **Overconfident**: Points below the line (predicting higher probabilities than actual frequency)
- **Underconfident**: Points above the line (predicting lower probabilities than actual frequency)

### ROC Curve
The ROC curve shows the trade-off between true positive rate and false positive rate:
- **Area Under Curve (AUC)**: 0.5 = random, 1.0 = perfect
- **Higher AUC** = Better discrimination between wins and losses

### Precision-Recall Curve
Shows the trade-off between precision and recall at different thresholds:
- **Higher curve** = Better performance
- **Average Precision (AP)** summarizes the curve

### Metrics with Confidence Intervals
Example: `Accuracy: 0.7234 [0.7012, 0.7456]`
- **Value**: Point estimate (0.7234)
- **95% CI**: [0.7012, 0.7456] - We're 95% confident the true value lies in this range
- **Narrow intervals** = More stable/reliable estimates

### Baseline Deltas
Shows performance improvement over simple baselines:
- **Positive delta** = Model outperforms baseline
- **95% CI not including 0** = Statistically significant improvement

### Drift Analysis
Tracks model performance over time (by season):
- **Increasing log loss** = Degrading calibration (consider retraining)
- **Changing accuracy** = Evolving match dynamics or team strategies

## Contributing

Contributions are welcome! Please ensure:
1. All tests pass: `make test`
2. Code is formatted: `black --line-length 120 .` and `isort --profile black .`
3. Linter passes: `flake8 --max-line-length 120 .`

Use pre-commit hooks:
```bash
pre-commit install
pre-commit run --all-files
```

## License

This project is provided for educational and research purposes. See dataset license on Kaggle.

## Citation

If you use this project in your research, please cite:
```
IPL Impact Intelligence
https://github.com/saint2706/HRR-MLBA-Project
```

## Acknowledgments

- IPL dataset from Kaggle (chaitu20/ipl-dataset2008-2025)
- Built with scikit-learn, XGBoost, SHAP, Streamlit, and pandas
