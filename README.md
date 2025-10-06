# IPL Impact Intelligence

This project is a data-driven analysis of the Indian Premier League (IPL), designed to uncover player archetypes, measure their on-field impact, and select a "Best XI" team based on a robust, data-informed methodology. It leverages machine learning and advanced statistical analysis to go beyond traditional cricket metrics and provide a deeper understanding of player performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Project Overview

The goal of this project is to analyze ball-by-ball IPL data to identify different types of players and quantify their impact on match outcomes. The key features of the project include:

- **Player Archetyping**: Using K-Means clustering, players are grouped into distinct roles (e.g., "Power Hitter," "Death Specialist") based on their performance metrics.
- **Impact Rating**: A machine learning model (XGBoost) is trained to predict match outcomes based on player performance. The model's feature importances, derived from SHAP analysis, are used to create a weighted "impact rating" for each player.
- **Best XI Selection**: A data-driven algorithm selects a balanced and high-performing team of 11 players, considering both their impact ratings and their assigned roles.
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
-   `requirements.txt`: Lists all the Python packages required to run the project.
-   `IPL.csv`: The raw dataset file (managed by Git LFS).
-   `GIT_LFS_GUIDE.md`: A guide on how to set up and use Git LFS for this project.
-   `outputs/`: The directory where all generated plots are saved.
-   `NON_PROGRAMMER_GUIDE.md`: A guide to the project for a non-technical audience.