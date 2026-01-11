# MOISE - Mood and Noise

**Music Analysis Pipeline using Machine Learning and NLP**

MOISE (Mood and Noise) is a comprehensive data science project that analyzes music through multiple lenses: genre classification, popularity prediction, lyrics analysis, and mood clustering based on psychological theory.

---

## Quick Start

```bash
git clone https://github.com/Xersher/project-MOISE.git
cd project-MOISE
pip install -r requirements.txt
python main.py
```

---

## Features

- **Genre Classification**: Supervised learning models (Random Forest, SVM, Gradient Boosting, MLP) trained on the GTZAN dataset
- **Popularity Prediction**: Regression models to predict song popularity based on Spotify audio features
- **Lyrics Analysis**: NLP pipeline including sentiment analysis, theme detection, and personal pronoun analysis
- **Mood Clustering**: Unsupervised clustering using the Circumplex Model of Affect (Russell, 1980) with K=4 quadrants
- **Exploratory Data Analysis**: Comprehensive visualizations for all datasets

---

## Key Results

| Analysis | Key Finding |
|----------|-------------|
| **Genre Classification** | SVM achieved **91.34% test accuracy** (90.58% CV), outperforming Random Forest (86.39%), Gradient Boosting (86.94%), and MLP (84.98%) |
| **Popularity Prediction** | Random Forest R² = **4.33%** — confirms that audio features alone cannot predict popularity; external factors (marketing, artist fame, playlists) dominate |
| **Lyrics Analysis** | **51.2% positive** sentiment, 39.1% neutral, 9.6% negative; dominant themes: Life (75.7%), Love (66.6%); avg polarity: 0.111, avg subjectivity: 0.505 |
| **Mood Clustering** | **82.7% of songs are high-energy**; Q4 (Calm/Peaceful) nearly absent (1.4%), revealing strong bias toward energetic content in popular music |

### Mood Cluster Distribution (Circumplex Model)

| Quadrant | Name | Songs | Percentage |
|----------|------|-------|------------|
| Q1 | High Energy Positive (Happy, Excited) | 2,813 | 43.2% |
| Q2 | High Energy Negative (Angry, Tense) | 2,573 | 39.5% |
| Q3 | Low Energy Negative (Sad, Melancholic) | 1,035 | 15.9% |
| Q4 | Low Energy Positive (Calm, Peaceful) | 92 | 1.4% |

---

## Project Structure

```
project-MOISE/
├── .venv/                         # Virtual environment (created during setup)
├── data/                          # Input datasets (CSV files)
│   ├── features_3_sec.csv         # GTZAN audio features
│   ├── spotify_top_songs_audio_features.csv
│   └── spotify_millsongdata.csv   # Million Song Dataset lyrics
│
├── results/                       # Output directory
│   ├── figures/                   # Generated visualizations (PNG)
│   ├── models/                    # Trained models (PKL)
│   └── report/                    # Generated reports
│
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── cli.py                     # Command-line interface
│   ├── data_loader.py             # Data ingestion and validation
│   ├── eda.py                     # Exploratory data analysis
│   ├── genre_classifier.py        # Genre classification models
│   ├── lyrics_analyzer.py         # NLP analysis
│   ├── mood_clustering.py         # Circumplex-based mood clustering
│   ├── popularity_predictor.py    # Popularity prediction models
│   └── preprocessing.py           # Data cleaning and feature engineering
│
├── main.py                        # Main entry point
├── PROPOSAL.md                    # Project proposal document
├── AI_USAGE.md                    # AI tools usage documentation
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd project-MOISE
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Linux/macOS
   source venv/bin/activate
   
   # On Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (required for lyrics analysis)
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('averaged_perceptron_tagger')
   ```

---

## Dependencies

The project dependencies are listed in `requirements.txt`:

```
# Core Data Science Libraries
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
wordcloud>=1.8.0

# Machine Learning
scikit-learn>=1.0.0

# Natural Language Processing
nltk>=3.6.0
textblob>=0.17.0

# Model Persistence
joblib>=1.1.0

# Progress Bars and Utilities
tqdm>=4.62.0

# Data Validation
pyarrow>=6.0.0

# Testing
pytest>=6.2.0

# Code Quality
black>=21.0
flake8>=4.0.0

# Documentation
sphinx>=4.3.0

# For reproducibility
python-dotenv>=0.19.0
```

---

## Usage

### Option 1: Run Complete Pipeline

Run all analysis modules with a single command:

```bash
python main.py
```

With custom directories:

```bash
python main.py --data-dir ./my_data --output-dir ./my_results
```

### Option 2: Command-Line Interface (CLI)

The CLI provides more granular control over which modules to run:

```bash
# Show help
python src/cli.py --help

# Run complete pipeline
python src/cli.py --all

# Run specific modules
python src/cli.py --genre          # Genre classification only
python src/cli.py --clustering     # Mood clustering only
python src/cli.py --popularity     # Popularity prediction only
python src/cli.py --lyrics         # Lyrics analysis only

# Run multiple modules
python src/cli.py --genre --clustering --popularity

# With custom directories
python src/cli.py --all --data-dir ./data --output-dir ./results
```

### Option 3: Run Individual Modules

You can also run each module independently:

```bash
# From the src directory
cd src

# Run individual modules
python data_loader.py
python preprocessing.py
python genre_classifier.py
python popularity_predictor.py
python lyrics_analyzer.py
python mood_clustering.py
python eda.py
```

### Option 4: Use as a Library

```python
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.genre_classifier import GenreClassifier

# Load data
loader = DataLoader('./data')
gtzan_data = loader.load_gtzan_data()

# Preprocess
preprocessor = Preprocessor()
X_train, X_test, y_train, y_test, features, label_encoder = \
    preprocessor.preprocess_gtzan(gtzan_data)

# Train classifier
classifier = GenreClassifier(model_directory='./results/models')
classifier.create_models()
results = classifier.train_and_evaluate(X_train, y_train, X_test, y_test, label_encoder)

# Save model
classifier.save_model()
```

---

## Datasets

The pipeline expects three CSV files in the `data/` directory:

| Dataset | Filename | Description |
|---------|----------|-------------|
| GTZAN | `features_3_sec.csv` | Audio features for genre classification (10 genres) |
| Spotify | `spotify_top_songs_audio_features.csv` | Audio features and streaming metrics |
| Lyrics | `spotify_millsongdata.csv` | Song lyrics from Million Song Dataset |

### Data Download Instructions

If the data files are not included in the repository, download them from the following sources:

1. **GTZAN Dataset**: 
   - Source: [Kaggle - GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
   - Download only the `features_3_sec.csv` file and place it in `data/`
   - Note: This project uses only the pre-extracted audio features, not the raw audio files

2. **Spotify Million Song Dataset (Lyrics)**:
   - Source: [Kaggle - Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset)
   - Download and rename to `spotify_millsongdata.csv`, place in `data/`

3. **Spotify Top Songs Audio Features**:
   - Source: [Kaggle - Spotify Top Songs and Audio Features](https://www.kaggle.com/datasets/julianoorlandi/spotify-top-songs-and-audio-features)
   - Download and rename to `spotify_top_songs_audio_features.csv`, place in `data/`

After downloading, your `data/` folder should contain:
```
data/
├── features_3_sec.csv
├── spotify_top_songs_audio_features.csv
└── spotify_millsongdata.csv
```

### Expected Columns

**GTZAN Dataset:**
- `filename`: Audio file identifier
- `label`: Genre label (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
- Audio features: `chroma_stft`, `spectral_centroid`, `tempo`, `mfcc1-20`, etc.

**Spotify Dataset:**
- `streams`: Number of streams (target variable)
- Audio features: `danceability`, `energy`, `valence`, `tempo`, `loudness`, `acousticness`, `instrumentalness`, etc.

**Lyrics Dataset:**
- `artist`: Artist name
- `song`: Song title
- `text`: Lyrics text

---

## Output

### Generated Figures

All visualizations are saved to `results/figures/`:

| # | Filename | Description |
|---|----------|-------------|
| 01 | `gtzan_genre_distribution.png` | Genre distribution in GTZAN |
| 02 | `spotify_audio_features.png` | Audio feature histograms |
| 03 | `lyrics_top_artists.png` | Top 15 artists by song count |
| 04 | `model_comparison.png` | Genre classifier comparison |
| 05 | `confusion_matrix.png` | Best model confusion matrix |
| 06 | `sentiment_distribution.png` | Lyrics sentiment analysis |
| 07 | `word_frequencies.png` | Most common words in lyrics |
| 08 | `song_themes.png` | Theme distribution |
| 09 | `personal_level_distribution.png` | Personal pronoun analysis |
| 10 | `popularity_model_comparison.png` | Regression model comparison |
| 11 | `popularity_predictions.png` | Actual vs predicted plots |
| 12 | `feature_importance.png` | Feature importance for popularity |
| 13 | `mood_clusters_2d.png` | 2D PCA cluster visualization |
| 14 | `cluster_profiles.png` | Cluster feature profiles |
| 15 | `circumplex_model.png` | Circumplex Model visualization |

### Saved Models

Trained models are saved to `results/models/` using joblib serialization:

- `best_genre_classifier_*.pkl` - Best genre classification model with metadata
- `best_popularity_predictor_*.pkl` - Best regression model with metadata

---

## Mood Clustering: Circumplex Model

The mood clustering module is based on Russell's (1980) **Circumplex Model of Affect**, which structures emotions along two dimensions:

- **Valence**: Pleasure-Displeasure (positive ↔ negative)
- **Arousal**: Activation-Deactivation (high energy ↔ low energy)

This creates four quadrants (K=4 clusters):

| Quadrant | Name | Emotions | Color |
|----------|------|----------|-------|
| Q1 | High Energy Positive | Happy, Excited, Energetic | Gold |
| Q2 | High Energy Negative | Angry, Tense, Aggressive | Red |
| Q3 | Low Energy Negative | Sad, Depressed, Melancholic | Blue |
| Q4 | Low Energy Positive | Calm, Peaceful, Relaxed | Green |

---

## Software Engineering Practices

This project follows software engineering best practices:

- **Type Annotations**: Full type hints for code clarity and IDE support
- **NumPy-style Docstrings**: Comprehensive documentation with Parameters, Returns, and Raises sections
- **Multicore Parallelization**: `n_jobs=-1` for scikit-learn estimators (RandomForest, KMeans, cross_val_score)
- **Model Persistence**: Joblib serialization with metadata for reproducibility
- **Reproducibility**: `random_state=42` used consistently across all models and data splits
- **Error Handling**: Informative error messages and exception handling
- **Modular Design**: Each module has a single responsibility with consistent interfaces

---

## Example Output

```
======================================================================
  MOISE - Mood and Noise
  Music Analysis Pipeline
======================================================================

STEP 2: GENRE CLASSIFICATION
======================================================================
Training genre classifiers...
  Training RandomForest...
  Training SVM...
  Training GradientBoosting...
  Training MLP...

BEST MODEL: SVM (Combined score: 0.9096)
======================================================================
  Test Accuracy: 91.34%
  CV Accuracy: 90.58%

STEP 3: POPULARITY PREDICTION
======================================================================
Training popularity predictors...

BEST MODEL: Random Forest (R² = 0.0380)
======================================================================
✓ Model saved to results/models/best_popularity_predictor_random_forest.pkl
✓ Popularity prediction completed successfully

STEP 4: LYRICS ANALYSIS
======================================================================
Preprocessing lyrics...
  - Total lyrics: 57300
  - Average length: 1222 characters

Text statistics summary:
  - Average word count: 219.68 words
  - Average lexical diversity: 0.45
  - Average personal pronoun ratio: 0.12

Sentiment Distribution:
  Positive    29365 (51.2%)
  Neutral     22429 (39.1%)
  Negative     5506 (9.6%)

Average Polarity: 0.111
Average Subjectivity: 0.505

Theme Distribution:
  Life: 43382 songs (75.7%)
  Love: 38168 songs (66.6%)
  Dreams: 25754 songs (44.9%)
  Party: 22900 songs (40.0%)
  Sadness: 20476 songs (35.7%)
  Freedom: 17557 songs (30.6%)
  Happiness: 11387 songs (19.9%)
  Money: 6984 songs (12.2%)

✓ Lyrics analysis completed successfully

STEP 5: MOOD CLUSTERING - Circumplex Model (Russell, 1980)
======================================================================
CIRCUMPLEX MODEL CLUSTER SUMMARY
==================================================
  Q1: High Energy Positive
      → 2815 songs (43.2%)
      → Emotions: Happy, Excited
  Q2: High Energy Negative
      → 2573 songs (39.5%)
      → Emotions: Angry, Tense
  Q3: Low Energy Negative
      → 1036 songs (15.9%)
      → Emotions: Sad, Depressed
  Q4: Low Energy Positive
      → 89 songs (1.4%)
      → Emotions: Calm, Peaceful
==================================================

======================================================================
PIPELINE COMPLETED
======================================================================
All results saved to: results/

Generated figures:
  EDA:
    - 01_gtzan_genre_distribution.png
    - 02_spotify_audio_features.png
    - 03_lyrics_top_artists.png
  Genre Classification:
    - 04_model_comparison.png
    - 05_confusion_matrix.png
  Lyrics Analysis:
    - 06_sentiment_distribution.png
    - 07_word_frequencies.png
    - 08_song_themes.png
    - 09_personal_level_distribution.png
  Popularity Prediction:
    - 10_popularity_model_comparison.png
    - 11_popularity_predictions.png
    - 12_feature_importance.png
  Mood Clustering (Circumplex Model):
    - 13_mood_clusters_2d.png
    - 14_cluster_profiles.png
    - 15_circumplex_model.png

Done!
```

---

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'src'**
   ```bash
   # Run from the project root directory
   cd project-MOISE
   python main.py
   ```

2. **NLTK data not found**
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

3. **FileNotFoundError: Data directory does not exist**
   ```bash
   # Ensure your data files are in the correct location
   ls data/
   # Should show: features_3_sec.csv, spotify_top_songs_audio_features.csv, spotify_millsongdata.csv
   ```

4. **Memory issues with large datasets**
   ```python
   # Use sample_size parameter for lyrics preprocessing
   lyrics_clean = preprocessor.preprocess_lyrics(lyrics_data, sample_size=5000)
   ```

---

## License

This project is for educational purposes as part of a data science course.

---

## Limitations & Future Work

### Limitations

- **Genre Classification**: Uses pre-extracted audio features rather than learning directly from raw audio waveforms or spectrograms
- **Popularity Prediction**: Audio features alone achieve R² < 5%; external factors (marketing, artist popularity, playlist placement) are not captured
- **GTZAN Dataset**: Known issues with mislabeled samples and audio quality (Sturm, 2013)
- **Genre Taxonomy**: Fixed 10-genre classification from 2002 doesn't include modern genres (EDM subgenres, reggaeton, K-pop)
- **Lyrics Analysis**: English-only; sentiment analysis may miss sarcasm, irony, and cultural context

### Future Work

- Implement CNN or Transformer architectures on spectrograms for genre classification
- Add temporal features (release timing, trends) and social media metrics for popularity prediction
- Extend lyrics analysis to multilingual datasets
- Explore multi-label genre classification (songs can belong to multiple genres)
- Incorporate artist metadata and collaborative filtering for popularity prediction

---

## References

- Bergstra, J., Casagrande, N., Erhan, D., Eck, D., & Kégl, B. (2006). Aggregate features and AdaBoost for music classification. *Machine Learning, 65*(2-3), 473–484.

- Costa, Y. M. G., Oliveira, L. S., & Silla Jr, C. N. (2017). An evaluation of convolutional neural networks for music classification using spectrograms. *Applied Soft Computing, 52*, 28–38.

- Downie, J. S. (2003). Music information retrieval. *Annual Review of Information Science and Technology, 37*(1), 295–340.

- Eerola, T., & Vuoskoski, J. K. (2011). A comparison of the discrete and dimensional models of emotion in music. *Psychology of Music, 39*(1), 18–49.

- Hu, X., Downie, J. S., & Ehmann, A. F. (2009). Lyric text mining in music mood classification. In *Proceedings of the 10th International Society for Music Information Retrieval Conference (ISMIR)*.

- Interiano, M., Kazemi, K., Wang, L., Yang, J., Yu, Z., & Komarova, N. L. (2018). Musical trends and predictability of success in contemporary songs in and out of the top charts. *Royal Society Open Science, 5*(5), 171274.

- Kim, Y. E., Schmidt, E. M., Migneco, R., Morton, B. G., Richardson, P., Scott, J., Speck, J. A., & Turnbull, D. (2010). Music emotion recognition: A state of the art review. In *Proceedings of the 11th International Society for Music Information Retrieval Conference (ISMIR)* (pp. 255–266).

- Malheiro, R., Panda, R., Gomes, P., & Paiva, R. P. (2018). Emotionally-relevant features for classification and regression of music lyrics. *IEEE Transactions on Affective Computing, 9*(2), 240–254.

- Pachet, F., & Roy, P. (2008). Hit song science is not yet a science. In *Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)* (pp. 355–360).

- Posner, J., Russell, J. A., & Peterson, B. S. (2005). The circumplex model of affect: An integrative approach to affective neuroscience, cognitive development, and psychopathology. *Development and Psychopathology, 17*(3), 715–734.

- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology, 39*(6), 1161–1178.

- Sturm, B. L. (2013). The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use. *arXiv preprint arXiv:1306.1461*.

- Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. *IEEE Transactions on Speech and Audio Processing, 10*(5), 293–302.

---

## Author

**Paul Morellec**  
paul.morellec@unil.ch  
University of Lausanne

GitHub Repository: [https://github.com/Xersher/project-MOISE](https://github.com/Xersher/project-MOISE)

---

## Acknowledgments

This project was developed as part of the **Advanced Programming 2025** course at the University of Lausanne.

**Tools used during development:**
- Claude (Anthropic): Model proposal, implementation assistance, bug fixes, optimization suggestions
- ChatGPT: Optimization suggestions and writing support
- GitHub Copilot: Inline code autocompletion

All generated code was carefully reviewed, understood, and modified to fit the project's needs. Design decisions were made independently.

---

## License

This project is for educational purposes as part of a data science course.
