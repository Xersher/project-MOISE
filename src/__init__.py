"""
Music Analysis Project - Source Code Package

This package contains modules for:
- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Genre classification
- Popularity prediction
- Lyrics analysis (NLP)
"""

__version__ = "1.0.0"
__author__ = "Music Analysis Team"

# Import main classes and functions for easy access
from .data_loader import DataLoader
from .preprocessing import Preprocessor
from .genre_classifier import GenreClassifier
from .popularity_predictor import PopularityPredictor
from .lyrics_analyzer import LyricsAnalyzer
from .mood_clustering import MoodClusterer

__all__ = [
    'DataLoader',
    'Preprocessor',
    'GenreClassifier',
    'PopularityPredictor',
    'LyricsAnalyzer'
]
