"""
Preprocessing Module
====================

This module handles the data cleaning, transformation and feature engineering 
for all datasets:
- GTZAN: scaling, encoding, train/test split
- Spotify: feature selection, log transform, scaling
- Lyrics: text cleaning, deduplication, filtering
"""
import re
from pathlib import Path
from typing import Tuple, List, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split



class Preprocessor:
    """
    Class for preprocessing datasets for modeling.

    This class provides methods to clean, transform, and prepare the GTZAN,
    Spotify, and lyrics datasets for machine learning tasks.

    Attributes
    ----------
    scaler : StandardScaler
        Scikit-learn StandardScaler for feature normalization
    label_encoder : LabelEncoder
        Scikit-learn LabelEncoder for encoding categorical labels
    feature_names : list of str or None
        List of feature column names after preprocessing
    """

    def __init__(self) -> None:
        """
        Initialize the Preprocessor.

        Creates instances of StandardScaler and LabelEncoder for use
        in preprocessing methods.
        """
        self.scaler: StandardScaler = StandardScaler()
        self.label_encoder: LabelEncoder = LabelEncoder()
        self.feature_names: Optional[List[str]] = None

    def preprocess_gtzan(
        self,
        gtzan_data: pd.DataFrame,
        test_size: float=0.2, 
        random_state: int=42
        ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[str], LabelEncoder]:
        """
        Preprocess GTZAN dataset for genre classification.

        Performs the following steps:
        1. Excludes non-feature columns ('filename', 'length', 'label')
        2. Imputes missing values with column medians
        3. Encodes genre labels as integers
        4. Splits data into train/test sets with stratification
        5. Scales features using StandardScaler

        Parameters
        ----------
        gtzan_data : pd.DataFrame
            Raw GTZAN data containing audio features and genre labels
        test_size : float, default=0.2
            Proportion of the dataset to include in the test split
        random_state : int, default=42
            Random seed for reproducibility

        Returns
        -------
        X_train : pd.DataFrame
            Scaled training features
        X_test : pd.DataFrame
            Scaled test features
        y_train : np.ndarray
            Encoded training labels
        y_test : np.ndarray
            Encoded test labels
        feature_names : list of str
            List of feature column names
        label_encoder : LabelEncoder
            Fitted LabelEncoder for inverse transformation

        Raises
        ------
        TypeError
            If gtzan_data is not a pandas DataFrame
        ValueError
            If required column 'label' is missing or no feature columns found
        """
        print("Preprocessing GTZAN dataset...")
        # Validation
        if not isinstance(gtzan_data, pd.DataFrame):
            raise TypeError("gtzan_data must be a pandas DataFrame")
        if 'label' not in gtzan_data.columns:
            raise ValueError("Required column 'label' is missing from GTZAN data")

        # Make a copy
        data = gtzan_data.copy()

        # Feature selection (exclude non-feature cols)
        exclude_columns = ['filename', 'length', 'label']
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        if len(feature_columns) == 0:
            raise ValueError("No feature columns found after excluding ['filename', 'length', 'label'].")

        # Handle missing values (median per feature seems to be the best strategy here for now)
        if data[feature_columns].isnull().values.any():
            data[feature_columns] = data[feature_columns].fillna(data[feature_columns].median())

        # Separate features and target
        X = data[feature_columns]
        y = data['label']

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )

        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Back to DataFrame (preserve column names & index)
        X_train = pd.DataFrame(X_train_scaled, columns=feature_columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=feature_columns, index=X_test.index)

        # Keep for later use
        self.feature_names = feature_columns

        print("GTZAN preprocessing complete")
        print(f"  - Training set: {X_train.shape}")
        print(f"  - Test set: {X_test.shape}")
        print(f"  - Number of classes: {len(np.unique(y_train))}")
        print(f"  - Features: {len(feature_columns)}")

        return X_train, X_test, y_train, y_test, feature_columns, self.label_encoder

    def preprocess_spotify(
        self,
        spotify_data: pd.DataFrame,
        target_col: str='streams',
        test_size: float=0.2, 
        random_state: int=42
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """
        Preprocess Spotify dataset for popularity prediction.

        Performs the following steps:
        1. Removes rows with missing target values
        2. Selects audio features (danceability, energy, etc.)
        3. Encodes categorical columns ('key', 'mode') if present
        4. Handles inf/NaN values and imputes with medians
        5. Applies log transformation to 'streams' target
        6. Splits data into train/test sets
        7. Scales features using StandardScaler

        Parameters
        ----------
        spotify_data : pd.DataFrame
            Raw Spotify data containing audio features and streaming metrics
        target_col : str, default='streams'
            Name of the target variable column
        test_size : float, default=0.2
            Proportion of the dataset to include in the test split
        random_state : int, default=42
            Random seed for reproducibility

        Returns
        -------
        X_train : pd.DataFrame
            Scaled training features
        X_test : pd.DataFrame
            Scaled test features
        y_train : pd.Series
            Training target values (log-transformed if target is 'streams')
        y_test : pd.Series
            Test target values (log-transformed if target is 'streams')
        feature_names : list of str
            List of feature column names used

        Raises
        ------
        ValueError
            If target column is not found or no usable feature columns exist
        """
        print("Preprocessing Spotify dataset...")

        # Make a copy
        data = spotify_data.copy()

        # Check target
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        # Remove rows with missing target
        before = len(data)
        data = data.dropna(subset=[target_col])
        dropped = before - len(data)
        if dropped:
            print(f"Note: dropped {dropped} rows with missing '{target_col}'")

        # Select audio features
        audio_features = ['danceability', 'energy', 'speechiness', 'acousticness','instrumentalness', 'liveness', 'valence', 
                         'loudness','tempo', 'duration_ms']
        available_features = [feature for feature in audio_features if feature in data.columns]

        # Add categorical encodings if available
        if 'key' in data.columns:
            data['key_encoded'] = data['key'].astype('category').cat.codes
            available_features.append('key_encoded')

        if 'mode' in data.columns:
            data['mode_encoded'] = data['mode'].astype('category').cat.codes
            available_features.append('mode_encoded')

        if 'time_signature' in data.columns:
            available_features.append('time_signature')

        if not available_features:
            raise ValueError("No usable feature columns found.")

        # Separate features and target
        X = data[available_features]

        # Ensure numeric-only for scaling; handle inf or NaN then impute medians
        X = X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        y = data[target_col]

        # Handle outliers in target (optional: log transform for streams)
        if target_col == 'streams':
            y = np.log1p(y)  # Log transform to handle skewness
            print("Applied log transformation to streams")

        # Split (no stratify here since this is regression-like)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale features
        X_train_idx, X_test_idx = X_train.index, X_test.index
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Back to DataFrames (preserve names and indices)
        X_train = pd.DataFrame(X_train_scaled, columns=available_features, index=X_train_idx)
        X_test = pd.DataFrame(X_test_scaled, columns=available_features, index=X_test_idx)

        print("\nSpotify preprocessing complete")
        print(f"  - Training set: {X_train.shape}")
        print(f"  - Test set: {X_test.shape}")
        print(f"  - Features: {len(available_features)}")
        print(f"  - Target: {target_col}")

        return X_train, X_test, y_train, y_test, available_features

        
    def preprocess_lyrics(
        self,
        lyrics_data: pd.DataFrame,
        sample_size: Optional[int]=None
        ) -> pd.DataFrame:
        """
        Preprocess lyrics dataset for NLP analysis.

        Performs the following steps:
        1. Optionally samples data for faster processing
        2. Removes rows with missing 'text' values
        3. Filters out very short lyrics (< 51 characters)
        4. Cleans text (lowercase, remove numbers/URLs/special chars)
        5. Removes duplicate lyrics

        Parameters
        ----------
        lyrics_data : pd.DataFrame
            Raw lyrics data with 'text' column
        sample_size : int, optional
            If provided, randomly sample this many rows for faster processing

        Returns
        -------
        pd.DataFrame
            Cleaned lyrics data with additional columns:
            - 'text_length': character count of original text
            - 'cleaned_text': preprocessed text for analysis

        Raises
        ------
        ValueError
            If 'text' column is not found in lyrics data
        """
        print("Preprocessing lyrics dataset...")

        # Work on a copy
        data = lyrics_data.copy()

        # Sample if requested
        if sample_size is not None and len(data) > sample_size:
            data = data.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} lyrics")

        # Ensure text column exists
        if 'text' not in data.columns:
            raise ValueError("'text' column not found in lyrics data")

        # Drop rows with missing text
        before = len(data)
        data = data.dropna(subset=['text'])
        dropped_missing = before - len(data)
        if dropped_missing:
            print(f"Dropped {dropped_missing} rows with missing 'text'")

        # Ensure strings
        data['text'] = data['text'].astype(str)

        # Remove very short lyrics (likely noise)
        data['text_length'] = data['text'].str.len()
        before_len_filter = len(data)
        data = data[data['text_length'] > 50].copy()  # keep lyrics with at least 51 chars
        removed_short = before_len_filter - len(data)
        if removed_short:
            print(f"Removed {removed_short} very short lyrics (< 51 chars)")

        # Basic text cleaning
        def clean_text(text: str) -> str:
            # lowercase
            text = text.lower()
            # remove numbers
            text = re.sub(r'\d+', '', text)
            # remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            # keep word chars, spaces, apostrophes, hyphens; replace others with space
            text = re.sub(r'[^\w\s\'\-]', ' ', text)
            # collapse repeated whitespace
            text = ' '.join(text.split())
            return text

        data['cleaned_text'] = data['text'].apply(clean_text)

        # Drop duplicate lyrics after cleaning
        before_dups = len(data)
        data = data.drop_duplicates(subset=['cleaned_text'])
        removed_dups = before_dups - len(data)
        if removed_dups:
            print(f"Removed {removed_dups} duplicate lyrics")

        print("\nLyrics preprocessing complete")
        print(f"  - Total lyrics: {len(data)}")
        print(f"  - Average length: {data['text_length'].mean():.0f} characters")

        return data
    

    
    def create_cross_dataset_features(
        self,
        gtzan_data: pd.DataFrame,
        spotify_data: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Create features that combine information from multiple datasets.

        This method is a placeholder for advanced feature engineering
        that combines insights from GTZAN and Spotify datasets.

        Parameters
        ----------
        gtzan_data : pd.DataFrame
            GTZAN data with audio features
        spotify_data : pd.DataFrame
            Spotify data with audio features

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame containing combined features:
            - 'gtzan_mean_tempo': mean tempo from GTZAN dataset
            - 'spotify_mean_tempo': mean tempo from Spotify dataset

        Notes
        -----
        This is currently a minimal implementation. Future versions may
        include more sophisticated cross-dataset feature engineering.
        """
        print("Creating cross-dataset features...")
        
        # This is a placeholder for advanced feature engineering
        # that combines insights from multiple datasets
        
        combined_features = {}
        
        # Example: Compare tempo distributions
        if 'tempo' in gtzan_data.columns and 'tempo' in spotify_data.columns:
            combined_features['gtzan_mean_tempo'] = gtzan_data['tempo'].mean()
            combined_features['spotify_mean_tempo'] = spotify_data['tempo'].mean()
        df = pd.DataFrame([combined_features])

        print("Cross-dataset features created")
        return df

if __name__ == "__main__":
    # Example usage
    try:
        # preferred when src is a package (use from project root: python -m src.preprocessing)
        from .data_loader import DataLoader
    except ImportError:
        # fallback when running the file directly (python src/preprocessing.py)
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parent))
        from data_loader import DataLoader
    
    # Load data
    loader = DataLoader('/files/project-MOISE/data')
    gtzan, spotify, lyrics = loader.load_all_data()
    
    # Preprocess
    preprocessor = Preprocessor()
    
    print("\nPreprocessing GTZAN...")
    X_train, X_test, y_train, y_test, features, le = preprocessor.preprocess_gtzan(gtzan)
    print(f"Classes: {le.classes_}")
    
    print("\nPreprocessing Spotify...")
    X_train_sp, X_test_sp, y_train_sp, y_test_sp, features_sp = preprocessor.preprocess_spotify(spotify)
    
    print("\nPreprocessing Lyrics...")
    lyrics_clean = preprocessor.preprocess_lyrics(lyrics, sample_size=1000)
    print(f"Sample cleaned text:\n{lyrics_clean['cleaned_text'].iloc[0][:200]}...")
