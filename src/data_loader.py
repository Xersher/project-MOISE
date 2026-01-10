"""
Data Loader Module
==================

This module handles loading and initial validation of all datasets:
- GTZAN features dataset
- Spotify top songs dataset
- Million Song Dataset lyrics
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import pandas as pd

class DataLoader:
    """
    Class to load and validate all datasets.

    Attributes
    ----------
    data_directory : Path
        Path to the data directory
    gtzan_data : pd.DataFrame or None
        GTZAN genre classification data
    spotify_data : pd.DataFrame or None
        Spotify top songs data
    lyrics_data : pd.DataFrame or None
        Million Song Dataset lyrics
    """
    
    def __init__(self, data_directory: str ='/files/project-MOISE/data') -> None:
        """
        Initialize the DataLoader.

        Parameters
        ----------
        data_directory : str, default='/files/project-MOISE/data'
            Path to the directory containing the CSV files

        Raises
        ------
        FileNotFoundError
            If the data directory does not exist
        """
        self.data_directory: Path = Path(data_directory)
        self.gtzan_data: Optional[pd.DataFrame] = None
        self.spotify_data: Optional[pd.DataFrame] = None
        self.lyrics_data: Optional[pd.DataFrame] = None

        if not self.data_directory.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_directory}")

    def load_gtzan_data(self, filename: str = 'features_3_sec.csv') -> pd.DataFrame:
        """
        Load GTZAN genre classification dataset.

        Parameters
        ----------
        filename : str, default='features_3_sec.csv'
            Name of the GTZAN CSV file

        Returns
        -------
        pd.DataFrame
            Loaded GTZAN data with audio features and genre labels

        Raises
        ------
        FileNotFoundError
            If the file does not exist
        ValueError
            If expected columns ('filename', 'label') are missing
        RuntimeError
            If any other error occurs during loading
        """
        filepath = self.data_directory / filename
        
        try:
            self.gtzan_data = pd.read_csv(filepath)
            
            #Validation: check for expected columns
            expected_cols = ['filename', 'label']
            for col in expected_cols:
                if col not in self.gtzan_data.columns:
                    raise ValueError(f"Missing expected column: {col}")
            return self.gtzan_data
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {filepath}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading GTZAN data: {e}") from e


    def load_spotify_data(self, filename: str = 'spotify_top_songs_audio_features.csv') -> pd.DataFrame:
        """
        Load Spotify top songs dataset.

        Parameters
        ----------
        filename : str, default='spotify_top_songs_audio_features.csv'
            Name of the Spotify CSV file

        Returns
        -------
        pd.DataFrame
            Loaded Spotify data with audio features and streaming metrics

        Raises
        ------
        FileNotFoundError
            If the file does not exist
        ValueError
            If expected columns ('streams', 'danceability', 'energy', 
            'valence', 'tempo') are missing
        RuntimeError
            If any other error occurs during loading
        """
        filepath = self.data_directory / filename

        try:
            df = pd.read_csv(filepath)

            # Validation (warning if expected columns are missing)
            expected_columns = ['streams', 'danceability', 'energy', 'valence', 'tempo']
            missing = [col for col in expected_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing expected column(s): {', '.join(missing)}")

            self.spotify_data = df
            return df

        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {filepath}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load Spotify data from {filepath}") from e


    def load_lyrics_data(self, filename: str = 'spotify_millsongdata.csv') -> pd.DataFrame:
        """
        Load Million Song Dataset lyrics.

        Parameters
        ----------
        filename : str, default='spotify_millsongdata.csv'
            Name of the lyrics CSV file

        Returns
        -------
        pd.DataFrame
            Loaded lyrics data with artist, song, and text columns

        Raises
        ------
        FileNotFoundError
            If the file does not exist
        ValueError
            If expected columns ('artist', 'song', 'text') are missing
        RuntimeError
            If any other error occurs during loading
        """
        filepath = self.data_directory / filename

        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {filepath}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load lyrics data from {filepath}") from e

        # Validation (fails if required columns are missing)
        expected_cols = ['artist', 'song', 'text']
        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing expected column(s): {', '.join(missing)}")

        self.lyrics_data = df
        return df


    def load_all_data(
        self,
        gtzan_file: str = 'features_3_sec.csv', 
        spotify_file: str = 'spotify_top_songs_audio_features.csv',
        lyrics_file: str = 'spotify_millsongdata.csv'
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all three datasets at once.

        Parameters
        ----------
        gtzan_file : str, default='features_3_sec.csv'
            GTZAN filename
        spotify_file : str, default='spotify_top_songs_audio_features.csv'
            Spotify filename
        lyrics_file : str, default='spotify_millsongdata.csv'
            Lyrics filename

        Returns
        -------
        tuple of pd.DataFrame
            A tuple containing (gtzan_data, spotify_data, lyrics_data)

        Raises
        ------
        FileNotFoundError
            If any of the files do not exist
        ValueError
            If expected columns are missing in any dataset
        RuntimeError
            If any other error occurs during loading
        """

        self.load_gtzan_data(gtzan_file)
        self.load_spotify_data(spotify_file)
        self.load_lyrics_data(lyrics_file)
            
        print("=" * 60)
        print("All datasets loaded successfully!")
        print("=" * 60)

        return self.gtzan_data, self.spotify_data, self.lyrics_data

    def get_data_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of the loaded datasets.

        Returns
        -------
        dict of str to dict
            Dictionary with dataset names as keys ('gtzan', 'spotify', 'lyrics')
            and summary dictionaries as values containing:
            - 'shape': tuple of (rows, columns)
            - 'columns': list of column names
            - 'genres': list of unique genres (GTZAN only)
            - 'missing_values': count of missing values
        """
        data_summary = {}

        if self.gtzan_data is not None:
            data_summary['gtzan'] = {
                'shape': self.gtzan_data.shape,
                'columns': list(self.gtzan_data.columns),
                'genres': self.gtzan_data['label'].unique().tolist() if 'label' in self.gtzan_data.columns else None,
                'missing_values': self.gtzan_data.isnull().sum().sum()
            }
        
        if self.spotify_data is not None:
            data_summary['spotify'] = {
                'shape': self.spotify_data.shape,
                'columns': list(self.spotify_data.columns),
                'missing_values': self.spotify_data.isnull().sum().sum()
            }
        
        if self.lyrics_data is not None:
            data_summary['lyrics'] = {
                'shape': self.lyrics_data.shape,
                'columns': list(self.lyrics_data.columns),
                'missing_values': self.lyrics_data.isnull().sum().sum()
            }
        
        return data_summary

    def save_processed_data(
        self,
        data: pd.DataFrame,
        filename: str, subdir: str = 'processed'
        ) -> Path:
        """
        Save processed data to CSV.

        Parameters
        ----------
        data : pd.DataFrame
            Data to save
        filename : str
            Output filename
        subdir : str, default='processed'
            Subdirectory within data_directory

        Returns
        -------
        Path
            Path to the saved CSV file

        Raises
        ------
        RuntimeError
            If saving fails
        """
        output_dir = self.data_directory / subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename

        try:
            data.to_csv(filepath, index=False)
        except Exception as e:
            raise RuntimeError(f"Failed to save data to {filepath}") from e

        return filepath

# Just a little simple test when running this file directly (not imported as a module)

if __name__ == "__main__":
    # Example usage
    loader = DataLoader('/files/project-MOISE/data')
    
    # Load all datasets
    gtzan, spotify, lyrics = loader.load_all_data()
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    summary = loader.get_data_summary()
    for dataset_name, info in summary.items():
        print(f"\n{dataset_name.upper()}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

