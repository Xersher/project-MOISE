"""
Exploratory Data Analysis Module
================================

This module performs exploratory data analysis on the MOISE datasets:
- GTZAN: Genre distribution analysis
- Spotify: Audio features distribution
- Lyrics: Text length and artist analysis
"""

from pathlib import Path
from typing import Union, Optional

import pandas as pd
import matplotlib.pyplot as plt


class EDAanalyser:
    """
    Class for performing exploratory data analysis on music datasets.

    Provides methods to analyze and visualize the GTZAN, Spotify, and
    Lyrics datasets, generating summary statistics and distribution plots.

    Attributes
    ----------
    output_directory : Path
        Directory to save the generated figures
    """

    def __init__(
        self,
        output_directory: Union[str, Path] = 'files/project-MOISE/results/figures'
        ) -> None:

        """
        This method initializes the EDAanalyser class.
        
        Arguments:
            output_directory (Union[str, Path]): Directory to save the figures.

        Returns:
            None
        """

        self.output_directory: Path = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        print(f"Figures are saved to: {self.output_directory}")

    def infos_about_dataset(self, data: pd.DataFrame, dataset_name: str = 'dataset') -> None:
        """
        Print basic information about a dataset.

        Displays shape, data types, missing values, and numerical
        feature summary statistics.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to analyze
        dataset_name : str, default='dataset'
            Name used when printing information (for context)
        """

        print(f"Basic information about the {dataset_name}:")
        print("-"*70 + "\n")
        print(f"\nShape : {data.shape}\n")
        print(f"\nData Types:")
        print(data.dtypes.value_counts())
        
        print("\nMissing Values:")
        missing = data.isnull().sum()
        missing_percentage = (missing / len(data)) * 100
        missing_df = pd.DataFrame({'Missing Count': missing,'Percentage': missing_percentage})
        print(missing_df[missing_df['Missing Count'] > 0])
        
        print("\nNumerical Features Summary:")
        print(data.describe())


    def analyze_gtzan(
        self,
        gtzan_data: pd.DataFrame,
        save_figs: bool = True,
        show: bool = False
        ) -> plt.Figure:
        """
        Perform EDA on GTZAN dataset.

        Generates genre distribution visualizations including bar chart
        and pie chart.

        Parameters
        ----------
        gtzan_data : pd.DataFrame
            GTZAN genre classification data with 'label' column
        save_figs : bool, default=True
            Whether to save figures to output_directory
        show : bool, default=False
            Whether to display figures interactively

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object
        """

        print("GTZAN DATASET ANALYSIS")
        print("-" * 70)
        
        self.infos_about_dataset(gtzan_data, "GTZAN Genre Classification")
        
        # Genre distribution
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        # Count plot
        genre_counts = gtzan_data['label'].value_counts()
        genre_counts.plot(kind='bar', ax=ax[0], color='blue', edgecolor='black')
        ax[0].set_title('Distribution of Music Genres', fontsize=14, fontweight='bold')
        ax[0].set_xlabel('Genre')
        ax[0].set_ylabel('Count')
        ax[0].tick_params(axis='x', rotation=45)
        
        # Pie chart
        genre_counts.plot(kind='pie', ax=ax[1], autopct='%1.1f%%', startangle=90)
        ax[1].set_ylabel('')
        ax[1].set_title('Genre Distribution (%)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_figs:
            plt.savefig(self.output_directory / '01_gtzan_genre_distribution.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        return fig


    def analyze_spotify(
        self,
        spotify_data: pd.DataFrame,
        save_figs: bool = True,
        show: bool = False
        ) -> Optional[plt.Figure]:
        """
        Perform EDA on Spotify dataset.

        Generates audio feature distribution histograms for danceability,
        energy, speechiness, acousticness, instrumentalness, liveness,
        and valence.

        Parameters
        ----------
        spotify_data : pd.DataFrame
            Spotify audio features data
        save_figs : bool, default=True
            Whether to save figures to output_directory
        show : bool, default=False
            Whether to display figures interactively

        Returns
        -------
        matplotlib.figure.Figure or None
            The generated figure object, or None if no audio features
            are available in the data
        """
        print("SPOTIFY DATASET ANALYSIS")
        print("=" * 70)

        # Basic stats
        self.infos_about_dataset(spotify_data, "Spotify Top Songs")

        # Audio features distribution
        audio_features = ['danceability', 'energy', 'speechiness', 'acousticness',
                        'instrumentalness', 'liveness', 'valence']
        audio_features = [f for f in audio_features if f in spotify_data.columns]

        # Initialize fig to None in case no features are available
        fig = None

        if len(audio_features) > 0:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.ravel()

            for i, feature in enumerate(audio_features):
                axes[i].hist(spotify_data[feature].dropna(), bins=50, 
                            color='red', alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{feature.capitalize()}', fontweight='bold')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(alpha=0.3)

            # Hide ALL extra subplots (not just the last one)
            for j in range(len(audio_features), len(axes)):
                axes[j].axis('off')

            plt.suptitle('Spotify Audio Features Distribution', fontsize=16, fontweight='bold')
            plt.tight_layout()
            if save_figs:
                plt.savefig(self.output_directory / '02_spotify_audio_features.png', 
                        dpi=300, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close()
        else:
            print("Warning: No audio features found in Spotify data")

        return fig


    def analyze_lyrics(
        self,
        lyrics_data: pd.DataFrame,
        save_figs: bool = True,
        show: bool = False
        ) -> plt.Figure:
        """
        Perform EDA on Lyrics dataset.

        Generates text length distribution plots and top artists bar chart.

        Parameters
        ----------
        lyrics_data : pd.DataFrame
            Lyrics data with 'text' and/or 'artist' columns
        save_figs : bool, default=True
            Whether to save figures to output_directory
        show : bool, default=False
            Whether to display figures interactively

        Returns
        -------
        matplotlib.figure.Figure or None
            The generated figure object, or None if no visualizations
            could be created (missing required columns)
        """
        print("\n" + "=" * 70)
        print("LYRICS DATASET ANALYSIS")
        print("=" * 70)
        
        # Basic stats
        self.infos_about_dataset(lyrics_data, "Million Song Dataset (Lyrics)")

        fig = None
        
        # Text length analysis
        if 'text' in lyrics_data.columns:
            lyrics_data['text_length'] = lyrics_data['text'].astype(str).apply(len)
            lyrics_data['word_count'] = lyrics_data['text'].astype(str).apply(lambda x: len(x.split()))
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Character length
            axes[0].hist(lyrics_data['text_length'], bins=50, color='orange', alpha=0.7, edgecolor='black')
            axes[0].set_title('Distribution of Lyrics Length (Characters)', fontweight='bold')
            axes[0].set_xlabel('Number of Characters')
            axes[0].set_ylabel('Frequency')
            axes[0].grid(alpha=0.3)
            
            # Word count
            axes[1].hist(lyrics_data['word_count'], bins=50, 
                        color='red', alpha=0.7, edgecolor='black')
            axes[1].set_title('Distribution of Lyrics Length (Words)', fontweight='bold')
            axes[1].set_xlabel('Number of Words')
            axes[1].set_ylabel('Frequency')
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            if save_figs:
                plt.savefig(self.output_directory / '09_lyrics_length_distribution.png', 
                          dpi=300, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close()
            
            print(f"\nAverage lyrics length: {lyrics_data['text_length'].mean():.0f} characters")
            print(f"Average word count: {lyrics_data['word_count'].mean():.0f} words")
        
        # Top artists
        if 'artist' in lyrics_data.columns:
            top_artists = lyrics_data['artist'].value_counts().head(15)
            
            fig_artist = plt.figure(figsize=(12, 6))
            top_artists.plot(kind='barh', color='darkgreen', edgecolor='black')
            plt.title('Top 15 Artists by Number of Songs', fontsize=14, fontweight='bold')
            plt.xlabel('Number of Songs')
            plt.ylabel('Artist')
            plt.grid(alpha=0.3, axis='x')
            plt.tight_layout()
            if save_figs:
                plt.savefig(self.output_directory / '03_lyrics_top_artists.png', dpi=300, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close()
            
            # Return artists figure if text figure wasn't created
            if fig is None:
                fig = fig_artist
        return fig

    def generate_full_report(
        self,
        gtzan_data: Optional[pd.DataFrame]=None,
        spotify_data: Optional[pd.DataFrame]=None,
        lyrics_data: Optional[pd.DataFrame]=None
        ) -> None:
        """
        Generate a complete EDA report for all datasets.

        Runs analysis methods for each provided dataset and saves
        all figures to the output directory.

        Parameters
        ----------
        gtzan_data : pd.DataFrame, optional
            GTZAN genre classification data
        spotify_data : pd.DataFrame, optional
            Spotify audio features data
        lyrics_data : pd.DataFrame, optional
            Lyrics data with text and artist columns
        """
        print("\n" + "=" * 70)
        print("GENERATING COMPLETE EDA REPORT")
        print("=" * 70)
        
        if gtzan_data is not None:
            self.analyze_gtzan(gtzan_data)
        
        if spotify_data is not None:
            self.analyze_spotify(spotify_data)
        
        if lyrics_data is not None:
            self.analyze_lyrics(lyrics_data)
        
        print("\n" + "=" * 70)
        print("EDA REPORT COMPLETE!")
        print(f"All figures saved to: {self.output_directory}")
        print("=" * 70)


if __name__ == "__main__":
    try:
        # Try relative imports when module is part of a package
        from .data_loader import DataLoader
    except Exception:
        # Fallback for running as a standalone script
        from data_loader import DataLoader
    
    # Load data
    loader = DataLoader('/files/project-MOISE/data')
    gtzan, spotify, lyrics = loader.load_all_data()
    
    # Run EDA
    eda = EDAanalyser('/files/project-MOISE/results/figures')
    eda.generate_full_report(gtzan, spotify, lyrics)