"""
Lyrics Analysis Module (NLP)

This module performs Natural Language Processing on song lyrics including:
- Sentiment analysis
- Word frequency analysis
- Topic modeling
- Text statistics
"""
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Counter as CounterType

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk import download as nltk_download

    try:
        # Ensure required NLTK data is present; download if missing
        stopwords.words('english')
    except LookupError:
        print("NLTK data not found; downloading 'stopwords', 'punkt', and 'averaged_perceptron_tagger'...")
        for pkg in ('stopwords', 'punkt', 'averaged_perceptron_tagger'):
            try:
                nltk_download(pkg, quiet=True)
            except Exception as e:
                print(f"Warning: failed to download '{pkg}': {e}")
except ImportError:
    print("NLTK not fully installed. Some features may not work.")
    stopwords = None
    word_tokenize = None

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None
    print("TextBlob not installed. Sentiment analysis will be limited.")


class LyricsAnalyzer:
    """Class to perfom analysis on song lyrics."""

    def __init__(self, output_directory: str='/files/project-MOISE/results/figures') -> None:
        """Initializes the LyricAnalyzer

        Arguments:
            output_directory: Directory to save the analysis results.
        
        Returns: None
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        # Load the stop words
        try:
            if stopwords is not None:
                self.stop_words = set(stopwords.words('english'))
            else:
                self.stop_words = set()
                print("Stop words not available (NLTK not installed)")
        except Exception as e:
            self.stop_words = set()
            print(f"Stop words not available: {e}")

    def calculate_text_statistics(self, lyrics_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic text statistics.
        
        Arguments:
            lyrics_data (pd.DataFrame): Lyrics data with 'cleaned_text' column
            
        Returns:
            pd.DataFrame: Statistics
        """
        if 'cleaned_text' not in lyrics_data.columns:
            raise ValueError("The 'cleaned_text' column is required in the lyrics data.")

        # List of personal pronouns
        personal_pronouns = {
            'i', 'me', 'my', 'mine', 'myself',
            'we', 'us', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves'
        }

        stats = []
        for idx, text in enumerate(lyrics_data['cleaned_text']):
            words = text.lower().split()
            unique_words = set(words)

            # Count personal pronouns
            pronoun_count = sum(1 for word in words if word in personal_pronouns)
            pronoun_ratio = pronoun_count / len(words) if words else 0
            
            # Categorize personal level
            if pronoun_ratio > 0.15:
                personal_level = "High"
            elif pronoun_ratio > 0.08:
                personal_level = "Medium"
            else:
                personal_level = "Low"
            
            stats.append({
                'char_count': len(text),
                'word_count': len(words),
                'unique_words': len(unique_words),
                'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
                'lexical_diversity': len(unique_words) / len(words) if words else 0,
                'personal_pronoun_count': pronoun_count,
                'personal_pronoun_ratio': pronoun_ratio,
                'personal_level': personal_level
            })

        stats_df = pd.DataFrame(stats)
        print("Text statistics summary:")
        print(stats_df.describe())

        return  stats_df
    
    def sentiment_analysis(self, lyrics_data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform sentiment analysis on lyrics.
        
        Arguments:
            lyrics_data (pd.DataFrame): Lyrics data with 'cleaned_text' column
            
        Returns:
            pd.DataFrame: Data with sentiment scores
        """
        if 'cleaned_text' not in lyrics_data.columns:
            raise ValueError("'cleaned_text' column not found in lyrics_data")

        sentiments = []

        # If TextBlob is unavailable, just return neutral scores for all texts
        if TextBlob is None:
            print("TextBlob is not installed. Returning neutral sentiment for all lyrics.")
            for _ in lyrics_data['cleaned_text']:
                sentiments.append({'polarity': 0.0, 'subjectivity': 0.0, 'sentiment': 'Neutral'})
        else:
            for text in lyrics_data['cleaned_text']:
                try:
                    blob = TextBlob(text)
                    polarity = float(blob.sentiment.polarity)
                    subjectivity = float(blob.sentiment.subjectivity)

                    if polarity > 0.1:
                        sentiment_label = 'Positive'
                    elif polarity < -0.1:
                        sentiment_label = 'Negative'
                    else:
                        sentiment_label = 'Neutral'

                    sentiments.append({
                        'polarity': polarity,
                        'subjectivity': subjectivity,
                        'sentiment': sentiment_label
                    })
                except Exception as e:
                    print(f"Warning: sentiment analysis failed for a lyric: {e}")
                    sentiments.append({'polarity': 0.0, 'subjectivity': 0.0, 'sentiment': 'Neutral'})

        sentiment_df = pd.DataFrame(sentiments)
        result_df = lyrics_data.copy()
        result_df['polarity'] = sentiment_df['polarity']
        result_df['subjectivity'] = sentiment_df['subjectivity']
        result_df['sentiment'] = sentiment_df['sentiment']

        print("\nSentiment Distribution:")
        print(sentiment_df['sentiment'].value_counts())
        print(f"\nAverage Polarity: {sentiment_df['polarity'].mean():.3f}")
        print(f"Average Subjectivity: {sentiment_df['subjectivity'].mean():.3f}")

        return result_df

    def plot_sentiment_distribution(self, sentiment_data, save_path = None, show=False) -> plt.Figure:
        """
        Plot sentiment distribution.
        
        Arguments:
            sentiment_data (pd.DataFrame): Data with sentiment columns
            save_path (str): Path to save the figure
        
        Returns: plt.Figure
        """
        if save_path is None:
            save_path = self.output_directory / "06_sentiment_distribution.png"
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Sentiment categories
        sentiment_counts = sentiment_data['sentiment'].value_counts()
        axes[0].bar(sentiment_counts.index, sentiment_counts.values, 
                   color=['green', 'gray', 'red'])
        axes[0].set_title('Sentiment Distribution', fontweight='bold')
        axes[0].set_xlabel('Sentiment')
        axes[0].set_ylabel('Count')
        axes[0].grid(alpha=0.3, axis='y')
        
        # Polarity distribution
        axes[1].hist(sentiment_data['polarity'], bins=50, color='blue', alpha=0.7)
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_title('Polarity Distribution', fontweight='bold')
        axes[1].set_xlabel('Polarity (-1 = Negative, +1 = Positive)')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(alpha=0.3)
        
        # Subjectivity distribution
        axes[2].hist(sentiment_data['subjectivity'], bins=50, color='purple', alpha=0.7)
        axes[2].set_title('Subjectivity Distribution', fontweight='bold')
        axes[2].set_xlabel('Subjectivity (0 = Objective, 1 = Subjective)')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        print(f"Sentiment distribution plot saved to {save_path}")

    def extract_word_frequencies(self, lyrics_data: pd.DataFrame, top_n: int=50) -> Counter:
        """
        Extract most common words from lyrics.
        
        Arguments:
            lyrics_data (pd.DataFrame): Lyrics data
            top_n (int): Number of top words to return
            
        Returns:
            Counter: Word frequencies
        """
        
        all_words = []
        for text in lyrics_data['cleaned_text']:
            words = text.lower().split()
            # Filter stopwords and short words
            words = [w for w in words if w not in self.stop_words and len(w) > 2]
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        n_print=min(10, top_n)

        print(f"\nTotal unique words: {len(word_freq)}")
        print(f"Total words: {sum(word_freq.values())}")
        print(f"\nTop {n_print} most common words:")
        for word, count in word_freq.most_common(n_print):
            print(f"  {word}: {count}")
        
        return word_freq
    
    def plot_word_frequencies(self, word_freq: Counter, top_n: int=20, save_path: Optional[Path]=None, show: bool=False) -> None:
        """
        Plot most common words.
        
        Arguments:
            word_freq (Counter): Word frequencies
            top_n (int): Number of top words to plot
            save_path (str): Path to save the figure
        
        Returns: None
        """
        if save_path is None:
            save_path = self.output_directory / "07_word_frequencies.png"
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
        top_words = dict(word_freq.most_common(top_n))
        
        plt.figure(figsize=(12, 8))
        plt.barh(list(top_words.keys()), list(top_words.values()), color='teal')
        plt.xlabel('Frequency', fontsize=12)
        plt.title(f'Top {top_n} Most Common Words in Lyrics', fontweight='bold', fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        print(f"Word frequencies plot saved to {save_path}")
    
    def analyze_song_themes(self, lyrics_data: pd.DataFrame) -> Dict[str, int]:
        """
        Identify common themes in lyrics using keyword analysis.
        
        Arguments:
            lyrics_data (pd.DataFrame): Lyrics data
            
        Returns:
            dict: Theme frequencies
        """
        if 'cleaned_text' not in lyrics_data.columns:
            raise ValueError("'cleaned_text' column not found in lyrics_data")
        
        if len(lyrics_data) == 0:
            print("No lyrics provided for theme analysis.")
            themes = {
                'Love': [], 'Party': [], 'Sadness': [], 'Happiness': [],
                'Freedom': [], 'Money': [], 'Life': [], 'Dreams': []
            }
            return {theme: 0 for theme in themes}

        # Define theme keywords
        themes = {
            'Love': ['love', 'heart', 'kiss', 'baby', 'forever', 'together'],
            'Party': ['party', 'dance', 'night', 'club', 'fun', 'drink'],
            'Sadness': ['cry', 'tears', 'sad', 'alone', 'broken', 'pain'],
            'Happiness': ['happy', 'smile', 'joy', 'sunshine', 'laugh'],
            'Freedom': ['free', 'fly', 'wild', 'run', 'escape'],
            'Money': ['money', 'cash', 'rich', 'gold', 'diamond'],
            'Life': ['life', 'live', 'time', 'day', 'world'],
            'Dreams': ['dream', 'hope', 'wish', 'star', 'sky']
        }
        
        theme_counts = {theme: 0 for theme in themes}
        
        for text in lyrics_data['cleaned_text']:
            text_lower = text.lower()
            for theme, keywords in themes.items():
                if any(keyword in text_lower for keyword in keywords):
                    theme_counts[theme] += 1
        
        print("\nTheme Distribution:")
        for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {theme}: {count} songs ({count/len(lyrics_data)*100:.1f}%)")
        
        return theme_counts
    
    def plot_themes(self, theme_counts: Dict[str, int], save_path: Optional[Path]=None, show: bool=False) -> None:
        """
        Plot theme distribution.
        
        Arguments:
            theme_counts (dict): Theme frequencies
            save_path (str): Path to save the figure

        Returns: None
        """
        if save_path is None:
            save_path = self.output_directory / "08_song_themes.png"
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
        sorted_themes = dict(sorted(theme_counts.items(), key=lambda x: x[1], reverse=True))
        
        plt.figure(figsize=(10, 6))
        plt.bar(sorted_themes.keys(), sorted_themes.values(), color='coral', alpha=0.7)
        plt.xlabel('Theme', fontsize=12)
        plt.ylabel('Number of Songs', fontsize=12)
        plt.title('Common Themes in Song Lyrics', fontweight='bold', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        print(f"Themes plot saved to {save_path}")

    def plot_personal_level_distribution(self, text_stats: pd.DataFrame, save_path: Optional[Path]=None, show: bool=False) -> None:
        """
        Plot the distribution of personal pronoun usage levels.
        Shows how personal/intimate the lyrics are.
        
        Arguments:
            text_stats (pd.DataFrame): Text statistics with personal_level column
            save_path (str): Path to save the figure
        
        Returns: None
        """
        if save_path is None:
            save_path = self.output_directory / "09_personal_level_distribution.png"
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if 'personal_level' not in text_stats.columns:
            raise ValueError("'personal_level' column not found in text_stats")
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Subplot 1: Pie chart of personal levels
        personal_counts = text_stats['personal_level'].value_counts()
        colors = ['#ff9999', '#ffcc99', '#99ccff']  # Red, Orange, Blue
        
        axes[0].pie(personal_counts.values, labels=personal_counts.index, 
                   autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0].set_title('Personal Level Distribution', fontweight='bold', fontsize=12)
        
        # Subplot 2: Histogram of personal pronoun ratio
        axes[1].hist(text_stats['personal_pronoun_ratio'], bins=30, 
                    color='skyblue', alpha=0.7, edgecolor='black')
        axes[1].axvline(text_stats['personal_pronoun_ratio'].mean(), 
                       color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {text_stats["personal_pronoun_ratio"].mean():.3f}')
        axes[1].set_xlabel('Personal Pronoun Ratio', fontsize=11)
        axes[1].set_ylabel('Number of Songs', fontsize=11)
        axes[1].set_title('Distribution of Personal Pronoun Usage', fontweight='bold', fontsize=12)
        axes[1].legend()
        axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        print(f"Personal level distribution plot saved to {save_path}")

    def generate_full_report(self, lyrics_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a complete NLP analysis report.
        
        Arguments:
            lyrics_data (pd.DataFrame): Cleaned lyrics data
        
        Returns: pd.DataFrame
        """
        print("\n" + "=" * 70)
        print("GENERATING COMPLETE LYRICS ANALYSIS REPORT")
        print("=" * 70)
        
        # Text statistics
        stats = self.calculate_text_statistics(lyrics_data)

        # Personal level distribution
        if stats is not None:
            self.plot_personal_level_distribution(stats)
        
        # Sentiment analysis
        sentiment_data = self.sentiment_analysis(lyrics_data)
        self.plot_sentiment_distribution(sentiment_data)
        
        # Word frequencies
        word_freq = self.extract_word_frequencies(lyrics_data)
        self.plot_word_frequencies(word_freq)
        
        # Theme analysis
        theme_counts = self.analyze_song_themes(lyrics_data)
        self.plot_themes(theme_counts)
        
        print("\n" + "=" * 70)
        print("LYRICS ANALYSIS REPORT COMPLETE!")
        print(f"All figures saved to: {self.output_directory}")
        print("=" * 70)
        
        return sentiment_data

if __name__ == "__main__":
    try:
        # Try relative imports when module is part of a package
        from .data_loader import DataLoader
        from .preprocessing import Preprocessor
    except Exception:
        # Fallback for running as a standalone script
        from data_loader import DataLoader
        from preprocessing import Preprocessor
    
    # Load and preprocess data
    loader = DataLoader('/files/project-MOISE/data')
    lyrics = loader.load_lyrics_data()
    
    preprocessor = Preprocessor()
    lyrics_clean = preprocessor.preprocess_lyrics(lyrics, sample_size=5000)
    
    # Analyze
    analyzer = LyricsAnalyzer()
    sentiment_data = analyzer.generate_full_report(lyrics_clean)
