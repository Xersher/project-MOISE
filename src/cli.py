"""
MOISE (Mood and Noise) - Command Line Interface

This CLI allows running the full analysis pipeline end-to-end without manual editing.

Implements PROPOSAL requirements:
- Command line interface to run the full workflow without manual editing
- Full pipeline from data loading to report generation

Usage:
    python cli.py --help
    python cli.py --all
    python cli.py --genre --clustering
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any


# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Try relative imports when module is part of a package
    from .data_loader import DataLoader
    from .preprocessing import Preprocessor
    from .genre_classifier import GenreClassifier
    from .mood_clustering import MoodClusterer
    from .popularity_predictor import PopularityPredictor
    from .lyrics_analyzer import LyricsAnalyzer
except Exception:
    # Fallback for running as a standalone script
    from data_loader import DataLoader
    from preprocessing import Preprocessor
    from genre_classifier import GenreClassifier
    from mood_clustering import MoodClusterer
    from popularity_predictor import PopularityPredictor
    from lyrics_analyzer import LyricsAnalyzer

class MOISEPipeline:
    """
    Main pipeline class for MOISE project.
    Orchestrates all analysis modules.
    """
    
    def __init__(self, data_dir: str='/files/project-MOISE/data', output_dir: str='/files/project-MOISE/results') -> None:
        """
        Initialize the MOISE pipeline.
        
        Args:
            data_dir (str): Directory containing datasets
            output_dir (str): Directory for results and reports
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        (self.output_dir / 'figures').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'models').mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.loader = DataLoader(str(self.data_dir))
        self.preprocessor = Preprocessor()
        
        print("MOISE Pipeline initialized")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
    
    def run_genre_classification(self) -> Optional[Dict[str, Any]]:
        """
        Run genre classification pipeline.
        
        Returns:
            A dictionary with:
                - 'results': full metrics per model (dict)
                - 'best_model': name of the best model (str)
                - 'confusion_matrix_path': path to saved confusion matrix figure (str)
            or None if the pipeline step failed.
        """
        print("\n" + "="*70)
        print("GENRE CLASSIFICATION")
        print("="*70)
        
        try:
            # Load GTZAN data
            print("Loading GTZAN dataset...")
            gtzan_data = self.loader.load_gtzan_data()
            
            if gtzan_data is None:
                print("Failed to load GTZAN data")
                return None
            
            # Preprocess
            print("Preprocessing data...")
            X_train, X_test, y_train, y_test, features, label_encoder = \
                self.preprocessor.preprocess_gtzan(gtzan_data)
            
            # Train classifier
            print("Training genre classifiers...")
            classifier = GenreClassifier(model_directory=str(self.output_dir / 'models'))
            classifier.create_models()
            results = classifier.train_and_evaluate(
                X_train, y_train, X_test, y_test, label_encoder
            )
            
            # Generate visualizations
            print("Generating visualizations...")
            fig_dir = self.output_dir / 'figures'
            
            classifier.plot_model_comparison(
                results, 
                save_path=str(fig_dir / '04_model_comparison.png')
            )
            
            confusion_path = str(fig_dir / '05_confusion_matrix.png')
            classifier.plot_confusion_matrix(
                y_test, 
                results[classifier.best_model_name]['predictions'],
                save_path=confusion_path
            )
            
            # Save model
            classifier.save_model()
            
            print("✓ Genre classification completed successfully")
            
            return {
                'results': results,
                'best_model': classifier.best_model_name,
                'confusion_matrix_path': confusion_path
            }
            
        except Exception as e:
            print(f"Genre classification failed: {e}!")
            import traceback
            traceback.print_exc()
            return None
    
    def run_clustering(self) -> Optional[Dict[str, Any]]:
        """
        Run mood clustering pipeline using the Circumplex Model of Affect (Russell, 1980).
        
        The Circumplex Model structures emotions along two dimensions:
        - Valence: Pleasure-Displeasure (positive to negative)
        - Arousal: Activation-Deactivation (high to low energy)
        
        These dimensions form 4 quadrants (K=4 clusters):
            - Q1: High Energy Positive (Happy, Excited)
            - Q2: High Energy Negative (Angry, Tense)
            - Q3: Low Energy Negative (Sad, Melancholic)
            - Q4: Low Energy Positive (Calm, Peaceful)
        
        Returns:
            A dictionary with:
                - 'descriptions': per-cluster feature summaries (dict)
                - 'names': mapping from cluster id to human-readable name (dict[int, str])
                - 'summary': text summary of clusters (str)
                - 'n_clusters': number of clusters (int, typically 4)
                - 'model': description of the model used (str)
                - 'circumplex_path': path to the Circumplex diagram figure (str)
                - 'cluster_plot_path': path to 2D cluster plot (str)
                - 'profiles_path': path to cluster profiles plot (str)
            or None if the pipeline step failed.
        """
        print("\n" + "="*70)
        print("MOOD CLUSTERING - Circumplex Model (Russell, 1980)")
        print("="*70)
        print("Using K=4 clusters based on psychological theory of emotions")
        print("Dimensions: Valence (pleasure) and Arousal (energy)")
        
        try:
            # Load Spotify data (contains valence and energy features)
            print("\nStep 1: Loading Spotify dataset...")
            spotify_data = self.loader.load_spotify_data()
            
            if spotify_data is None:
                print("Failed to load Spotify data")
                return None
            
            print(f"Loaded {len(spotify_data)} songs")
            
            # Check required features for Circumplex Model
            if 'valence' not in spotify_data.columns or 'energy' not in spotify_data.columns:
                print("Missing required features for Circumplex Model: valence and/or energy")
                return None
            
            # Select mood-related features for clustering
            print("\nStep 2: Selecting mood-related features...")
            mood_features = ['danceability', 'energy', 'valence', 'tempo', 
                           'loudness', 'acousticness', 'instrumentalness']
            available_features = [f for f in mood_features if f in spotify_data.columns]
            print(f"  ✓ Using features: {available_features}")
            
            # Initialize clusterer with Circumplex Model (K=4)
            print("\nStep 3: Initializing MoodClusterer (K=4, Circumplex Model)...")
            fig_dir = self.output_dir / 'figures'
            clusterer = MoodClusterer(output_directory=str(fig_dir))
            
            # Generate Circumplex Model theory visualization
            print("\nStep 4: Visualizing Circumplex Model theory...")
            circumplex_path = str(fig_dir / '15_circumplex_model.png')
            clusterer.visualize_circumplex_theory(
                spotify_data,
                valence_col='valence',
                energy_col='energy',
                save_path=circumplex_path,
                show=False
            )
            print(f"Circumplex Model visualization saved: {circumplex_path}")
            
            # Fit K-means clusters (K=4, reordered to match quadrants)
            print("\nStep 5: Fitting K-means clustering (K=4)...")
            cluster_labels = clusterer.fit_clusters(spotify_data, available_features)
            
            # Describe clusters with feature averages
            print("\nStep 6: Describing clusters...")
            cluster_descriptions = clusterer.describe_clusters(spotify_data, available_features)
            
            # Name clusters based on Circumplex quadrants
            print("\nStep 7: Naming clusters based on Circumplex quadrants...")
            cluster_names = clusterer.name_clusters()
            
            # Generate 2D PCA visualization
            print("\nStep 8: Creating 2D cluster visualization (PCA)...")
            cluster_plot_path = str(fig_dir / '13_mood_clusters_2d.png')
            clusterer.plot_clusters_2d(
                spotify_data, 
                available_features,
                save_path=cluster_plot_path,
                show=False
            )
            print(f"  ✓ 2D cluster plot saved: {cluster_plot_path}")
            
            # Generate cluster profiles
            print("\nStep 9: Creating cluster profiles...")
            profiles_path = str(fig_dir / '14_cluster_profiles.png')
            clusterer.plot_cluster_profiles(
                save_path=profiles_path,
                show=False
            )
            print(f"Cluster profiles saved: {profiles_path}")
            
            # Get text summary
            summary = clusterer.get_cluster_summary()
            
            # Print cluster distribution
            print("\n" + "="*50)
            print("CIRCUMPLEX MODEL CLUSTER SUMMARY")
            print("="*50)
            for cluster_id, name in cluster_names.items():
                count = (clusterer.cluster_labels == cluster_id).sum()
                pct = 100 * count / len(clusterer.cluster_labels)
                quadrant_info = clusterer.quadrants[cluster_id]
                emotions = ', '.join(quadrant_info['emotions'][:2])
                print(f"  Q{cluster_id+1}: {name}")
                print(f"      → {count} songs ({pct:.1f}%)")
                print(f"      → Emotions: {emotions}")
            print("="*50)
            
            print("\n✓ Mood clustering (Circumplex Model) completed successfully")
            
            return {
                'descriptions': cluster_descriptions,
                'names': cluster_names,
                'summary': summary,
                'n_clusters': 4,
                'model': 'Circumplex Model (Russell, 1980)',
                'circumplex_path': circumplex_path,
                'cluster_plot_path': cluster_plot_path,
                'profiles_path': profiles_path
            }
            
        except Exception as e:
            print(f"Mood clustering failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_popularity_prediction(self) -> Optional[Dict[str, Any]]:
        """
        Run popularity prediction pipeline.
        
        Returns:
            A dictionary with:
                - 'results':
                    - 'best_model': name of the best regression model (str)
                    - 'metrics': metric dict for the best model
                      (contains keys like 'train_r2', 'test_r2', 'mae', 'rmse', 'mape', 'cv_mean', 'cv_std')
                - 'comparison_path': path to model comparison figure (str)
                - 'predictions_path': path to actual vs predicted / residuals plot (str)
                - 'feature_importance_path': path to feature importance figure (str)
            or None if the pipeline step failed.
        """
        print("\n" + "="*70)
        print("POPULARITY PREDICTION")
        print("="*70)
        
        try:
            # Load Spotify data
            print("Loading Spotify dataset...")
            spotify_data = self.loader.load_spotify_data()
            
            if spotify_data is None:
                print("Failed to load Spotify data")
                return None
            
            # Preprocess
            print("Preprocessing data...")
            X_train, X_test, y_train, y_test, features = \
                self.preprocessor.preprocess_spotify(spotify_data)
            
            # Train predictor
            print("Training popularity predictor...")
            predictor = PopularityPredictor(model_directory=str(self.output_dir / 'models'))
            predictor.create_models()
            results = predictor.train_and_evaluate(X_train, X_test, y_train, y_test)
            
            # Generate visualizations
            print("Generating visualizations...")
            fig_dir = self.output_dir / 'figures'
            
            # Model comparison
            comparison_path = str(fig_dir / '10_popularity_model_comparison.png')
            if hasattr(predictor, 'plot_model_comparison'):
                predictor.plot_model_comparison(results, save_path=comparison_path)
            
            # Predictions scatter plot
            predictions_path = str(fig_dir / '11_popularity_predictions.png')
            predictor.plot_predictions(
                y_test, 
                results[predictor.best_model_name]['predictions'],
                save_path=predictions_path
            )
            
            # Feature importance
            feature_importance_path = str(fig_dir / '12_feature_importance.png')
            predictor.plot_feature_importance(
                features,
                save_path=feature_importance_path
            )
            
            # Save model
            predictor.save_model()
            
            print("✓ Popularity prediction completed successfully")
            
            return {
                'results': {
                    'best_model': predictor.best_model_name,
                    'metrics': results[predictor.best_model_name]
                },
                'comparison_path': comparison_path,
                'predictions_path': predictions_path,
                'feature_importance_path': feature_importance_path
            }
            
        except Exception as e:
            print(f"Popularity prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_lyrics_analysis(self) -> Optional[Dict[str, str]]:
        """
        Run lyrics analysis pipeline.

        Returns:
            A dictionary with paths to generated figures:
                - 'sentiment_path'
                - 'word_freq_path'
                - 'themes_path'
                - 'personal_level_path'
            (and any additional plots you actually generate),
            or None if the pipeline step failed.
        """
        print("\n" + "="*70)
        print("LYRICS ANALYSIS")
        print("="*70)
        
        try:
            # Load lyrics data
            print("Loading lyrics dataset...")
            lyrics_data = self.loader.load_lyrics_data()
            
            if lyrics_data is None:
                print("Failed to load lyrics data")
                return None
            
            # Preprocess
            print(f"Preprocessing lyrics...")
            lyrics_clean = self.preprocessor.preprocess_lyrics(
                lyrics_data,
            )
            
            # Analyze
            print("Analyzing lyrics...")
            analyzer = LyricsAnalyzer(output_directory=str(self.output_dir / 'figures'))
            
            # Generate full report (includes all visualizations)
            report = analyzer.generate_full_report(lyrics_clean)
            
            print("✓ Lyrics analysis completed successfully")
            
            fig_dir = self.output_dir / 'figures'
            
            return {
                'sentiment_path': str(fig_dir / '06_sentiment_distribution.png'),
                'word_freq_path': str(fig_dir / '07_word_frequencies.png'),
                'themes_path': str(fig_dir / '08_song_themes.png'),
                'personal_level_path': str(fig_dir / '09_personal_level_distribution.png'),
            }
            
        except Exception as e:
            print(f"Lyrics analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_all(self) -> None:
        """
        Run the complete MOISE pipeline:
        - Genre classification
        - Mood clustering (Circumplex Model)
        - Popularity prediction
        - Lyrics analysis
        
        Prints a summary of generated outputs and saves all figures/models
        under the configured output directory.
        """
        print("\n" + "="*70)
        print("RUNNING COMPLETE MOISE PIPELINE")
        print("="*70)
        print("Modules: Genre Classification, Mood Clustering (Circumplex),")
        print("         Popularity Prediction, Lyrics Analysis")
        
        # Run all analyses
        genre_results = self.run_genre_classification()
        clustering_results = self.run_clustering()
        popularity_results = self.run_popularity_prediction()
        lyrics_results = self.run_lyrics_analysis()
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED")
        print("="*70)
        print(f"All results saved to: {self.output_dir}")
        
        # Summary of generated files
        print("\nGenerated figures:")
        print("  EDA:")
        print("    - 01_gtzan_genre_distribution.png")
        print("    - 02_spotify_audio_features.png")
        print("    - 03_lyrics_top_artists.png")
        print("  Genre Classification:")
        print("    - 04_model_comparison.png")
        print("    - 05_confusion_matrix.png")
        print("  Lyrics Analysis:")
        print("    - 06_sentiment_distribution.png")
        print("    - 07_word_frequencies.png")
        print("    - 08_song_themes.png")
        print("    - 09_personal_level_distribution.png")
        print("  Popularity Prediction:")
        print("    - 10_popularity_model_comparison.png")
        print("    - 11_popularity_predictions.png")
        print("    - 12_feature_importance.png")
        print("  Mood Clustering (Circumplex Model):")
        print("    - 13_mood_clusters_2d.png")
        print("    - 14_cluster_profiles.png")
        print("    - 15_circumplex_model.png")
        


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='MOISE (Mood and Noise) - Music Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python cli.py --all
  
  # Run specific analyses
  python cli.py --genre
  python cli.py --clustering      # Uses Circumplex Model (K=4)
  python cli.py --popularity
  python cli.py --lyrics
  
  # Run multiple analyses
  python cli.py --genre --clustering --popularity
  
  # Specify custom directories
  python cli.py --all --data-dir ./my_data --output-dir ./my_results
  

Mood Clustering uses the Circumplex Model of Affect (Russell, 1980):
  - K=4 clusters corresponding to emotional quadrants
  - Q1: High Energy Positive (Happy, Excited)
  - Q2: High Energy Negative (Angry, Tense)
  - Q3: Low Energy Negative (Sad, Melancholic)
  - Q4: Low Energy Positive (Calm, Peaceful)
        """
    )
    
    # Pipeline options
    parser.add_argument('--all', action='store_true',
                       help='Run complete pipeline')
    parser.add_argument('--genre', action='store_true',
                       help='Run genre classification')
    parser.add_argument('--clustering', action='store_true',
                       help='Run mood clustering (Circumplex Model, K=4)')
    parser.add_argument('--popularity', action='store_true',
                       help='Run popularity prediction')
    parser.add_argument('--lyrics', action='store_true',
                       help='Run lyrics analysis')
    
    # Directory options
    parser.add_argument('--data-dir', default='/files/project-MOISE/data',
                       help='Directory containing datasets (default: data)')
    parser.add_argument('--output-dir', default='/files/project-MOISE/results',
                       help='Directory for results (default: results)')
    
    # Analysis options
    args = parser.parse_args()
    
    # Check if any action specified
    if not any([args.all, args.genre, args.clustering, args.popularity, 
                args.lyrics]):
        parser.print_help()
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = MOISEPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Run requested analyses
    genre_results = None
    clustering_results = None
    popularity_results = None
    lyrics_results = None
    
    if args.all:
        pipeline.run_all()
    else:
        if args.genre:
            genre_results = pipeline.run_genre_classification()
        
        if args.clustering:
            clustering_results = pipeline.run_clustering()
        
        if args.popularity:
            popularity_results = pipeline.run_popularity_prediction()
        
        if args.lyrics:
            lyrics_results = pipeline.run_lyrics_analysis()
    
    print("\nDone!")


if __name__ == "__main__":
    main()

