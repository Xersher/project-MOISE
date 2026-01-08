"""
MOISE (Mood and Noise) - Main Entry Point

This module runs the complete MOISE analysis pipeline, including:
1. Exploratory Data Analysis (EDA)
2. Genre Classification (Supervised Learning)
3. Popularity Prediction (Regression)
4. Lyrics Analysis (NLP)
5. Mood Clustering (Circumplex Model, K=4)

Usage:
    python main.py
    python main.py --data-dir ./my_data --output-dir ./my_results

Output:
    All results are saved to the output directory:
    - <output_dir>/figures/  : All visualizations (PNG)
    - <output_dir>/models/   : Trained models (PKL)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so local modules can be imported when running from other CWDs
project_root = Path(__file__).parent.resolve()
src_dir = project_root / 'src'
sys.path.insert(0, str(src_dir if src_dir.exists() else project_root))

from data_loader import DataLoader
from preprocessing import Preprocessor
from genre_classifier import GenreClassifier
from popularity_predictor import PopularityPredictor
from lyrics_analyzer import LyricsAnalyzer
from mood_clustering import MoodClusterer
from eda import EDAanalyser



def run_pipeline(data_dir, output_dir):
    """
    Run the complete MOISE pipeline.
    
    Args:
        data_dir (Path): Directory containing datasets
        output_dir (Path): Directory for results
        sample_size (int): Sample size for lyrics analysis
    """
    print("\n" + "="*70)
    print("  MOISE - Mood and Noise")
    print("  Music Analysis Pipeline")
    print("="*70)
    print("\nThis pipeline performs:")
    print("  1. Exploratory Data Analysis (EDA)")
    print("  2. Genre Classification (Random Forest, SVM, Gradient Boosting, MLP)")
    print("  3. Popularity Prediction (Linear, Ridge, Lasso, Random Forest, GB)")
    print("  4. Lyrics Analysis (Sentiment, Themes, Personal Pronouns)")
    print("  5. Mood Clustering (Circumplex Model, K=4)")
    print("="*70)
    
    # Convert to Path objects
    if not Path(data_dir).is_absolute():
        data_dir = project_root / data_dir
    else:
        data_dir = Path(data_dir)
    
    if not Path(output_dir).is_absolute():
        output_dir = project_root / output_dir
    else:
        output_dir = Path(output_dir)

    figures_dir = output_dir / 'figures'
    models_dir = output_dir / 'models'
    
    # Create output directories
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Initialize components
    print("\n" + "-"*70)
    print("Initializing components...")
    print("-"*70)
    
    loader = DataLoader(str(data_dir))
    preprocessor = Preprocessor()
    
    print("✓ DataLoader initialized")
    print("✓ Preprocessor initialized")
    
    # =========================================================================
    # 1. EXPLORATORY DATA ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    try:
        # Load all datasets
        print("\nLoading datasets...")
        gtzan_data = loader.load_gtzan_data()
        spotify_data = loader.load_spotify_data()
        lyrics_data = loader.load_lyrics_data()
        
        if gtzan_data is not None:
            print(f"  ✓ GTZAN: {len(gtzan_data)} samples")
        if spotify_data is not None:
            print(f"  ✓ Spotify: {len(spotify_data)} songs")
        if lyrics_data is not None:
            print(f"  ✓ Lyrics: {len(lyrics_data)} songs")
        
        # Run EDA
        print("\nGenerating EDA visualizations...")
        eda = EDAanalyser(output_directory=str(figures_dir))
        eda.generate_full_report(gtzan_data, spotify_data, lyrics_data)
        
        print("\n✓ EDA completed successfully")
        
    except Exception as e:
        print(f"❌ EDA failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # 2. GENRE CLASSIFICATION
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: GENRE CLASSIFICATION")
    print("="*70)
    
    try:
        # Load and preprocess GTZAN data
        print("\nLoading GTZAN dataset...")
        gtzan_data = loader.load_gtzan_data()
        
        if gtzan_data is None:
            print("❌ Failed to load GTZAN data")
        else:
            print(f"  ✓ Loaded {len(gtzan_data)} samples")
            
            print("\nPreprocessing data...")
            X_train, X_test, y_train, y_test, features, label_encoder = \
                preprocessor.preprocess_gtzan(gtzan_data)
            print(f"  ✓ Train set: {len(X_train)} samples")
            print(f"  ✓ Test set: {len(X_test)} samples")
            print(f"  ✓ Features: {len(features)}")
            
            # Train classifiers
            print("\nTraining genre classifiers...")
            classifier = GenreClassifier(model_directory=str(models_dir))
            classifier.create_models()
            results = classifier.train_and_evaluate(
                X_train, y_train, X_test, y_test, label_encoder
            )
            
            # Generate visualizations
            print("\nGenerating visualizations...")
            classifier.plot_model_comparison(
                results, 
                save_path=str(figures_dir / '04_model_comparison.png')
            )
            print("  ✓ Model comparison plot saved")
            
            classifier.plot_confusion_matrix(
                y_test, 
                results[classifier.best_model_name]['predictions'],
                save_path=str(figures_dir / '05_confusion_matrix.png')
            )
            print("  ✓ Confusion matrix saved")
            
            # Save model
            classifier.save_model()
            print(f"  ✓ Best model saved: {classifier.best_model_name}")
            
            print(f"\n✓ Genre classification completed")
            print(f"  Best model: {classifier.best_model_name}")
            print(f"  Test accuracy: {results[classifier.best_model_name]['test_accuracy']:.2%}")
    
    except Exception as e:
        print(f"❌ Genre classification failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # 3. POPULARITY PREDICTION
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: POPULARITY PREDICTION")
    print("="*70)
    
    try:
        # Load and preprocess Spotify data
        print("\nLoading Spotify dataset...")
        spotify_data = loader.load_spotify_data()
        
        if spotify_data is None:
            print("❌ Failed to load Spotify data")
        else:
            print(f"  ✓ Loaded {len(spotify_data)} songs")
            
            print("\nPreprocessing data...")
            X_train, X_test, y_train, y_test, features = \
                preprocessor.preprocess_spotify(spotify_data)
            print(f"  ✓ Train set: {len(X_train)} samples")
            print(f"  ✓ Test set: {len(X_test)} samples")
            
            # Train predictors
            print("\nTraining popularity predictors...")
            predictor = PopularityPredictor(model_directory=str(models_dir))
            predictor.create_models()
            results = predictor.train_and_evaluate(X_train, X_test, y_train, y_test)
            
            # Generate visualizations
            print("\nGenerating visualizations...")
            
            if hasattr(predictor, 'plot_model_comparison'):
                predictor.plot_model_comparison(
                    results, 
                    save_path=str(figures_dir / '10_popularity_model_comparison.png')
                )
                print("  ✓ Model comparison plot saved")
            
            predictor.plot_predictions(
                y_test, 
                results[predictor.best_model_name]['predictions'],
                save_path=str(figures_dir / '11_popularity_predictions.png')
            )
            print("  ✓ Predictions plot saved")
            
            predictor.plot_feature_importance(
                features,
                save_path=str(figures_dir / '12_feature_importance.png')
            )
            print("  ✓ Feature importance plot saved")
            
            # Save model
            predictor.save_model()
            print(f"  ✓ Best model saved: {predictor.best_model_name}")
            
            print(f"\n✓ Popularity prediction completed")
            print(f"  Best model: {predictor.best_model_name}")
            print(f"  R² score: {results[predictor.best_model_name]['test_r2']:.4f}")
    
    except Exception as e:
        print(f"❌ Popularity prediction failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # 4. LYRICS ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: LYRICS ANALYSIS")
    print("="*70)
    
    try:
        # Load lyrics data
        print("\nLoading lyrics dataset...")
        lyrics_data = loader.load_lyrics_data()
        
        if lyrics_data is None:
            print("❌ Failed to load lyrics data")
        else:
            print(f"  ✓ Loaded {len(lyrics_data)} songs")
            
            # Preprocess lyrics
            print(f"\nPreprocessing lyrics...")
            lyrics_clean = preprocessor.preprocess_lyrics(lyrics_data)
            print(f"  ✓ Cleaned {len(lyrics_clean)} lyrics")
            
            # Analyze lyrics
            print("\nAnalyzing lyrics...")
            analyzer = LyricsAnalyzer(output_directory=str(figures_dir))
            report = analyzer.generate_full_report(lyrics_clean)
            
            print("\n✓ Lyrics analysis completed")
            print("  Generated visualizations:")
            print("    - 06_sentiment_distribution.png")
            print("    - 07_word_frequencies.png")
            print("    - 08_song_themes.png")
            print("    - 09_personal_level_distribution.png")
    
    except Exception as e:
        print(f"❌ Lyrics analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # 5. MOOD CLUSTERING (CIRCUMPLEX MODEL)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: MOOD CLUSTERING - Circumplex Model (Russell, 1980)")
    print("="*70)
    print("Using K=4 clusters based on psychological theory of emotions")
    print("Dimensions: Valence (pleasure) × Arousal (energy)")
    
    try:
        # Load Spotify data
        print("\nLoading Spotify dataset...")
        spotify_data = loader.load_spotify_data()
        
        if spotify_data is None:
            print("❌ Failed to load Spotify data")
        else:
            print(f"  ✓ Loaded {len(spotify_data)} songs")
            
            # Check required features
            if 'valence' not in spotify_data.columns or 'energy' not in spotify_data.columns:
                print("❌ Missing required features: valence and/or energy")
            else:
                # Select mood features
                print("\nSelecting mood-related features...")
                mood_features = ['danceability', 'energy', 'valence', 'tempo', 
                               'loudness', 'acousticness', 'instrumentalness']
                available_features = [f for f in mood_features if f in spotify_data.columns]
                print(f"  ✓ Using features: {available_features}")
                
                # Initialize clusterer
                print("\nInitializing MoodClusterer (K=4, Circumplex Model)...")
                clusterer = MoodClusterer(output_directory=str(figures_dir))
                
                # Visualize Circumplex theory
                print("\nVisualizing Circumplex Model theory...")
                clusterer.visualize_circumplex_theory(
                    spotify_data,
                    valence_col='valence',
                    energy_col='energy',
                    save_path=str(figures_dir / '15_circumplex_model.png'),
                    show=False
                )
                print("  ✓ Circumplex Model visualization saved")
                
                # Fit clusters
                cluster_labels = clusterer.fit_clusters(spotify_data, available_features)
                
                # Describe clusters
                cluster_descriptions = clusterer.describe_clusters(spotify_data, available_features)
                
                # Name clusters
                print("\nNaming clusters based on Circumplex quadrants...")
                cluster_names = clusterer.name_clusters()
                
                # Generate visualizations
                print("\nGenerating visualizations...")
                
                clusterer.plot_clusters_2d(
                    spotify_data, 
                    available_features,
                    save_path=str(figures_dir / '13_mood_clusters_2d.png'),
                    show=False
                )
                print("  ✓ 2D cluster plot saved")
                
                clusterer.plot_cluster_profiles(
                    save_path=str(figures_dir / '14_cluster_profiles.png'),
                    show=False
                )
                print("  ✓ Cluster profiles saved")
                
                # Print summary
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
                
                print("\n✓ Mood clustering completed successfully")
    
    except Exception as e:
        print(f"❌ Mood clustering failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("PIPELINE COMPLETED")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}")
    
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
    
    print("\nSaved models:")
    print(f"  - {models_dir}/genre_classifier.pkl")
    print(f"  - {models_dir}/popularity_predictor.pkl")
    
    print("\n" + "="*70)
    print("✓ MOISE Pipeline completed successfully!")
    print("="*70)


def main():
    """Main CLI function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='MOISE (Mood and Noise) - Music Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python main.py
  
  # Specify custom directories
  python main.py --data-dir ./my_data --output-dir ./my_results


Pipeline steps:
  1. EDA: Genre distribution, audio features, top artists
  2. Genre Classification: RF, SVM, GB, MLP on GTZAN
  3. Popularity Prediction: 5 regression models on Spotify
  4. Lyrics Analysis: Sentiment, themes, personal pronouns
  5. Mood Clustering: Circumplex Model (Russell, 1980) with K=4
        """
    )
    
    # Directory options
    parser.add_argument('--data-dir', default='/files/project-MOISE/data',
                       help='Directory containing datasets (default: data)')
    parser.add_argument('--output-dir', default='/files/project-MOISE/results',
                       help='Directory for results (default: results)')
    
    args = parser.parse_args()
    
    # Run pipeline
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
