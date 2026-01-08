"""
Mood Clustering Module - Circumplex Model Implementation

This module implements unsupervised learning (K-means) to discover "mood families" 
by clustering songs based on their sonic profile, guided by the Circumplex Model 
of Affect (Russell, 1980).

The Circumplex Model structures emotions along two fundamental dimensions:
1. Valence: Pleasure-Displeasure (positive-negative)
2. Arousal: Activation-Deactivation (high energy-low energy)

These two dimensions form four quadrants representing distinct emotional states,
naturally suggesting K=4 clusters for mood classification.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns



from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


class MoodClusterer:
    """
    Mood clustering based on the Circumplex Model of Affect (Russell, 1980).
    
    Uses K=4 clusters corresponding to the four quadrants of the Circumplex:
    - Quadrant 1: High Energy, Positive Valence (Happy/Excited)
    - Quadrant 2: High Energy, Negative Valence (Angry/Tense)
    - Quadrant 3: Low Energy, Negative Valence (Sad/Melancholic)
    - Quadrant 4: Low Energy, Positive Valence (Calm/Peaceful)
    """
    
    def __init__(self, output_directory: str = '/files/project-MOISE/results/figures') -> None:
        """
        Initialize the Mood Clusterer with K=4 (Circumplex Model).
        
        Arguments:
            output_directory (str): Directory to save figures
        """
        self.n_clusters = 4  # Fixed at 4 based on Circumplex Model
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.kmeans = None
        self.pca = None
        self.cluster_labels = None
        self.cluster_descriptions = None
        
        # Circumplex quadrant definitions
        self.quadrants = {
            0: {
                'name': 'High Energy Positive',
                'emotions': ['Happy', 'Excited', 'Energetic', 'Joyful'],
                'color': '#FFD700'  # Gold
            },
            1: {
                'name': 'High Energy Negative',
                'emotions': ['Angry', 'Tense', 'Aggressive', 'Intense'],
                'color': '#FF4444'  # Red
            },
            2: {
                'name': 'Low Energy Negative',
                'emotions': ['Sad', 'Depressed', 'Melancholic', 'Gloomy'],
                'color': '#4444FF'  # Blue
            },
            3: {
                'name': 'Low Energy Positive',
                'emotions': ['Calm', 'Peaceful', 'Relaxed', 'Serene'],
                'color': '#44FF44'  # Green
            }
        }
        
        print(f"MoodClusterer initialized with K={self.n_clusters} (Circumplex Model)")
        print("Based on Russell (1980): 2D emotion space (Valence × Arousal)")
    
    def visualize_circumplex_theory(self, data: pd.DataFrame, valence_col: str = 'valence', energy_col: str = 'energy', save_path: Optional[Path] = None,
                                    show: bool = False) -> None:
        """
        Visualize the theoretical Circumplex Model with actual data distribution.
        
        Arguments:
            data (pd.DataFrame): Music data
            valence_col (str): Column name for valence
            energy_col (str): Column name for energy
            save_path (Optional[Path]): Path to save figure
            show (bool): Whether to display the plot
        """
        if save_path is None:
            save_path = self.output_directory / '15_circumplex_model.png'
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get valence and energy
        valence = data[valence_col].values
        energy = data[energy_col].values
        
        # Assign theoretical quadrants
        quadrants = np.zeros(len(data), dtype=int)
        quadrants[(energy >= 0.5) & (valence >= 0.5)] = 0  # Q1: High Energy Positive
        quadrants[(energy >= 0.5) & (valence < 0.5)] = 1   # Q2: High Energy Negative
        quadrants[(energy < 0.5) & (valence < 0.5)] = 2    # Q3: Low Energy Negative
        quadrants[(energy < 0.5) & (valence >= 0.5)] = 3   # Q4: Low Energy Positive
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Draw quadrant boundaries
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)
        
        # Draw quadrant backgrounds with labels
        quad_alpha = 0.15
        
        # Quadrant 0 (top-right): High Energy Positive
        ax.fill_between([0.5, 1.0], 0.5, 1.0, color=self.quadrants[0]['color'], 
                       alpha=quad_alpha, zorder=0)
        ax.text(0.75, 0.95, f"Q1: {self.quadrants[0]['name']}\n({', '.join(self.quadrants[0]['emotions'][:2])})", 
               ha='center', va='top', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Quadrant 1 (top-left): High Energy Negative
        ax.fill_between([0.0, 0.5], 0.5, 1.0, color=self.quadrants[1]['color'], 
                       alpha=quad_alpha, zorder=0)
        ax.text(0.25, 0.95, f"Q2: {self.quadrants[1]['name']}\n({', '.join(self.quadrants[1]['emotions'][:2])})", 
               ha='center', va='top', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Quadrant 2 (bottom-left): Low Energy Negative
        ax.fill_between([0.0, 0.5], 0.0, 0.5, color=self.quadrants[2]['color'], 
                       alpha=quad_alpha, zorder=0)
        ax.text(0.25, 0.05, f"Q3: {self.quadrants[2]['name']}\n({', '.join(self.quadrants[2]['emotions'][:2])})", 
               ha='center', va='bottom', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Quadrant 3 (bottom-right): Low Energy Positive
        ax.fill_between([0.5, 1.0], 0.0, 0.5, color=self.quadrants[3]['color'], 
                       alpha=quad_alpha, zorder=0)
        ax.text(0.75, 0.05, f"Q4: {self.quadrants[3]['name']}\n({', '.join(self.quadrants[3]['emotions'][:2])})", 
               ha='center', va='bottom', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot data points
        for quad_id in range(4):
            mask = quadrants == quad_id
            if mask.sum() > 0:
                ax.scatter(valence[mask], energy[mask], 
                         c=self.quadrants[quad_id]['color'],
                         label=f"{self.quadrants[quad_id]['name']} (n={mask.sum()})",
                         alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Add axis labels with references
        ax.set_xlabel('Valence (Pleasure)\nNegative ← → Positive', fontsize=14, fontweight='bold')
        ax.set_ylabel('Arousal (Activation)\nLow Energy ← → High Energy', fontsize=14, fontweight='bold')
        ax.set_title('Circumplex Model of Affect (Russell, 1980)\nApplied to Music Mood Classification', fontsize=16, fontweight='bold')
        
        # Set limits and aspect
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        print(f"✓ Circumplex Model visualization saved to {save_path}")
    
    def fit_clusters(self, data: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit K-means clustering on the data with K=4 (Circumplex Model).
        
        Reorders clusters to match Circumplex quadrants based on mean valence and energy.
        
        Args:
            data (pd.DataFrame): Music data
            feature_columns (Optional[List[str]]): Columns to use for clustering. 
                If None, uses all numeric columns
        
        Returns:
            np.ndarray: Cluster labels for each song (reordered to match quadrants)
        """
        print(f"\nFitting K-means clustering with K={self.n_clusters} (Circumplex Model)...")
        
        # Select features
        if feature_columns is None:
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        X = data[feature_columns].copy()
        
        # Remove any NaN values
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit K-means
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        raw_labels = self.kmeans.fit_predict(X_scaled)
        
        # REORDER CLUSTERS TO MATCH CIRCUMPLEX QUADRANTS
        # Map each K-means cluster to its corresponding Circumplex quadrant
        # based on mean valence and energy
        
        cluster_to_quadrant = {}
        
        for cluster_id in range(self.n_clusters):
            # Get songs in this cluster
            mask = raw_labels == cluster_id
            cluster_data = data[mask]
            
            # Calculate mean valence and energy
            if 'valence' in cluster_data.columns and 'energy' in cluster_data.columns:
                mean_valence = cluster_data['valence'].mean()
                mean_energy = cluster_data['energy'].mean()
            else:
                # Fallback: use cluster centers
                print("  ⚠️  Warning: valence/energy not found, using cluster ID as-is")
                cluster_to_quadrant[cluster_id] = cluster_id
                continue
            
            # Map to Circumplex quadrant
            if mean_energy >= 0.5 and mean_valence >= 0.5:
                quadrant = 0  # High Energy Positive
            elif mean_energy >= 0.5 and mean_valence < 0.5:
                quadrant = 1  # High Energy Negative
            elif mean_energy < 0.5 and mean_valence < 0.5:
                quadrant = 2  # Low Energy Negative
            elif mean_energy < 0.5 and mean_valence >= 0.5:
                quadrant = 3  # Low Energy Positive
            
            cluster_to_quadrant[cluster_id] = quadrant
            print(f"  K-means cluster {cluster_id} → Quadrant {quadrant} ({self.quadrants[quadrant]['name']})")
            print(f"  (mean valence={mean_valence:.3f}, mean energy={mean_energy:.3f})")
        
        # Check for conflicts (two clusters mapped to same quadrant)
        quadrant_counts = {}
        for cluster_id, quadrant in cluster_to_quadrant.items():
            quadrant_counts[quadrant] = quadrant_counts.get(quadrant, 0) + 1
        
        if any(count > 1 for count in quadrant_counts.values()):
            print("  ⚠️  Warning: Multiple clusters mapped to same quadrant!")
            print("  Using original K-means labels without reordering.")
            self.cluster_labels = raw_labels
        else:
            # Create mapping: old_cluster_id → new_cluster_id (quadrant)
            # Invert the mapping
            quadrant_to_cluster = {v: k for k, v in cluster_to_quadrant.items()}
            
            # Relabel: for each data point, map its old cluster to its quadrant
            self.cluster_labels = np.array([cluster_to_quadrant[label] for label in raw_labels])
            
            # Reorder cluster centers to match quadrants
            new_centers = np.zeros_like(self.kmeans.cluster_centers_)
            for quadrant in range(self.n_clusters):
                old_cluster = quadrant_to_cluster[quadrant]
                new_centers[quadrant] = self.kmeans.cluster_centers_[old_cluster]
            self.kmeans.cluster_centers_ = new_centers
            
            print("  ✓ Clusters reordered to match Circumplex quadrants!")
        
        # Compute Silhouette Score for quality assessment
        silhouette = silhouette_score(X_scaled, self.cluster_labels)
        
        print(f"\n✓ K-means clustering completed with {self.n_clusters} clusters")
        print(f"✓ Silhouette Score: {silhouette:.4f}")
        
        if silhouette > 0.35:
            print(f"  ✅ Good cluster quality for continuous data (score > 0.35)")
        else:
            print(f"  ⚠️  Lower quality but expected for continuous mood data")
        
        return self.cluster_labels
    
    def apply_pca(self, data: pd.DataFrame, feature_columns: Optional[List[str]] = None, 
                    n_components: int = 2, refit_scaler: bool = True) -> np.ndarray:
        """
        Apply PCA for dimensionality reduction and visualization.
        
        Arguments:
            data (pd.DataFrame): Music data
            feature_columns (Optional[List[str]]): Columns to use for PCA
            n_components (int): Number of PCA components
            refit_scaler (bool): Whether to refit the scaler
        
        Returns:
            np.ndarray: PCA-transformed data
        """
        print(f"Applying PCA with {n_components} components...")
        
        # Select features
        if feature_columns is None:
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        X = data[feature_columns].copy()
        X = X.fillna(X.mean())
        
        # Scale features
        if refit_scaler or not hasattr(self.scaler, 'mean_'):
            X_scaled = self.scaler.fit_transform(X)
            print("  ✓ Scaler fitted")
        else:
            X_scaled = self.scaler.transform(X)
            print("  ✓ Using existing scaler")
        
        # Apply PCA
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Log explained variance
        explained_var = self.pca.explained_variance_ratio_
        print(f"✓ PCA completed. Explained variance: {explained_var}")
        
        return X_pca
    
    def describe_clusters(self, data: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Describe each cluster using average values of key features.
        
        Since clusters are reordered to match quadrants, cluster_id = quadrant_id.
        
        Arguments:
            data (pd.DataFrame): Music data
            feature_columns (Optional[List[str]]): Features to describe clusters with
        
        Returns:
            pd.DataFrame: Cluster descriptions
        
        Raises:
            RuntimeError: If clusters haven't been fitted yet
        """
        print("Describing clusters...")
        
        if self.cluster_labels is None:
            raise RuntimeError("Must fit clusters first!")
        
        # Add cluster labels to data
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = self.cluster_labels
        
        # Select features to describe
        if feature_columns is None:
            possible_features = ['tempo', 'valence', 'energy', 'danceability', 
                                'loudness', 'acousticness', 'instrumentalness']
            feature_columns = [f for f in possible_features if f in data.columns]
    
        # Calculate averages per cluster
        descriptions = []
        for cluster_id in range(self.n_clusters):
            cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
            desc = {'cluster': cluster_id, 'count': len(cluster_data)}
            
            for feature in feature_columns:
                if feature in cluster_data.columns:
                    desc[f'avg_{feature}'] = cluster_data[feature].mean()
            
            descriptions.append(desc)
        
        self.cluster_descriptions = pd.DataFrame(descriptions)
        
        # Print descriptions
        print("\n" + "=" * 70)
        print("CLUSTER DESCRIPTIONS (Circumplex Model)")
        print("=" * 70)
        for idx, row in self.cluster_descriptions.iterrows():
            cluster_id = int(row['cluster'])
            quadrant_name = self.quadrants[cluster_id]['name']
            quadrant_emotions = ', '.join(self.quadrants[cluster_id]['emotions'][:3])
            
            print(f"\nCluster {cluster_id} = Quadrant {cluster_id+1}: {quadrant_name}")
            print(f"  Emotions: {quadrant_emotions}")
            print(f"  Size: {int(row['count'])} songs")
            print(f"  Characteristics:")
            
            for col in self.cluster_descriptions.columns:
                if col not in ['cluster', 'count']:
                    print(f"    {col}: {row[col]:.3f}")
        print("=" * 70)
        
        return self.cluster_descriptions
    
    def name_clusters(self) -> Optional[Dict[int, str]]:
        """
        Name clusters based on Circumplex Model quadrants.
        
        Since clusters are reordered to match quadrants, cluster_id = quadrant_id.
        
        Returns:
            Optional[Dict[int, str]]: Mapping of cluster ID to name
        """
        if self.cluster_descriptions is None:
            print("Must describe clusters first!")
            return None
        
        cluster_names = {}
        
        for idx, row in self.cluster_descriptions.iterrows():
            cluster_id = int(row['cluster'])
            
            # After reordering, cluster_id directly corresponds to quadrant
            # Quadrant 0: High Energy Positive
            # Quadrant 1: High Energy Negative
            # Quadrant 2: Low Energy Negative
            # Quadrant 3: Low Energy Positive
            
            base_name = self.quadrants[cluster_id]['name']
            
            # Add modifiers based on other features
            modifiers = []
            
            if 'avg_tempo' in row:
                tempo = row['avg_tempo']
                if tempo > 140:
                    modifiers.append("Very Fast")
                elif tempo < 80:
                    modifiers.append("Slow")
            
            if 'avg_danceability' in row:
                if row['avg_danceability'] > 0.75:
                    modifiers.append("Danceable")
            
            if 'avg_acousticness' in row:
                if row['avg_acousticness'] > 0.6:
                    modifiers.append("Acoustic")
            
            # Build name
            if modifiers:
                cluster_names[cluster_id] = f"{base_name} ({modifiers[0]})"
            else:
                cluster_names[cluster_id] = base_name
        
        # Print names
        print("\n" + "=" * 70)
        print("CLUSTER NAMES (Circumplex Quadrants)")
        print("=" * 70)
        for cluster_id in sorted(cluster_names.keys()):
            count = int(self.cluster_descriptions.iloc[cluster_id]['count'])
            emotions = ', '.join(self.quadrants[cluster_id]['emotions'][:2])
            print(f"Cluster {cluster_id}: {cluster_names[cluster_id]}")
            print(f"  Quadrant: {self.quadrants[cluster_id]['name']}")
            print(f"  Emotions: {emotions}")
            print(f"  Songs: {count}")
        print("=" * 70)
        
        return cluster_names
    
    def plot_clusters_2d(self, data: pd.DataFrame, feature_columns: Optional[List[str]] = None, 
                        save_path: Optional[Path] = None, show: bool = False, refit: bool = False) -> None:
        """
        Visualize clusters in 2D space using PCA.
        
        Arguments:
            data (pd.DataFrame): Music data
            feature_columns (Optional[List[str]]): Features for PCA
            save_path (Optional[Path]): Path to save figure
            show (bool): Whether to display the plot
            refit (bool): Force refitting of clusters
        """
        print("Creating 2D cluster visualization...")
        
        if save_path is None:
            save_path = self.output_directory / '13_mood_clusters_2d.png'
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Fit clusters if needed
        if self.cluster_labels is None or refit:
            print("  Fitting clusters...")
            self.fit_clusters(data, feature_columns)
        
        # Apply PCA
        print("  Applying PCA...")
        if feature_columns is None:
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        X = data[feature_columns].copy()
        X = X.fillna(X.mean())
        X_scaled = self.scaler.transform(X)
        
        if self.pca is None or refit:
            self.pca = PCA(n_components=2)
            X_pca = self.pca.fit_transform(X_scaled)
            explained_var = self.pca.explained_variance_ratio_
            print(f"  ✓ PCA completed. Explained variance: {explained_var}")
        else:
            X_pca = self.pca.transform(X_scaled)
        
        # Get cluster names
        cluster_names = self.name_clusters()
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot each cluster with Circumplex colors
        for cluster_id in range(self.n_clusters):
            mask = self.cluster_labels == cluster_id
            cluster_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            n_points = mask.sum()
            
            if n_points > 0:
                plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"{cluster_name} (n={n_points})",
                color=self.quadrants[cluster_id]['color'],alpha=0.6, s=50, edgecolors='black',linewidth=0.5)
        
        # Add cluster centers
        centers_scaled = self.scaler.transform(self.kmeans.cluster_centers_)
        centers_pca = self.pca.transform(centers_scaled)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', marker='X', s=300, edgecolors='white',linewidths=2,
            label='Cluster Centers',zorder=5)
        
        plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        plt.title(f'Mood Clusters in 2D PCA Space (K={self.n_clusters}, Circumplex Model)', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        print(f"✓ 2D cluster visualization saved to {save_path}")
    
    def plot_cluster_profiles(self, save_path: Optional[Path] = None, show: bool = False) -> None:
        """
        Create bar plots showing the profile of each cluster.
        
        Arguments:
            save_path (Optional[Path]): Path to save figure
            show (bool): Whether to display the plot
        
        Raises:
            RuntimeError: If clusters haven't been described yet
        """
        if self.cluster_descriptions is None:
            raise RuntimeError("Must describe clusters first!")
        
        print("Creating cluster profile visualization...")
        
        if save_path is None:
            save_path = self.output_directory / '14_cluster_profiles.png'
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get feature columns
        feature_cols = [col for col in self.cluster_descriptions.columns 
                       if col not in ['cluster', 'count']]
        
        # Normalize features to 0-1
        df_norm = self.cluster_descriptions.copy()
        for col in feature_cols:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val > min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        
        # Create subplots
        fig, axes = plt.subplots(1, self.n_clusters, figsize=(5*self.n_clusters, 6))
        
        cluster_names = self.name_clusters()
        
        for idx, ax in enumerate(axes):
            cluster_data = df_norm[df_norm['cluster'] == idx]
            if len(cluster_data) == 0:
                continue
            
            values = [cluster_data.iloc[0][col] for col in feature_cols]
            
            # Use Circumplex colors
            color = self.quadrants[idx]['color']
            bars = ax.barh(range(len(feature_cols)), values, color=color, alpha=0.7)
            
            ax.set_yticks(range(len(feature_cols)))
            ax.set_yticklabels([col.replace('avg_', '').title() for col in feature_cols])
            ax.set_xlim(0, 1)
            ax.set_xlabel('Normalized Value', fontsize=10)
            ax.set_title(
                f'{cluster_names.get(idx, f"Cluster {idx}")}\n({int(self.cluster_descriptions.iloc[idx]["count"])} songs)', 
                fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        
        plt.suptitle('Cluster Profiles (Circumplex Model)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        print(f"✓ Cluster profiles saved to {save_path}")
    
    def get_cluster_summary(self) -> str:
        """
        Get a text summary of all clusters based on Circumplex Model.
        
        Returns:
            str: Markdown-formatted summary
        """
        if self.cluster_descriptions is None:
            return "No clusters fitted yet."
        
        cluster_names = self.name_clusters()
        
        summary = "## Mood Cluster Summary (Circumplex Model)\n\n"
        summary += f"Based on Russell's (1980) Circumplex Model of Affect, we identified **{self.n_clusters}** distinct mood quadrants:\n\n"
        
        for idx, row in self.cluster_descriptions.iterrows():
            cluster_id = int(row['cluster'])
            name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            count = int(row['count'])
            quadrant = self.quadrants[cluster_id]
            
            summary += f"### Quadrant {cluster_id+1}: {name}\n"
            summary += f"- **Size**: {count} songs ({100*count/self.cluster_descriptions['count'].sum():.1f}%)\n"
            summary += f"- **Emotions**: {', '.join(quadrant['emotions'])}\n"
            
            # Add key characteristics
            feature_cols = [col for col in self.cluster_descriptions.columns 
                           if col not in ['cluster', 'count']]
            
            for col in feature_cols:
                val = row[col]
                feature_name = col.replace('avg_', '').title()
                summary += f"- **{feature_name}**: {val:.3f}\n"
            
            summary += "\n"
        
        return summary


if __name__ == "__main__": 
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        from .data_loader import DataLoader
    except ImportError:
        sys.path.append(str(Path(__file__).resolve().parent))
        from data_loader import DataLoader
    
    print("\n" + "="*70)
    print("MOOD CLUSTERING - Circumplex Model (Russell, 1980)")
    print("="*70)
    print("K=4 clusters based on psychological theory of emotions\n")
    
    # Load Spotify data
    print("Step 1: Loading Spotify dataset...")
    loader = DataLoader('/files/project-MOISE/data')
    spotify_data = loader.load_spotify_data()
    
    if spotify_data is None:
        print("Failed to load Spotify data")
        sys.exit(1)
    
    print(f"✓ Loaded {len(spotify_data)} songs")
    print(f"✓ Available columns: {list(spotify_data.columns)}\n")
    
    # Check required features
    if 'valence' not in spotify_data.columns or 'energy' not in spotify_data.columns:
        print("Missing required features: valence and/or energy")
        sys.exit(1)
    
    # Select mood-related features
    print("Step 2: Selecting mood-related features...")
    mood_features = ['danceability', 'energy', 'valence', 'tempo', 
                     'loudness', 'acousticness', 'instrumentalness']
    available_features = [f for f in mood_features if f in spotify_data.columns]
    print(f"✓ Using features: {available_features}\n")
    
    # Initialize clusterer
    print("Step 3: Initializing MoodClusterer (K=4, Circumplex)...")
    clusterer = MoodClusterer(
        output_directory='/files/project-MOISE/results/figures'
    )
    
    # Visualize Circumplex theory
    print("\nStep 4: Visualizing Circumplex Model theory...")
    clusterer.visualize_circumplex_theory(spotify_data, valence_col='valence', energy_col='energy',show=False)
    
    # Fit clusters
    print("\nStep 5: Fitting K-means clustering (K=4)...")
    cluster_labels = clusterer.fit_clusters(spotify_data, available_features)
    
    # Describe clusters
    print("\nStep 6: Describing clusters...")
    cluster_descriptions = clusterer.describe_clusters(spotify_data, available_features)
    
    # Name clusters
    print("\nStep 7: Naming clusters based on Circumplex quadrants...")
    cluster_names = clusterer.name_clusters()
    
    # Generate visualizations
    print("\nStep 8: Generating visualizations...")
    
    # 2D visualization
    print("  - Creating 2D cluster visualization...")
    clusterer.plot_clusters_2d(spotify_data, available_features, show=False)
    
    # Cluster profiles
    print("  - Creating cluster profiles...")
    clusterer.plot_cluster_profiles(show=False)
    
    # Get summary
    print("\nStep 9: Generating cluster summary...")
    summary = clusterer.get_cluster_summary()
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(summary)
    
    print("\n" + "="*70)
    print("CLUSTERING COMPLETED SUCCESSFULLY")
    print("="*70)
    print("Generated files:")
    print("  - results/figures/13_mood_clusters_2d.png")
    print("  - results/figures/14_cluster_profiles.png")
    print("  - results/figures/15_circumplex_model.png")
    print(f"\nIdentified {clusterer.n_clusters} mood families using Circumplex Model!")
    print("   Based on Russell (1980): Valence and Arousal dimensions")
    print("="*70)