Category: Data Analysis and Visualization / Statistical Analysis Tools / Data Processing Pipelines

Motivation

When a music is described, some words like “genre” or “vibe” are usually used, but those terms are subjective and are often based on intuition. The goal of MOISE (Mood and Noise) is to build a reproducible analysis and modeling toolkit that treats songs as data objects. MOISE will try to:

1.	Classify the genre of a sung from its audio features
2.	Discover “mood families” by clustering songs based on their sonic profile
3.	Relate measurable and lyrical characteristics to flags like explicitness and popularity

The goal of the project is to figure out if we can systematically map and explain the emotional/sonic identity of a music using data instead of intuition.

Planned Approach and Technologies

This project will be implemented as modular Python 3.10+ codebase with a clear separation of concerns:

Dataset ingestion: (load and validate three datasets)

•	GTZAN features (features_30_sec.csv): Around 1000 audio clips with MFCCs (numbers that describe the timbre of a song/audio file), spectral descriptors, tempo, …, plus a label column (genre).
•	Spotify audio features (spotify_top_songs_audio_features.csv): artist, song title, danceability, energy, valence (does it sound happy or sad), loudness, tempo, explicit flag, popularity.
•	Millsong lyrics (spotify_millsongdata.csv): artist, song title, full lyrics text.

Feature engineering:

•	For audio: scale and prepare the numeric features for machine learning
•	For lyrics: create simple text features such as how positive/negative the language is, how personal it is (“I”, “me”, “we”, …) and how explicit it is.


Supervised Learning:

•	Train model to:
o	Predict musical genre from audio features  GTZAN
o	Predict explicit or high popularity from audio and lyric features  Spotify

Unsupervised learning:

•	Use k-means and PCA to group songs by sound and mood without using any labels.
•	Describe each group using averages like tempo, valence and energy.

Reporting:

•	Generate a Markdown report than includes model metrics (accuracy, precision, confusion matrix), visualizations of clusters in 2D and short interpretations.
•	Expose a command-line interface so the full pipeline can run end-to-end on a dataset.

Expected Challenges and Mitigation:

Data alignment: 
Lyrics and audio rows may not be perfectly matched by artist or title. I will implement fuzzy matching and log unmatched rows instead of failing.
Class imbalance: 
Some genre or labels may be rare or dominant. I will use stratified train/test splits and evaluate per-class recall, not just overall accuracy.
Interpretability of clusters: 
K-means does not explain itself. I will summarize each cluster by its average tempo, valence, danceability and lyric tone so that is becomes musically meaningful.

Success Criteria

•	The command line interface can run the full workflow without manual editing.
•	Genre classification on GTZAN performs clearly above random baselines and produces a readable confusion matrix.
•	Clustering produces at least three distinct mood/energy groups that can be described in an easy way.
•	A final Markdown report is generated with plots, metrics and interpretations.

Stretch Goals

•	Extract audio features (MFFCs, spectral descriptors, tempo, …) of a file in .wav/.mp3 format
•	Predict the danceability, valence, … of an input song
•	Produce “insight sentences” automatically (ex: “Explicit and high-danceability tracks dominate the popularity tier”).
