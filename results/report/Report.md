<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>MOISE: A Machine Learning Pipeline for Music Analysis and Mood Classification | Advanced Programming 2025</title>
    <meta name="description" content="Course materials and resources for Advanced Programming at UNIL">
    <link rel="stylesheet" href="/course-materials/assets/css/style.css">
    <link rel="canonical" href="https://ap-unil-2025.github.io/course-materials/assets/templates/project_report_template_md.txt">
    
    <!-- Favicons -->
    <link rel="icon" type="image/svg+xml" href="/course-materials/favicon.svg">
    <link rel="icon" type="image/svg+xml" sizes="32x32" href="/course-materials/favicon-32x32.svg">
    <link rel="apple-touch-icon" href="/course-materials/apple-touch-icon.svg">
    <link rel="mask-icon" href="/course-materials/favicon.svg" color="#003aff">
    <meta name="theme-color" content="#003aff">
    
    <!-- Citation Metadata -->
    <meta name="citation_title" content="MOISE: A Machine Learning Pipeline for Music Analysis and Mood Classification">
    <meta name="citation_author" content="Morellec, Paul">
    <meta name="citation_publication_date" content="2025">
    <meta name="citation_journal_title" content="HEC Lausanne Course Materials">
    <meta name="citation_public_url" content="https://ap-unil-2025.github.io/course-materials/assets/templates/project_report_template_md.txt">
    <meta name="citation_pdf_url" content="https://ap-unil-2025.github.io/course-materials/assets/course-materials.pdf">
    
    <!-- Dublin Core Metadata -->
    <meta name="DC.title" content="MOISE: A Machine Learning Pipeline for Music Analysis and Mood Classification">
    <meta name="DC.creator" content="Paul Morellec">
    <meta name="DC.subject" content="Data Science">
    <meta name="DC.subject" content="Python Programming">
    <meta name="DC.subject" content="Machine Learning">
    <meta name="DC.subject" content="Statistical Learning">
    <meta name="DC.description" content="Advanced course introducing Python programming, statistical learning, and high-performance computing for Master's students in Economics and Finance">
    <meta name="DC.publisher" content="HEC Lausanne, University of Lausanne">
    <meta name="DC.date" content="2025-12-26">
    <meta name="DC.type" content="Course Materials">
    <meta name="DC.format" content="text/html">
    <meta name="DC.identifier" content="https://ap-unil-2025.github.io/course-materials/assets/templates/project_report_template_md.txt">
    <meta name="DC.language" content="en">
    <meta name="DC.rights" content="Creative Commons Attribution-ShareAlike 4.0 International License">
    
    <!-- Schema.org structured data for Google Scholar -->
    <script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@type": "Course",
      "name": "Data Science and Advanced Programming 2025",
      "description": "Advanced course introducing Python programming, statistical learning, and high-performance computing",
      "provider": {
        "@type": "Organization",
        "name": "HEC Lausanne, University of Lausanne",
        "sameAs": "https://www.unil.ch/hec/"
      },
      "instructor": [
        {
          "@type": "Person",
          "name": "Simon Scheidegger",
          "url": "https://sites.google.com/site/simonscheidegger/"
        },
        {
          "@type": "Person",
          "name": "Anna Smirnova"
        }
      ],
      "courseCode": "DSAP2025",
      "hasCourseInstance": {
        "@type": "CourseInstance",
        "courseMode": "https://schema.org/OnlineOnly",
        "startDate": "2025-09-15",
        "endDate": "2025-12-15",
        "location": {
          "@type": "Place",
          "name": "Internef 263",
          "address": {
            "@type": "PostalAddress",
            "addressLocality": "Lausanne",
            "addressCountry": "CH"
          }
        }
      },
      "license": "https://creativecommons.org/licenses/by-sa/4.0/"
    }
    </script>
    
    <!-- Begin Jekyll SEO tag v2.8.0 -->
<title>MOISE: A Machine Learning Pipeline for Music Analysis and Mood Classification | Advanced Programming 2025</title>
<meta name="generator" content="Jekyll v4.3.4" />
<meta property="og:title" content="MOISE: A Machine Learning Pipeline for Music Analysis and Mood Classification" />
<meta name="author" content="Paul Morellec (paul.morellec@unil.ch)" />
<meta property="og:locale" content="en_US" />
<meta name="description" content="Course materials and resources for Advanced Programming at UNIL" />
<meta property="og:description" content="Course materials and resources for Advanced Programming at UNIL" />
<link rel="canonical" href="https://ap-unil-2025.github.io/course-materials/assets/templates/project_report_template_md.txt" />
<meta property="og:url" content="https://ap-unil-2025.github.io/course-materials/assets/templates/project_report_template_md.txt" />
<meta property="og:site_name" content="Advanced Programming 2025" />
<meta property="og:type" content="article" />
<meta property="article:published_time" content="2025-12-01T00:00:00+00:00" />
<meta name="twitter:card" content="summary" />
<meta property="twitter:title" content="MOISE: A Machine Learning Pipeline for Music Analysis and Mood Classification" />
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"BlogPosting","author":{"@type":"Person","name":"Paul Morellec (paul.morellec@unil.ch)"},"dateModified":"2025-12-01T00:00:00+00:00","datePublished":"2025-12-01T00:00:00+00:00","description":"Course materials and resources for Advanced Programming at UNIL","headline":"MOISE: A Machine Learning Pipeline for Music Analysis and Mood Classification","mainEntityOfPage":{"@type":"WebPage","@id":"https://ap-unil-2025.github.io/course-materials/assets/templates/project_report_template_md.txt"},"url":"https://ap-unil-2025.github.io/course-materials/assets/templates/project_report_template_md.txt"}</script>
<!-- End Jekyll SEO tag -->

</head>
<body>
    <header class="site-header">
        <div class="wrapper">
            <nav class="site-nav" role="navigation" aria-label="Main navigation">
    <div class="site-branding">
        <a class="site-title" href="/course-materials/" aria-label="Homepage">
            <img src="/course-materials/assets/images/unil.jpeg" alt="UNIL Logo" class="site-logo">
            <span class="site-title-text">DSAP</span>
        </a>
        <a href="https://nuvolos.cloud" target="_blank" rel="noopener noreferrer" class="powered-by-header" aria-label="Powered by Nuvolos">
            <span class="powered-text">Powered by</span>
            <img src="/course-materials/assets/images/nuvolos_logo.svg" alt="Nuvolos" class="header-nuvolos-logo">
        </a>
    </div>
    
    <button class="nav-toggle" aria-label="Toggle navigation menu" aria-expanded="false">
        <span class="hamburger"></span>
        <span class="hamburger"></span>
        <span class="hamburger"></span>
    </button>
    
    <div class="nav-links" id="nav-links">
        <a href="/course-materials/" >Home</a>
        <a href="/course-materials/syllabus" >Syllabus</a>
        <a href="/course-materials/weekly-materials" >Weekly Materials</a>
        <a href="/course-materials/assignments" >Assignments</a>
        <a href="/course-materials/exercises" >Exercises</a>
        <a href="/course-materials/projects" >Projects</a>
        <a href="/course-materials/help-support" >Help & Support</a>
        <a href="/course-materials/citation" >Cite</a>
        
        
    </div>
</nav>

<script>
document.addEventListener('DOMContentLoaded', function() {
    const navToggle = document.querySelector('.nav-toggle');
    const navLinks = document.querySelector('.nav-links');
    
    if (navToggle && navLinks) {
        navToggle.addEventListener('click', function() {
            const isExpanded = navToggle.getAttribute('aria-expanded') === 'true';
            navToggle.setAttribute('aria-expanded', !isExpanded);
            navLinks.classList.toggle('active');
            document.body.classList.toggle('nav-open');
        });
        
        // Close menu when clicking outside
        document.addEventListener('click', function(event) {
            if (!navToggle.contains(event.target) && !navLinks.contains(event.target)) {
                navToggle.setAttribute('aria-expanded', 'false');
                navLinks.classList.remove('active');
                document.body.classList.remove('nav-open');
            }
        });
        
        // Close menu on escape key
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                navToggle.setAttribute('aria-expanded', 'false');
                navLinks.classList.remove('active');
                document.body.classList.remove('nav-open');
            }
        });
    }
});
</script>

<style>
/* Make navigation more compact */
.site-nav {
    padding: 0.5rem 0;
}

.nav-links {
    gap: 0.75rem;
    font-size: 0.9rem;
}

.nav-links a {
    padding: 0.25rem 0.4rem;
}

.powered-by-header {
    gap: 0.3rem;
}

.powered-text {
    font-size: 0.75rem;
}

.header-nuvolos-logo {
    height: 20px;
}

.nav-toggle {
    display: none;
    background: none;
    border: none;
    cursor: pointer;
    padding: 0.5rem;
    flex-direction: column;
    gap: 0.25rem;
    z-index: 1001;
}

.hamburger {
    width: 24px;
    height: 2px;
    background-color: var(--text-primary);
    transition: all 0.3s ease;
    transform-origin: center;
}

.nav-toggle[aria-expanded="true"] .hamburger:nth-child(1) {
    transform: rotate(45deg) translate(6px, 6px);
}

.nav-toggle[aria-expanded="true"] .hamburger:nth-child(2) {
    opacity: 0;
}

.nav-toggle[aria-expanded="true"] .hamburger:nth-child(3) {
    transform: rotate(-45deg) translate(6px, -6px);
}

.nav-links a.active {
    color: var(--primary-color);
    font-weight: 600;
    background-color: rgba(59, 130, 246, 0.05);
}

/* Hide the underline pseudo-element completely */
.nav-links a::after {
    display: none !important;
}

.external-links {
    margin-left: 2rem;
    padding-left: 2rem;
    border-left: 1px solid var(--border-color);
}

.external-icon {
    font-size: 0.8em;
    opacity: 0.7;
    margin-left: 0.25rem;
}

@media (max-width: 768px) {
    .site-nav {
        position: relative;
    }
    
    .nav-toggle {
        display: flex;
    }
    
    .nav-links {
        position: absolute;
        top: 100%;
        left: 0;
        right: 0;
        background-color: var(--background-color);
        border: 1px solid var(--border-color);
        border-top: none;
        border-radius: 0 0 0.5rem 0.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        flex-direction: column;
        gap: 0;
        padding: 1rem 0;
        transform: translateY(-10px);
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
        z-index: 1000;
    }
    
    .nav-links.active {
        transform: translateY(0);
        opacity: 1;
        visibility: visible;
    }
    
    .nav-links a {
        padding: 0.75rem 1.5rem;
        display: block;
        color: var(--text-primary);
        border-bottom: 1px solid var(--border-color);
    }
    
    .nav-links a:last-child {
        border-bottom: none;
    }
    
    .nav-links a:hover {
        background-color: var(--surface-color);
    }
    
    .nav-links a::after {
        display: none;
    }
    
    .external-links {
        margin-left: 0;
        padding-left: 0;
        border-left: none;
        border-top: 1px solid var(--border-color);
        padding-top: 1rem;
        margin-top: 1rem;
    }
    
    body.nav-open {
        overflow: hidden;
    }
}
</style>
        </div>
    </header>

    <main class="page-content page-transition">
        <div class="wrapper page-wrapper">
            
            
            <article class="page">
    <header class="page-header">
        <h1 class="page-title">Your Project Title Here</h1>
        
        <p class="page-subtitle">Advanced Programming 2025 - Final Project Report</p>
        
    </header>

    <div class="page-content">
        # Abstract

This project implements a wide music analysis system using machine learning algorithms on three datasets, applied on four different modules: GTZAN for genre classification, Spotify Top Songs Audio Features for mood clustering and popularity prediction, and finally Million Song Dataset for lyrics analysis. The genre classification module uses supervised learning and achieves a 91.33% f1 score. The second module, lyrics analyzer, utilizes NLP, while the third, the mood clustering module, applies the Circumplex Model of Affect (Russel, 1980) with K-means (K=4) to discover four distinct mood quadrants based on valence and arousal dimensions. Quadrant 3 (Low Energy Positive music like classical or ambient) is absent from the mainstream Spotify dataset, while quadrant 1 (High Energy Negative) split into two distinct subclusters, suggesting nuanced emotional gradation within angry/tense music. Finally, the popularity prediction model utilizes regression model with R2 of 4.33%, suggesting that popularity is better predicted through non-audio features.

**Keywords:** data science, Python, machine learning, [add your keywords]

\newpage

# 1. Introduction

## 1.1 Background and Motivation

Music is a universal human expression that cuts over linguistic and cultural barriers. A significant amount of music data is now accessible for computer analysis because to the emergence of digital music platforms and streaming devices. Millions of people use services like Spotify every day, and the music industry produces billions of streams every year. Recommendation systems, playlist creation, copyright management, and even cultural studies can all benefit from an understanding of what makes a song popular, how genres can be automatically identified, what emotions lyrics transmit, and how songs can be grouped by mood. The richness, subjectivity, and multidimensionality of music make these concerns essentially difficult. Furthermore, human perceptions of music are fundamentally subjective and differ among people and cultures.

## 1.2 Objective and Goals

The goal of this project, MOISE (Mood and Noise), is to create a thorough modular machine learning pipeline for music analysis that tackles all the issues mentioned above. Implementing and comparing various supervised machine learning models for automatic genre classification is the first goal. The second objective of this study is to develop regression models that use only audio characteristics to predict song popularity while critically examining the inherent drawbacks of this methodology. Thirdly, a thorough NLP analysis of lyrics is conducted. Then, K-means clustering is used to identify four mood quadrants using the Circumplex Model of Affect (Russel, 1980), offering a theoretically grounded substitute for data-driver cluster selection. Creating a modular codebase that is ready for production while adhering to engineering best practices is another key goal. Then, implementing and comparing various supervised machine learning models for automatic genre classification is on another key objective of this project. Lastly, this study seeks to develop regression models that use solely audio characteristics to predict song popularity while critically examining the inherent drawbacks of this methodology.

# 2. Literature Review

## 2.1 Music Information Retrieval (MIR)

Music Information Retrieval (MIR) is an interdisciplinary research field that combines signal processing, machine learning, musicology, and psychology to extract meaningful information from music [^3]. This field has grown significantly since the early 2000s, driven by the digitalization of music libraries and the availability of large-scale datasets.

## 2.2 Genre Classification

Automatic music genre classification has been a fundamental problem in MIR since the late 1990s. The GTZAN dataset, created by Tzanetakis and Cook (2002)[^13], has become a standard benchmark despite some known issues with mislabeling and audio quality[^12]. Classical machine learning algorithms (SVM, k-NN, Random Forest) have shown strong performance on genre classification, achieving 80-90% accuracy on GTZAN[^1]. More recently, deep learning methods using Convolutional Neural Network (CNNs) on spectrograms have pushed accuracy above 90%[^2], but they require significantly more computational resources and training data.

## 2.3 Popularity Prediction

Predicting song popularity is a particularly challenging task that has gained attention with the rise of streaming platforms. Unlike genre classification which depends primarily on audio content, popularity is influenced by many different factors[^9]. While audio features such as tempo, energy, and danceability contribute to a song’s charm, external factors often play a more important role. These include marketing, promotion campaigns by record labels, the existing popularity and fan base of the artist, placement on playlists, and many more. Research has consistently shown that audio features alone provide low predictive power for popularity with R² typically lower than 10%[^6]. This finding demonstrates that non-audio factors play the dominant in determining the popularity of a song.

## 2.4 Sentiment Analysis in Lyrics

Natural Language Processing (NLP) techniques have been successfully applied to music lyrics for sentiment analysis, theme extraction and style analysis[^5]. Some of the key approaches include Bag-of-words models (simple but effective for theme detection) sentiment lexicons such as VADER, TextBlob for polarity (positive or negative) and subjectivity, word embeddings (Word2Vec, Glove for semantic analysis), and topic modeling (LDA for discovering latent themes). Studies have discovered correlations between the sentiment of lyrics and various musical characteristics, although this relationship varies in complexity and depends on the genre[^8]. Typically, the lyrics enhance the emotional messages expressed with the musical elements; however, there are instances where artists intentionally create contrasts, such as cheerful music combined with melancholic lyrics.

## 2.5 The Circumplex Model of Affect

The Circumplex Model of Affect (Russell, 1980)[^11] is a foundational theory in affective psychology that represents emotions in a two-dimensional space. This model has been extensively applied to music emotion recognition (Eerola & Vuoskoski, 2011; Kim et al., 2010)[^4,^7]. The model theorizes two orthogonal dimensions: valence, representing the pleasure-displeasure continuum, from negative to positive affect, and arousal, representing the activation-deactivation continuum from low to high energy[^10].
Eerola and Vuoskoski[^4] demonstrated that dimensional models like the Circumplex often outperform discrete emotion categories in capturing the nuanced affective qualities of music. The model’s two dimensions align naturally with acoustic features, since the valence correlates with harmonic content, mode (major, minor, …), and timbral brightness, while the arousal correlates with tempo, loudness, and rhythmic complexity. Spotify’s audio features, particularly energy and valence, where explicitly designed to capture these dimensions, making them ideal for implementing the Circumplex Model in computational music analysis. 

# 3. Methodology

## 3.1 Data Description

Describe your dataset(s):

The first dataset used in this project is the GTZAN Genre Collection. It is a widely used benchmark dataset for music classification research[^13]. This version contains approximately 10,000 audio samples (3-second clips), 10 music genres, as well as multiple features. The dataset has balanced classes with approximately 1000 samples per genre, facilitating fair-evaluation across categories. 
![Figure 1](/results/figures/01_gtzan_genre_distribution.png)
*Figure 1: Distribution of music genres in the GTZAN dataset*
The second dataset is the Spotify Top Song Audio Features dataset, which contains audio features for popular songs extracted via the Spotify Web API. It contains more than 6500 songs and contains audio metrics [0; 1] such as energy, valence, danceability, speechiness, acousticness, and liveness. It also contains information about categorical features such as the key, mode and time signature, and metadata such as the artist, the track name and duration. The target variable for popularity prediction is the stream count. Figure 2 displays the distribution of key audio features, showing that popular songs tend toward high danceability and energy values while exhibiting relatively low speechiness and instrumentalness. 
![Figure 2](/results/figures/02_spotify_audio_features.png)
*Figure 2: Distribution of Spotify audio features across the dataset*
The third and last dataset used in this project is the Million Song Dataset. It provides lyrics text, artist and song title for more than 57,000 entries. The songs are spread across multiple genres and decades. Finally, the text is encoded in UTF-8, which is suitable for NLP analysis. Figure 3 shows the top artists by song count, displaying the range of artists represented in the corpus. 
![Figure 3](/results/figures/03_lyrics_top_artists.png)
*Figure 3: Top 15 artists by number of songs in the lyrics dataset.*
## 3.2 Approach

### 3.2.1 Preprocessing

#### Gtzan Preprocessing
The preprocessing pipeline for the GTZAN dataset includes features selection, where non-feature columns are removed, such as the filename and the length. Missing values are handled by replacing them by the median for all numerical features, ensuring robustness against outliers. Genres are converted into integer labels from 0 to 9. All features are normalized using StandardScaler to zero mean and unit variance, which is essential for algorithms sensitive to feature scales such as SVM and neural networks. Finally, the data is split into train-test split (stratified 80/20 split preserving class distribution).

#### Spotify Preprocessing
Spotify Top Song Audio Feature’s data processing involves additional feature engineering on top of basic cleaning. First, rows with missing stream count values are removed to ensure valid target labels. Then, categorical variables including key, mode, and time signature are encoded numerically. The target variable (stream counts) is logarithmically transformed to reduce the severe right-skewness, which usual for popularity distributions, and improve model performance. All features are scaled using StandardScaler. Lastly, the data is split into training and test sets using an 80/20 ratio.

#### Lyrics Preprocessing
Finally, to process the raw lyrics data for NLP analysis, text is first converted to lowercase to ensure consistent matching. Then, numbers and URLS are removed since they typically do not contribute to semantic meaning. Only word characters, spaces, apostrophes, and hyphens are kept while removing any other punctuation and special characters. Whitespace is normalized by merging multiple spaces into single spaces. Very short texts containing less than 51 characters are filtered out as they often represent incomplete or corrupted entries. Finally, duplicated lyrics (based on the cleaned text) are removed to prevent bias in the analysis.

### 3.2.2 Algorithms, Models and Evaluation

#### Genre classification Models
The genre classification module tests four different machine learning algorithms and compares them one to another. The best model is chosen by combining test accuracy and 5-fold cross-validation. The first, Random Forest, uses 300 decision trees with a maximum depth of 20. The trees are trained on different random parts of the data to reduce overfitting while still having strong prediction power. The Support Vector Machine uses an RBF (Radical Basic Function) kernel with regularization parameter C = 10. This allows it the model control how much it tries to fit the training data verses keeping the boundary simple. The Gradient Boosting classifier uses 100 sequential estimators with a learning rate of 0.1, making this way small improvements at with each new model. The Multi-Layer Perceptron consists of two hidden layers with 100 and 50 neurons respectively, using early stopping to prevent overfitting. Model selection is based on a combined score incorporating both test set accuracy and 5-fold cross-validation accuracy to ensure generalization.

#### Popularity Prediction Models
To predict popularity, five different regression models were compared. The best model is chosen by combining the R² and the 5-fold cross-validation score. The first is a simple linear regression that serves as a baseline model to provide interpretable coefficients for each of the feature. Then, a Ridge Regression extends the linear regression by adding a “penalty” with L2 regularization (α = 1.0) to prevent the model from relying on one specific feature. A Lasso Regression is also performed let the model drop less useful features.  Finally, a Random Forest Regression employs 100 decision trees to capture non-linear relationships without explicit feature engineering.

#### Mood Clustering
The mood clustering module uses K-means clustering with K = 4 clusters. This choice is motivated by the Circumplex Model rather than purely data-driven methods like the elbow technique. The clustering uses seven audio features: danceability, energy, valence, tempo, loudness, acousticness, and instrumentalness. The algorithm is configured with 10 random initializations to ensure stability and convergence to a good solution. After clustering, the clusters are reordered to align with the four theoretical quadrants of the Circumplex Model based on their positions in the valence-arousal space.

#### Lyrics Analysis
The NLP analysis pipeline employs TextBlob for computing sentiment polarity (ranging from -1 for negative to +1 for positive) and subjectivity scores (ranging from 0 for objective to 1 for subjective). Theme detection uses keyword-based matching for eight predefined themes: Love, Life, Party, Sadness, Dreams, Freedom, Happiness, and Money. The personal pronoun analysis computes the ratio of personal pronouns (I, me, my, we, us, our, you, your) to total words, categorizing songs into high (> 15%), medium (8 -15%), and low (< 8%) intimacy levels.

### 3.2.3 Complexity Analysis
The different technics used in this project have different training complexity. For example, Random Forests’training complexity is O(n·m·log n·T) where n is the number of samples, m is the number of features, and T is the number of trees, while prediction requires O(T·log n) per sample. SVM with RBF kernel has training complexity from O(n²·m) to O(n³) depending on the implementation with prediction complexity of O(n_sv·m) where n_sv is the number of support vectors. Gradient Boosting requires O(n·m·T) for training and O(T) for prediction. K-means clustering has complexity O(n·K·I·m) where K is the number of clusters and I is the number of iterations, with prediction requiring only O(K·M) distance computations.


## 3.3 Implementation

### 3.3.1 Code Architecture

The MOISE project’s pipeline is organized into 9 Python modules, who each have their specific responsability. The data ingestion and validation are carried out with the `data_loader.py` module. This allows for consistent interfaces for loading the three datasets. The `preprocessing.py` module implements all data cleaning and feature engineering pipelines. Then every different analysis such as genre classification or NLP analysis is distributed in four different modules: `genre_calssifier.py` for supervised genre classification, `popularity_predictor.py` for regression-based popularity prediction, `lyrics_analyzer.py` for NLP-based lyrics analysis, and `mood_clustering.py` for Circumplex-based mood clustering. The `eda.py` module provides more information about the data used in this project through visualization. Finally, `cli.py` implements a command-line interface that runs through the entire pipeline.
The implementation uses several software engineering best practices to ensure that the quality and maintainability of the code. Each module defines a main class (such as GenreClassifier or MoodClusterer) that runs related functionality with consistent interfaces. All functions include comprehensive type annotations for improved code readability and IDE support:

```python
def load_gtzan_data(
    self, 
    filename: str = 'features_3_sec.csv'
    ) -> pd.DataFrame:
    """
    Load GTZAN genre classification dataset.
    
    Arguments:
        filename (str): Name of the GTZAN CSV file
        
    Returns:
        pd.DataFrame: Loaded GTZAN data

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If expected columns are missing
        RuntimeError: If any other error occurs during loading
    """
    filepath = self.data_directory / filename
```
Errors are handled throughout the modules with informative error messages that help the user find and resolve the occurring issues quickly.  All public methods include detailed docstrings following the NumPy/Google style conventions, documenting parameters, return values, and usage examples.
To improve performance, the implementation executes independent computations simultaneously across multiple processor cores when the task allows it. For example, Random Forest training utilizes the n_jobs = -1 parameter to allow all decision trees to be built at the same time using every available processor core, rather than one after another. Similarly, cross-validation, which tests the model on multiple data splits, runs these tests in parallel, considerably reducing the total computation time. For tree-based models, the calculation of feature importance scores also benefits from this parallel approach, making the analysis faster overall.
A command-line interface enables end-to-end pipeline execution with customizable parameters. Users can run all analysis with a single command (`python src/cli.py -all`) or execute specific modules individually. Custom parameters allow the user to specify the data directory and output directory.
The mood clustering module integrates the Circumplex Model through nine steps: loading the Spotify dataset, selecting mood-related features, initializing the clusterer with K=4, visualizing the theoretical Circumplex Model, fitting K-means clusters with automatic quadrant reordering, describing clusters with feature averages, naming clusters based on psychological quadrants, generating 2D PCA visualizations, and creating cluster profile comparisons. This approach ensures reproducibility and provides detailed progress feedback through console output.
Each module includes test code in the if `__name__== “__main__”` block, enabling testing and validation of individual module. Thes testing approach validates data loading and preprocessing transformations, model training and prediction pipelines, visualization generation, and error handling for edge cases such as missing data or invalid inputs.
Trained models are saved to disk using joblib for efficient storage and subsequent loading without needing retraining. The serialization includes not only the model object but also associated metadata such as the label encoder, feature names, and evaluation results, enabling complete reproducibility of predictions.

### 3.3.2 Dependencies management
Project dependencies are handled through a requirements.txt file that specifies minimum version requirements for all external packages. Core dependencies include pandas (≥1.3.0) and numpy (≥1.21.0) for data manipulation, scikit-learn (≥1.0.0) for machine learning algorithms, matplotlib (≥3.4.0) and seaborn (≥0.11.0) for visualization, and nltk (≥3.6.0) with textblob (≥0.17.0) for natural language processing. The joblib package (≥1.1.0) handles model serialization and parallel processing.

# 4. Results

## 4.1 Genre Classification
Figure 4 compares the performance of four classification algorithms for the genre classification module. The SVM classifier achieves the best overall performance with 91.34% test accuracy, followed by the Gradient Boosting at 86.94%, Random Forest at 86.39% and MLP at 84.98%. The cross-validation results on the right panel confirm the stability of these rankings, with SVM achieving the highest mean CV accuracy of 90.58%.
![Figure 4: Model Comparison](/results/figures/04_model_comparison.png)
*Figure 4: Comparison of genre classification models showing train/test accuracy (left) and cross-validation performance with error bars (right).*

The confusion matrix for the SVM classifier (figure 5) depicts the classification patterns across all ten genres. The strong diagonal shows good overall classification performance, with most genre achieving precision above 85%. Classical music achieves the highest precision above 97.5%, benefiting from its distinctive acoustic characteristics. Notable confusion patterns emerge between acoustically similar genre, for example country and rock share 8 misclassifications probably due to their common use of guitar-driven arrangements, pop and disco show 8 misclassifications reflecting their dance-oriented rhythmic patterns, and jazz and classical show 9 cross-classifications due to their shared emphasis on acoustic instrumentation and complex harmonic structures.
![Figure 5: Confusion Matrix](/results/figures/05_confusion_matrix.png)
*Figure 5: Confusion matrix for the SVM genre classifier showing predictions across 10 genres.*

## 4.2 Popularity Prediction
The popularity prediction results, displayed in Figure 6, reveal that all models only manage to explain a low percentage of the popularity through audio features, represented by R2 scores. Random Forest performs best with an R2 of 4.33%, followed by the Linear Regression and Ridge at 1.75%, Gradient Boosting at 1.75%, and Lasso at 1.23%. The Mean Absolute Error remains relatively consistent across all models at approximately 1.27-1.30 log-stream units, indicating similar absolute prediction accuracy despite differences in variance explained.
![Figure 6: Popularity Model Comparison](/results/figures/10_popularity_model_comparison.png)
*Figure 6: Comparison of popularity prediction models by R² score (left) and MAE (right).*

Figure 7 plots the Random Forest Regression. The wide scatter around the perfect prediction line (in red) confirms the limited predictive power of audio features for popularity. The residual plot on the right shows no systematic bias but considerable variance, validating that the model captures little of the true variation in popularity.
![Figure 7: Predictions](/results/figures/11_popularity_predictions.png)
*Figure 7: Actual vs. predicted popularity (left) and residual distribution (right) for Random Forest.*

The feature importance analysis in Figure 8 shows that speechiness, tempo, and loudness are the most influential audio features for popularity prediction, though their individual contributions remain modest. This suggests that while certain acoustic characteristics may correlate weakly with popularity, the dominant factors lie outside of the audio domain.
![Figure 8: Feature Importance](/results/figures/12_feature_importance.png)
*Figure 8: Top 15 features for popularity prediction ranked by Random Forest importance.*

## 4.3 Lyrics Analysis
The sentiment analysis results, presented in Figure 9, show the emotional tone distribution across the analyzed cleaned lyrics dataset of 57,300 song. Approximately 51.2% songs display positive sentiment (29,364 songs), 39.1% are classified as neutral (22,431 songs), and only 9.6% show negative sentiment (5,5005 songs). This predominance of positive content indicates that an overall positive leaning in popular music, though it may also reflect the commercial preference for uplifting music. The average polarity score of 0.111 confirms this slight positive bias, while the average subjectivity of 0.505 indicates that most lyrics fall in the moderately subjective range.
![Figure 9: Sentiment Distribution](/results/figures/06_sentiment_distribution.png)
*Figure 9: Sentiment distribution showing categorical breakdown (left), polarity histogram (center), and subjectivity histogram (right).*
The word frequency analysis in Figure 10 shows the lexical patterns in the lyrics corpus after stop word removal. The word “love” dominates with 91,726 occurrences, followed by “know” (72,224), “like” (63,308), and “got” (50,070). This shows that some of the most predominant themes in popular music are love and relationship.
![Figure 10: Word Frequencies](/results/figures/07_word_frequencies.png)
*Figure 10: Top 20 most frequent words in lyrics after stopword removal.*
The theme analysis in Figure 11 confirms this finding, showing that Life (75.7%, 43,382 songs) and Love (66.6%, 38168 songs) are the most common themes. Dreams (44.9%), Party (40.0%), and Sadness (35.7%) form a middle tier, while freedom (30.6%), Happiness (19.9%), and Money (12.2%) appear less frequently. 
![Figure 11: Song Themes](/results/figures/08_song_themes.png)
*Figure 11: Common themes identified in song lyrics through keyword-based analysis.*
The personal pronoun analysis in Figure 12 plots the intimacy level of lyrics through pronoun usage. The mean personal pronoun ratio of 11.6% indicates that popular songs typically have moderate intimacy levels, which might indicate that on average popular songs try to feel somewhat personal and direct, addressing the listeners, while still avoiding being too self-focused.

## 6.4 Mood Clustering
Figure 13 shows the application of the Circumplex Model to the Spotify dataset. Songs are plotted according to their valence (x-axis) and energy/arousal (y-axis) values, with colors indicating the theoretical quadrants assignment. The four quadrants appearing clearly: High Energy Positive in the upper right (n = 2,813 songs, 43.2%, representing happy, excited music), High Energy Negative in the upper left (n = 2,573 songs, 39.5%, representing angry, tense music), Low Energy Negative in the lower left (n = 1,035 songs, 15.9%, representing sad, melancholic music), and Low Energy Positive in the lower right (n = 92 songs, 1.4%, representing calm, peaceful music). The uneven distribution across quadrants shows the strong bias in popular music toward high-energy content.
![Figure 13: Circumplex Model](/results/figures/15_circumplex_model.png)
*Figure 13: Circumplex Model of Affect applied to music mood classification, showing four emotional quadrants with song distributions.*
Figure 14 presents the K-means clustering results projected in 2D PCA space. 51.2% of the total variance can be explained by the two principal components (35.1% and 16.1% respectively). The clustering achieves a Silhouette Score of 0.18, which is expected for continuous mood data where emotional states naturally overlap rather than forming discrete categories. While the clusters overlap, reflecting the continuous nature of emotional expression in music, the cluster centers (marked with X) are well-separated, indicating that the four-cluster solution captures significant structure in the data.
![Figure 14: Mood Clusters 2D](/results/figures/13_mood_clusters_2d.png)
*Figure 14: Mood clusters visualized in 2D PCA space with cluster centers marked.*
Figure 15 shows the average audio characteristics of each of the different mood clusters. High Energy Positive songs (cluster 0, 2,813 songs) exhibit high values across danceability (0.748), energy (0.729), valence (0.685), and low acousticness (0.187), consistent with upbeat, feel-good music. High Energy Negative songs (Cluster 1, 2,573 songs) share the high energy (0.632) but have much lower valence (0.337), encapsulating aggressive or tense music. Low Energy Negative/Acoustic songs (Cluster 2, 1,035 songs) are characterized by high acousticness (0.638) and low values for energy (0.403) and loudness (-9.6 dB), representing melancholic ballads. Low Energy Positive songs (Cluster 3, 92 songs) show high instrumentalness (0.576) with moderate acousticness, portraying ambient and peaceful instrumental music.
![Figure 15: Cluster Profiles](/results/figures/14_cluster_profiles.png)
*Figure 15: Feature profiles for each mood cluster showing normalized feature values.*

# 5. Conclusion

## 5.1 Summary of Findings
This project successfully has developed and evaluated a comprehensive machine learning pipeline for music analysis producing several significant findings. The genre classification module demonstrates that SVM with RDF kernel achieves 91.34% accuracy on the GTZAN benchmark, confirming that hand-crafted audio features remain effective for genre classification even as deep learning methods gain notoriety. Cross-validation results confirm model stability with low variance across folds (CV accuracy of 90.58%), suggesting that the model can be extrapolated to new data.
The popularity prediction experiments yielded consistently R2 values below 5% across all models, giving empirical confirmation that audio features alone are not sufficient to predict song popularity. This finding has important implications for music recommendation systems and the music industry, showing that for a song to have commercial success, marketing investment, artist reputation, playlist placement, and social media usage is more important than focusing solely on audio features.
The lyrics analysis shows that popular music is mostly positive (51.2%), with Life (75.7%) and Love (66.6) as main themes. The personal pronoun analysis gives a new quantitative metric for quantifying lyrical intimacy, revealing that popular songs usually maintain moderate pronoun usage (11.6%).
The mood clustering experimentations validates the use of the Circumplex Model to create mood clusters. The four-cluster choice aligns well with psychological theory and produces interpretable and musically meaningful clusters, which uncover coherent audio characteristics for each quadrant, allowing applications in mood-based music recommendation and playlist generation.

## 5.2 Contributions
This project makes several contributions to the field of MIR. The first to provide a pipeline for computer assisted music analysis with comprehensive documentation that can serve as a base for future research and application. The second is to provide empiric results that audio features by themselves are not sufficient to predict a song’s future popularity, contributing to the ongoing debate on how to secure a song’s commercial success. Thirdly, it shows that the Circumplex Model of Affect can be applicable to mood clustering, proving that it has theoretical grounding for cluster selection. Fourthly, this project introduces a new metric which provides personal pronoun use for analyzing a song’s lyrical intimacy. Finally, it delivers an end-to-end command-line interface which facilitates replication and extension of this work.

## 5.3 Limitations and Future Work
This project has many limitations. The first one being that the genre classification module uses pre-extracted features rather than learning directly from raw audio. To resolve this issue, future work could explore CNN or Transformer architectures directly applied to spectrograms or even waveforms. Second, the popularity prediction module could be improved by adding temporal features such as release timing or social media activity. Another limitation of this project is the fixed 10-genre taxonomy, which treats genres as mutually exclusive, when a song can have many genres. Adding more genres could make the predictions less accurate since genres tend to overlap. Moreover, since the dataset dates from 2002 it doesn’t contain new genres such as EDM subgenres (dubstep, trap, hardstyle), reggaeton, or K-pop and makes the application of this project difficult to recent music.

# References

1. Bergstra, J., Casagrande, N., Erhan, D., Eck, D., & Kégl, B. (2006). Aggregate features and AdaBoost for music classification. *Machine Learning*, 65(2-3), 473-484.

2. Costa, Y. M. G., Oliveira, L. S., & Silla Jr, C. N. (2017). An evaluation of convolutional neural networks for music classification. *Applied Soft Computing*, 52, 28-38.

3. Downie, J. S. (2003). Music information retrieval. *Annual Review of Information Science and Technology*, 37(1), 295-340.

4. Eerola, T., & Vuoskoski, J. K. (2011). A comparison of the discrete and dimensional models of emotion in music. *Psychology of Music*, 39(1), 18-49.

5. Hu, X., Downie, J. S., & Ehmann, A. F. (2009). Lyric text mining in music mood classification. *Proceedings of ISMIR*.

6. Interiano, M., Kazemi, K., Wang, L., Yang, J., Yu, Z., & Komarova, N. L. (2018). Musical trends and predictability of success in contemporary songs. *Royal Society Open Science*, 5(5).

7. Kim, Y. E., Schmidt, E. M., Migneco, R., Morton, B. G., Richardson, P., Scott, J., Speck, J. A., & Turnbull, D. (2010). Music emotion recognition: A state of the art review. *Proceedings of ISMIR*, 255-266.

8. Malheiro, R., Panda, R., Gomes, P., & Paiva, R. P. (2018). Emotionally-relevant features for classification and regression of music lyrics. *IEEE Transactions on Affective Computing*, 9(2), 240-254.

9. Pachet, F., & Roy, P. (2012). Hit song science is not yet a science. *Proceedings of ISMIR*.

10. Posner, J., Russell, J. A., & Peterson, B. S. (2005). The circumplex model of affect: An integrative approach. *Development and Psychopathology*, 17(3), 715-734.

11. Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161-1178.

12. Sturm, B. L. (2013). The GTZAN dataset: Its contents, its faults, their effects on evaluation. *ACM Transactions on Intelligent Systems and Technology*, 5(1).

13. Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. *IEEE Transactions on Speech and Audio Processing*, 10(5), 293-302.
# Appendices

## Appendix A: Helper Tools
During this project, several tools and resources were utilized. Claude was used for: model proposal, Silhouette score implementation and interpretation, graphics drafting, bug fixes, optimization suggestions, cli.py codding (50-60%), writting assistance, and module building assistance. All generated code was carefully reviewed, understood, and modified to feat the project's needs. Chat GPT was used for optimization suggestions and writing support. Caffeine, as a hardware helper tool, was intensively utilized for project support. All design decisions were made independently, and AI was used to accelerate developpment, not to replace comprehension.


## Appendix B: Code Repository

**GitHub Repository:** https://github.com/yourusername/project-repo

### Repository Structure
```
project-MOISE/
├── PROPOSAL.md
├── requirements.txt
├── main.py
├── data/
│   ├── features_3_sec.csv
│   ├── spotify_top_songs_audio_features.csv
│   └── spotify_millsongdata.csv
├── src/
│   ├── __init__.py.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── genre_classifier.py
│   ├── popularity_predictor.py
│   ├── lyrics_analyzer.py
│   ├── mood_clustering.py
│   ├── eda.py
│   └── cli.py
└── results/
    ├── figures/
    └── models/
    └── report/
```
    </div>
</article>
        </div>
    </main>

    <footer class="site-footer">
        <div class="wrapper">
            <p class="license">
                <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">
                    <img alt="Creative Commons License" style="border-width:0; height: 31px; width: auto; display: block; margin: 0 auto 0.5rem auto;" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" />
                </a>
                This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
            </p>
            <p class="citation">
                <strong>Cite as:</strong> Scheidegger, S., & Smirnova, A. (2025). <em>Data Science and Advanced Programming 2025</em>. HEC Lausanne, University of Lausanne. 
                <a href="/course-materials/citation">View citation formats →</a>
            </p>
            <p class="credits">Made with 💙 by Anna Smirnova, Prof. Simon Scheidegger, and Claude 🤖</p>
            <p class="powered-by">
                <a href="https://nuvolos.cloud" target="_blank" style="display: inline-flex; align-items: center; gap: 0.5rem; color: inherit; text-decoration: none;">
                    Powered by
                    <img src="/course-materials/assets/images/nuvolos_logo.svg" alt="Nuvolos" style="height: 20px; width: auto; opacity: 0.8;">
                </a>
            </p>
        </div>
    </footer>

    <!-- Smooth Page Transitions -->
    <style>
    /* No CSS transitions - just AJAX navigation */
    
    /* Smooth scrolling for same-page links */
    html {
        scroll-behavior: smooth;
    }
    
    /* Removed underline animation - clean hover effect only */
    </style>

    <script>
    // Smooth SPA-style navigation
    document.addEventListener('DOMContentLoaded', function() {
        const pageContent = document.querySelector('.page-wrapper');
        const navLinks = document.querySelectorAll('.nav-links a');
        
        // Handle navigation clicks with AJAX
        navLinks.forEach(link => {
            // Skip external links
            if (link.hostname !== window.location.hostname) {
                return;
            }
            
            link.addEventListener('click', function(e) {
                e.preventDefault();
                
                const url = this.href;
                const currentPath = window.location.pathname;
                
                // Skip if same page
                if (url === window.location.href) {
                    return;
                }
                
                // Fetch new page content
                fetch(url)
                    .then(response => response.text())
                    .then(html => {
                        // Parse the new page
                        const parser = new DOMParser();
                        const newDoc = parser.parseFromString(html, 'text/html');
                        const newContent = newDoc.querySelector('.page-wrapper');
                        const newTitle = newDoc.querySelector('title');
                        
                        // Update content instantly
                        if (newContent) {
                            pageContent.innerHTML = newContent.innerHTML;
                            
                            // Re-execute any script tags in the new content
                            const scripts = pageContent.querySelectorAll('script');
                            scripts.forEach(script => {
                                const newScript = document.createElement('script');
                                if (script.src) {
                                    newScript.src = script.src;
                                } else {
                                    newScript.textContent = script.textContent;
                                }
                                script.parentNode.replaceChild(newScript, script);
                            });
                        }
                        if (newTitle) {
                            document.title = newTitle.textContent;
                        }
                        
                        // Update URL without reload
                        history.pushState({}, '', url);
                        
                        // Update active navigation
                        updateActiveNav(url);
                        
                        // Scroll to top
                        window.scrollTo({ top: 0, behavior: 'smooth' });
                    })
                    .catch(error => {
                        console.log('AJAX failed, falling back to normal navigation');
                        window.location.href = url;
                    });
            });
        });
        
        // Handle browser back/forward
        window.addEventListener('popstate', function() {
            window.location.reload();
        });
        
        // Update active navigation state
        function updateActiveNav(currentUrl) {
            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.href === currentUrl || 
                    (currentUrl.includes('/week/') && link.href.includes('/weekly-materials'))) {
                    link.classList.add('active');
                }
            });
        }
        
        // Initial active state
        updateActiveNav(window.location.href);
    });
    </script>
</body>
</html>