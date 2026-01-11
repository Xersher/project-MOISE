# AI Tools Usage Documentation

**Project**: MOISE (Mood and Noise)  
**Author**: Paul Morellec  
**Course**: Advanced Programming 2025  
**Date**: January 2026

---

## Overview

This document describes how AI tools were used during the development of the MOISE project. All AI-generated code was carefully reviewed, understood, and modified to fit the project's needs. Design decisions were made independently, and AI was used to **accelerate development, not to replace understanding**.

---

## AI Tools Used

| Tool | Provider | Primary Use |
|------|----------|-------------|
| **Claude** | Anthropic | Implementation assistance, debugging, documentation |
| **ChatGPT** | OpenAI | Optimization suggestions, writing support |
| **GitHub Copilot** | GitHub/Microsoft | Inline code autocompletion |

---

## Detailed Usage by Tool

### 1. Claude (Anthropic)

Claude was the primary AI assistant used throughout the project.

#### ✅ Use Cases

| Category | Description | Example |
|----------|-------------|---------|
| **Model Proposal** | Helped structure the initial project architecture and module design | Suggested separation of concerns between data loading, preprocessing, and analysis modules |
| **Implementation Assistance** | Assisted with implementing specific algorithms and methods | Silhouette score calculation and interpretation for K-means clustering |
| **Debugging** | Identified and fixed bugs in code | Fixed undefined variable errors, missing imports, type annotation issues |
| **Optimization** | Suggested performance improvements | Added `n_jobs=-1` for parallel processing in scikit-learn models |
| **Documentation** | Helped write docstrings and README | NumPy-style docstrings, README structure and content |
| **Code Review** | Reviewed modules for best practices compliance | Identified missing type hints, inconsistent return types, docstring format issues |
| **CLI Development** | Assisted with command-line interface implementation | `cli.py` module (approximately 50-60% AI-assisted) |

#### 📝 Specific Examples

**Example 1: Debugging — Missing Import**
```python
# Bug identified by Claude in main.py
# EDAanalyser was used but never imported

# Before (bug):
eda = EDAanalyser(output_directory=str(figures_dir))  # NameError!

# After (fixed):
from eda import EDAanalyser
```

**Example 2: Optimization — Parallel Processing**
```python
# Claude suggested adding n_jobs=-1 for multicore utilization

# Before:
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

# After:
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
```

**Example 3: Documentation — NumPy-style Docstrings**
```python
# Claude helped convert docstrings to NumPy format

# Before:
def load_data(self, filename):
    """Load data from file.
    
    Args:
        filename: Name of file to load
    """

# After:
def load_data(self, filename: str) -> pd.DataFrame:
    """
    Load data from file.

    Parameters
    ----------
    filename : str
        Name of the file to load

    Returns
    -------
    pd.DataFrame
        Loaded data as a pandas DataFrame
    """
```

---

### 2. ChatGPT (OpenAI)

ChatGPT was used as a secondary tool for specific tasks.

#### ✅ Use Cases

| Category | Description |
|----------|-------------|
| **Optimization Suggestions** | Alternative approaches to improve code efficiency |
| **Writing Support** | Grammar and clarity improvements for documentation and report |

---

### 3. GitHub Copilot

Copilot was used passively during coding sessions.

#### ✅ Use Cases

| Category | Description |
|----------|-------------|
| **Autocompletion** | Inline code suggestions while typing |
| **Boilerplate Code** | Repetitive patterns like try/except blocks, file I/O |

#### 📝 Example

```python
# Copilot suggested completing repetitive patterns like:
if save_figs:
    plt.savefig(self.output_directory / 'figure_name.png', dpi=300, bbox_inches='tight')
if show:
    plt.show()
else:
    plt.close()
```

---

## Code Ownership Statement

### What I Wrote Independently

- **Project architecture and design decisions**: Module structure, class hierarchies, data flow
- **Circumplex Model implementation**: Decision to use K=4 based on Russell (1980) theory
- **Feature engineering**: Selection of audio features for each analysis module
- **Evaluation methodology**: Metrics selection, cross-validation strategy
- **Core logic in all modules** (data_loader, preprocessing, genre_classifier, popularity_predictor, lyrics_analyzer, mood_clustering, eda)

### What AI Assisted With

- **cli.py**: ~50-60% AI-assisted (command-line argument parsing, help text)
- **Docstrings**: AI helped format to NumPy style
- **README.md**: AI helped structure and write sections
- **Bug fixes**: AI identified several bugs that I then understood and fixed
- **Type annotations**: AI suggested adding type hints throughout

### Verification Process

For all AI-generated code, I followed this process:

1. **Read and understand** every line of suggested code
2. **Test** the code to ensure it works correctly
3. **Modify** as needed to fit the project's specific requirements
4. **Integrate** only after full comprehension

---

## Compliance with Course Policy

| Policy Requirement | Status | Evidence |
|--------------------|--------|----------|
| ✅ AI used for debugging help | Compliant | Bug fixes documented above |
| ✅ AI used for learning new libraries | Compliant | Learned joblib, TextBlob patterns |
| ✅ AI used for code review suggestions | Compliant | Type hints, docstring format |
| ✅ AI used for documentation writing | Compliant | README, docstrings |
| ❌ Code I don't understand | **Not Done** | All code reviewed and understood |
| ❌ AI wrote entire project | **Not Done** | Core logic written independently |

---

## Summary

AI tools were valuable **learning accelerators** in this project. They helped me:

- Write cleaner and create more consistent code
- Follow Python best practices (type hints, docstrings)
- Debug faster and learn from mistakes
- Produce professional documentation

However, all **design decisions**, **algorithm choices**, and **core implementations** were made independently. I understand every line of code in this project and can explain how it works.

---

## References

- Anthropic Claude: https://www.anthropic.com/claude
- OpenAI ChatGPT: https://chat.openai.com
- GitHub Copilot: https://github.com/features/copilot
