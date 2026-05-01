# YouTube Thumbnail Virality Predictor

A machine learning pipeline that predicts how viral a YouTube thumbnail is likely to be, using CLIP embeddings and XGBoost regression. Users upload a thumbnail and receive a predicted virality score, estimated views per day, and AI-generated improvement suggestions powered by Claude.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Setup Instructions](#setup-instructions)
- [How to Run — Step by Step](#how-to-run--step-by-step)
- [Methodology](#methodology)
- [Model Summary](#model-summary)
- [Key Findings](#key-findings)
- [Using the Demo](#using-the-demo)
- [Limitations & Future Work](#limitations--future-work)

---

## Project Overview

This project was developed as a Data Science Practicum at Regis University. It builds a predictive system that evaluates YouTube thumbnails for viral potential using two feature extraction strategies — CLIP semantic embeddings and manually engineered visual features — fed into an XGBoost regression model.

The system distinguishes between **Short-form** and **Long-form** video formats, as the two respond to completely different visual signals. The final demo allows a user to upload any thumbnail image and receive an instant virality prediction alongside actionable design feedback.

**Dataset:** 4,650 YouTube thumbnails across Music, People & Blogs, and Gaming categories, collected via the YouTube Data API v3.

**Best model performance:**
- Long-form (M8): R² = 0.736
- Short-form (M5): R² = 0.523

---

## Folder Structure

All project files live under a single base directory on Google Drive. The expected layout is:

```
MyDrive/capstone/
│
├── Project2data/
│   ├── raw/
│   │   ├── video_metadata.csv          # Raw metadata from YouTube API
│   │   ├── video_metadata_aligned.csv  # Metadata filtered to videos with thumbnails on disk
│   │   └── progress.json               # Tracks completed search queries (for resuming collection)
│   │
│   ├── thumbnails/                     # Downloaded .jpg thumbnail images
│   │   └── <video_id>_<quality>.jpg
│   │
│   ├── processed/
│   │   └── visual_features.csv         # 15 engineered visual features per video
│   │
│   └── embeddings/
│       ├── clip_embeddings_512.npy     # Raw CLIP vectors (shape: n_videos × 512)
│       ├── clip_embeddings_50.npy      # PCA-reduced vectors (shape: n_videos × 50)
│       └── video_ids.npy               # Video IDs matching each embedding row
│
├── models/
│   ├── m8_longform_clip.pkl            # Trained XGBoost model for long-form videos
│   ├── m8_longform_scaler.pkl          # StandardScaler fitted on long-form training data
│   ├── m5_shorts_clip.pkl              # Trained XGBoost model for Shorts
│   ├── m5_shorts_scaler.pkl            # StandardScaler fitted on Shorts training data
│   └── clip_feature_names.json         # Ordered list of CLIP feature column names
│
├── outputs/
│   └── ablation_results.csv            # R², MAE, RMSE for all 9 experimental models
│
└── notebooks/
    ├── Project2_0.ipynb                # Step 1: Data collection, preprocessing, EDA
    ├── project2_1.ipynb                # Step 2: CLIP embedding + visual feature extraction
    ├── project2_2.ipynb                # Step 3: Model training, ablation study, SHAP analysis
    └── Final_Model.ipynb               # Step 4: Interactive demo with Claude AI suggestions
```

> **Note:** The `thumbnails/` folder can be large (4,000+ images). Make sure you have sufficient Google Drive space before running the collection notebook.

---

## Setup Instructions

### Prerequisites

- A Google account with Google Drive and Google Colab access
- A YouTube Data API v3 key (free, via [Google Cloud Console](https://console.cloud.google.com))
- An Anthropic API key for the Claude-powered suggestions (optional — a fallback is built in)

### 1. Clone or upload the notebooks

Upload all four notebooks to your Google Colab environment or open them directly from Drive.

### 2. Set your base path

Every notebook uses a `BASE` variable pointing to your project folder on Drive. Update this line in each notebook to match your own Drive path:

```python
BASE = "/content/drive/MyDrive/capstone"
```

### 3. Install dependencies

Each notebook installs its own requirements at the top. You can also install everything manually in a Colab cell:

```bash
pip install xgboost scikit-learn shap scipy isodate \
    opencv-python Pillow pytesseract \
    google-api-python-client tqdm \
    umap-learn matplotlib seaborn \
    git+https://github.com/openai/CLIP.git \
    torch torchvision
```

> **Important:** After installing CLIP for the first time, Colab requires a **Runtime → Restart session** before importing it. Each affected notebook includes a note indicating where to restart.

### 4. Add your API keys

In `Project2_0.ipynb`, replace the placeholder values with your YouTube Data API keys:

```python
API_KEYS = [
    "YOUR_YOUTUBE_API_KEY_1",
    "YOUR_YOUTUBE_API_KEY_2",   # optional second key for quota rotation
]
```

In `Final_Model.ipynb` (and `project2_2.ipynb`), replace the Anthropic key:

```python
ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY"
```

If no valid Anthropic key is provided, the demo automatically falls back to rule-based suggestions derived from model training results.

---

## How to Run — Step by Step

Run the notebooks **in order**. Each notebook reads from files written by the previous one.

### Step 1 — `Project2_0.ipynb`: Data Collection & EDA

This notebook calls the YouTube Data API to search for videos across multiple categories and queries, downloads their thumbnail images, and saves metadata to CSV.

Key outputs:
- `Project2data/raw/video_metadata.csv`
- `Project2data/thumbnails/<video_id>_<quality>.jpg` (one file per video)
- `Project2data/raw/progress.json` (allows resuming if quota runs out mid-collection)

The notebook also performs exploratory data analysis including view count distributions, category breakdowns, Shorts vs. long-form ratios, and the age-adjusted `log_views_per_day` target variable.

> The API has a daily quota of ~10,000 units. The notebook supports two rotating API keys and saves progress after each search query so you can resume the next day without re-downloading completed batches.

---

### Step 2 — `project2_1.ipynb`: Feature Extraction

This notebook extracts two types of features from the downloaded thumbnails.

**CLIP embeddings** — loads OpenAI's ViT-B/32 model, encodes every thumbnail into a 512-dimensional vector, then applies PCA to reduce to 50 dimensions (retaining ~58% of variance).

Key outputs:
- `Project2data/embeddings/clip_embeddings_512.npy`
- `Project2data/embeddings/clip_embeddings_50.npy`
- `Project2data/embeddings/video_ids.npy`

**Manually engineered features** — uses OpenCV to extract 15 pixel-level measurements per image, then filters to the 8 with correlation ≥ |0.05| against the target:

| Feature | Description |
|---|---|
| `saturation` | Mean HSV saturation — how vivid colors are |
| `colorfulness` | Range of colors across the image |
| `warm_ratio` | Fraction of warm-toned pixels (red/orange/yellow) |
| `face_count` | Number of detected faces (Haar cascade) |
| `contrast` | Standard deviation of grayscale pixel values |
| `dark_ratio` | Fraction of pixels with brightness < 50/255 |
| `sharpness` | Laplacian variance — higher = crisper image |
| `edge_density` | Fraction of Canny edge pixels — higher = busier image |

Key output:
- `Project2data/processed/visual_features.csv`

---

### Step 3 — `project2_2.ipynb`: Model Training & Ablation Study

This notebook merges all features, trains models, and runs a systematic ablation study across 9 configurations.

**Baseline:** Ridge regression (R² = 0.045) — establishes a minimum threshold to beat.

**Ablation study:** Nine XGBoost models are trained varying the data subset (all / Shorts only / long-form only) and feature type (engineered / CLIP / both). All models use an 80/20 train-test split and are evaluated on R², MAE, and RMSE.

XGBoost hyperparameters used across all experiments:

```python
XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
```

SHAP (SHapley Additive exPlanations) values are then computed for the two best models (M8 and M5) to identify which CLIP dimensions drive predictions in each format.

Key outputs saved to `models/`:
- `m8_longform_clip.pkl` and `m8_longform_scaler.pkl`
- `m5_shorts_clip.pkl` and `m5_shorts_scaler.pkl`
- `clip_feature_names.json`

Key output saved to `outputs/`:
- `ablation_results.csv`

---

### Step 4 — `Final_Model.ipynb`: Interactive Demo

This is the user-facing demo. It loads the saved models and presents an `ipywidgets` interface inside Colab.

**To use the demo:**
1. Run all cells in order (restart after CLIP install if prompted)
2. Select **Long-form** or **Short** using the toggle buttons
3. Click **Upload Thumbnail** and choose a `.jpg` or `.png` file
4. Results appear automatically — no button press needed

**Output includes:**
- A preview of the uploaded thumbnail
- Virality score (log views/day), estimated views/day, and a tier label (🔥 High / ⚡ Medium / 📉 Low)
- AI-generated suggestions via Claude, or rule-based fallback suggestions if the API is unavailable

---

## Methodology

### Target Variable

Raw view counts are biased by video age — a three-year-old video naturally has more views than a one-week-old video. To correct for this, view counts are normalized by video age:

```
views_per_day = view_count / age_days
log_views_per_day = log(1 + views_per_day)
```

`log_views_per_day` is the model's prediction target throughout all experiments.

### CLIP Feature Extraction

CLIP (Contrastive Language-Image Pre-training) is a neural network trained to understand images in a semantically rich way — it encodes not just pixel patterns but high-level concepts like mood, composition, subject type, and visual style into a 512-dimensional vector. PCA is applied to reduce this to 50 dimensions before passing to XGBoost, balancing information retention with model efficiency.

### Why Separate Models for Shorts vs. Long-form?

Analysis showed that the two formats respond to fundamentally different visual signals. Training a single model on the combined dataset produced lower performance than training format-specific models. The SHAP analysis confirmed this: the top CLIP dimensions driving long-form virality (clip_3, clip_1 — associated with production quality and clear subject framing) are entirely different from those driving Shorts virality (clip_11, clip_7, clip_0 — associated with dynamic, energetic, motion-heavy content).

### AI Suggestions Integration

The demo sends the virality score, format, estimated views, and top CLIP dimensions to Claude via the Anthropic Messages API. Claude is given a system prompt describing the model's training findings and asked to generate four specific pieces of feedback in under 200 words. If the API call fails for any reason (rate limit, missing key, network error), the system falls back automatically to pre-written rule-based suggestions derived directly from SHAP analysis results.

---

## Model Summary

| Model | Data | Features | R² |
|---|---|---|---|
| M1 | All | Engineered | 0.345 |
| M2 | All | CLIP | 0.573 |
| M3 | All | CLIP + Engineered | 0.567 |
| M4 | Shorts | Engineered | 0.309 |
| **M5** | **Shorts** | **CLIP** | **0.523** ✓ |
| M6 | Shorts | CLIP + Engineered | 0.516 |
| M7 | Long-form | Engineered | 0.493 |
| **M8** | **Long-form** | **CLIP** | **0.736** ✓ |
| M9 | Long-form | CLIP + Engineered | 0.717 |

M5 and M8 are the production models used in the demo.

---

## Key Findings

- **CLIP features substantially outperform engineered pixel features** in all three data subsets, confirming that semantic understanding of thumbnails matters more than raw measurements like brightness or saturation.
- **Dark thumbnails consistently underperform.** `dark_ratio` was the single strongest negative predictor in the engineered feature analysis.
- **Cluttered compositions hurt performance.** High `edge_density` was the second-strongest negative predictor.
- **Vivid, colorful thumbnails do better.** `saturation`, `colorfulness`, `warm_ratio`, and `contrast` all had positive correlations with virality.
- **Face presence is a positive signal**, but not the dominant one — CLIP's semantic dimensions carry more predictive weight than face detection alone.
- **Shorts and long-form are fundamentally different problems.** The CLIP dimensions that matter for each format share no overlap in their top 5, which is why format-specific models are essential.

---

## Limitations & Future Work

The current system is scoped to three content categories (Music, People & Blogs, Gaming) and does not account for title text, channel authority, or posting time — all of which affect real-world performance. The model treats each thumbnail as independent and has no access to time-series view data, which limits its ability to detect true "tipping point" virality behavior.

Future improvements could include expanding the dataset to more categories with more varied search queries, incorporating user-provided variables like intended category and video duration for more precise benchmarking, and building personalized channel-level models that factor in a creator's historical click-through rate and subscriber base alongside platform-wide visual patterns.
