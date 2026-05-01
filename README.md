# 🎬 YouTube Thumbnail Virality Predictor

> **Masters in Data Science — Capstone Project**  
> Predicting how viral a YouTube thumbnail will be using CLIP embeddings and XGBoost — before the video is even uploaded.

---

## 📌 Project Overview

A thumbnail is the first — and often only — impression a video makes in a feed. Viewers decide whether to click in under 2 seconds, yet most creators choose thumbnails by gut feeling alone.

This project builds a machine learning pipeline that:
- Collects YouTube video metadata and thumbnails via the YouTube Data API v3
- Extracts visual features from thumbnails using OpenAI's CLIP model and OpenCV
- Trains separate XGBoost regression models for **Shorts** and **Long-form** videos
- Provides a virality score, estimated views per day, and AI-powered suggestions via an interactive demo

**Best model result: R² = 0.736 on long-form videos using CLIP embeddings**

---

## 📁 File Structure

```
capstone/
│
├── Project2_0.ipynb          # Notebook 1 — Data collection, preprocessing & EDA
├── project2_1.ipynb          # Notebook 2 — CLIP embeddings & visual feature extraction
├── project2_2.ipynb          # Notebook 3 — Model training, ablation study & SHAP analysis
├── Final_Model.ipynb         # Interactive demo — upload a thumbnail, get a prediction
│
└── Project2data/
    ├── raw/
    │   ├── video_metadata.csv          # 4,710 rows × 23 cols — master dataset
    │   ├── video_metadata_aligned.csv  # 3,576 rows — videos with thumbnails on disk
    │   └── progress.json               # checkpoint tracker for data collection
    │
    ├── thumbnails/
    │   └── <video_id>_<quality>.jpg    # 3,555 downloaded thumbnail images
    │
    ├── embeddings/
    │   ├── clip_embeddings_512.npy     # raw CLIP vectors  (3555, 512)
    │   ├── clip_embeddings_50.npy      # PCA-reduced vectors (3555, 50)
    │   └── video_ids.npy               # matching video IDs for each row
    │
    └── processed/
        └── visual_features.csv         # 3,576 rows × 16 cols — OpenCV features

models/
    ├── m8_longform_clip.pkl            # best long-form model  (R²=0.736)
    ├── m8_longform_scaler.pkl          # StandardScaler for M8
    ├── m5_shorts_clip.pkl              # best Shorts model  (R²=0.523)
    ├── m5_shorts_scaler.pkl            # StandardScaler for M5
    └── clip_feature_names.json         # ordered list of CLIP feature column names

outputs/
    ├── ablation_results.csv            # 9-experiment results table
    ├── top10_predicted_viral.png       # top 10 thumbnails M8 predicts will go viral
    ├── bottom10_predicted_low.png      # bottom 10 thumbnails M8 predicts will underperform
    └── top_vs_bottom_features.png      # visual feature comparison chart
```

---

## 🚀 How to Run

### Prerequisites

- Google Colab (recommended — GPU available for CLIP)
- Google Drive mounted at `/content/drive/MyDrive/capstone`
- Two Google Cloud projects with YouTube Data API v3 enabled
- Python 3.10+

### Step 1 — Install dependencies

Run the install cell in any notebook:

```python
!pip install -q xgboost scikit-learn shap scipy isodate \
    opencv-python Pillow pytesseract \
    google-api-python-client tqdm \
    umap-learn matplotlib seaborn

!pip install git+https://github.com/openai/CLIP.git torch torchvision
```

### Step 2 — Set your API keys

In `Project2_0.ipynb`, replace the placeholder values:

```python
API_KEYS = [
    "YOUR_FIRST_API_KEY_HERE",
    "YOUR_SECOND_API_KEY_HERE"
]
```

> ⚠️ **Never commit your API keys to GitHub.** Add them directly in Colab and keep this cell out of version control.

### Step 3 — Run notebooks in order

| Order | Notebook | What it does | Estimated time |
|---|---|---|---|
| 1 | `Project2_0.ipynb` | Collect data via YouTube API, run EDA | 2–3 days (API quota limits) |
| 2 | `project2_1.ipynb` | Extract CLIP embeddings and OpenCV features | 20–40 min |
| 3 | `project2_2.ipynb` | Train models, run ablation study, SHAP analysis | 15–30 min |
| 4 | `Final_Model.ipynb` | Interactive demo — upload and predict | Instant |

### Step 4 — Run the interactive demo

Open `Final_Model.ipynb` and run all cells. You will see:

1. Select **Long-form** or **Short** using the toggle button
2. Click **Upload Thumbnail** and choose any `.jpg` or `.png` image
3. The model will output:
   - Virality score (log_views_per_day)
   - Estimated views per day
   - Virality tier (🔥 High / ⚡ Medium / 📉 Low)
   - AI-powered improvement suggestions

---

## 🗂️ CSV Files Explained

### `video_metadata.csv`
The master dataset. One row per video. Saved after every completed search query via checkpoint system so no data is lost if the API quota runs out mid-collection.

| Column | Description |
|---|---|
| `video_id` | Unique YouTube ID |
| `title` | Video title |
| `channel / channel_id` | Creator name and ID |
| `published_at` | Upload timestamp (UTC) |
| `duration` | ISO 8601 format e.g. PT3M45S |
| `is_short` | True if duration ≤ 60 seconds |
| `view_count / like_count / comment_count` | Engagement metrics |
| `category_id / category` | YouTube category number and our label |
| `tags` | Creator-added tags list |
| `thumbnail_url` | Direct CDN image URL |
| `thumbnail_quality / width / height` | Downloaded quality level |
| `thumbnail_path` | Local Drive file path |
| `log_views / year / age_days` | Added during EDA |
| `views_per_day / log_views_per_day` | **Target variable** — age-adjusted virality metric |

### `video_metadata_aligned.csv`
Same columns as above but filtered to only the 3,576 videos that have a thumbnail file saved on disk. This is the dataset used for all feature extraction and model training.

### `visual_features.csv`
OpenCV-extracted features. One row per video, 16 columns.

| Column | What it measures |
|---|---|
| `video_id` | Join key |
| `brightness` | Mean grayscale value / 255 |
| `contrast` | Std deviation of grayscale |
| `saturation` | Mean HSV S-channel |
| `colorfulness` | Hasler-Susstrunk colorfulness score |
| `edge_density` | Fraction of pixels that are edges (Canny) |
| `sharpness` | Laplacian variance |
| `avg_red / avg_green / avg_blue` | Mean channel values |
| `warm_ratio` | Fraction of warm-toned pixels |
| `dark_ratio` | Fraction of very dark pixels (< 50 brightness) |
| `bright_ratio` | Fraction of very bright pixels (> 200) |
| `aspect_ratio` | Width / height |
| `face_count / face_present` | Haar cascade face detection |

### `ablation_results.csv`
Results from the 9-experiment ablation study. One row per experiment.

| Column | Description |
|---|---|
| `Experiment` | M1 through M9 |
| `Filter` | all / short / long |
| `Features` | Engineered / CLIP / Both |
| `N_rows` | Number of videos in this subset |
| `R2 / MAE / RMSE` | Model performance metrics |

---

## 🧠 How the Model Works

### Virality target variable

Raw `view_count` is biased by age — a 2015 video has had years to accumulate views. We normalised it:

```python
age_days          = (today - published_at).days.clip(lower=1)
views_per_day     = view_count / age_days
log_views_per_day = log1p(views_per_day)   # ← this is what the model predicts
```

### Two feature types

**CLIP embeddings (50 dimensions)**  
OpenAI's CLIP model converts each thumbnail into a 512-number semantic fingerprint capturing mood, composition, and content type. PCA reduces this to 50 dimensions before training.

**OpenCV engineered features (8 kept of 15)**  
Hand-crafted pixel measurements selected by Pearson correlation analysis (threshold |r| ≥ 0.05):
`saturation, colorfulness, contrast, sharpness, warm_ratio, edge_density, dark_ratio, face_count`

### Two production models

| Model | Data | Features | R² | MAE |
|---|---|---|---|---|
| M5 | Shorts only | CLIP (50 dims) | 0.523 | 1.144 |
| M8 | Long-form only | CLIP (50 dims) | 0.736 | 1.015 |

### Key findings

- CLIP embeddings outperform hand-crafted features in every single experiment
- Long-form videos are far more predictable than Shorts (R²=0.736 vs 0.523)
- Adding engineered features on top of CLIP slightly hurts performance — they are redundant, not additive
- Shorts and long-form respond to completely different visual cues — zero shared top CLIP dimensions
- `dark_ratio` is the single strongest negative predictor — darker thumbnails consistently underperform

---

## 🤖 AI Suggestions — Work in Progress

The interactive demo includes an AI suggestion feature powered by the Anthropic Claude API. When a thumbnail is uploaded, Claude receives the model's prediction results along with key findings from the SHAP analysis and provides:

- A one-sentence performance summary
- Two specific things to improve based on what the model found hurts virality
- Two things to keep or amplify based on what helps virality
- A format-specific tip for Shorts or long-form content

**Current status:** The AI suggestion feature is a working proof of concept. It currently requires an active Anthropic API key with sufficient credits. If no credits are available the demo automatically falls back to model-based suggestions derived directly from the SHAP findings — no API call required.

**Planned improvements:**

- Replace the fallback with a locally hosted small language model so suggestions always work without API credits
- Make suggestions category-aware — advice for a gaming thumbnail should differ from a music thumbnail
- Add visual analysis so Claude can actually see the uploaded thumbnail rather than just receiving the numerical scores
- Build a Streamlit web application so the demo is accessible without Google Colab
- Add A/B comparison mode — upload two thumbnails and get a recommendation on which will perform better

---

## ⚠️ Limitations

- **Search bias** — data was collected via keyword search, not random sampling. Popular videos are overrepresented
- **Incomplete collection** — only 9 of 24 planned search queries were completed due to daily API quota limits. Remaining categories: Entertainment, How-to, Education, Science & Tech, Sports
- **No text detection** — thumbnail text overlays were excluded because OCR was unreliable. This is a known predictor of click-through rate
- **CLIP dimension opacity** — the top CLIP dimensions (clip_3, clip_1 for long-form) cannot be directly interpreted in human terms
- **Shorts predictability** — R²=0.523 for Shorts reflects the fact that many Shorts use auto-generated video frames rather than intentionally designed thumbnails

---

## 📊 Model Performance Summary

| Experiment | Filter | Features | R² | MAE | RMSE |
|---|---|---|---|---|---|
| M1 | All | Engineered | 0.345 | 1.756 | 2.389 |
| M2 | All | CLIP | 0.573 | 1.291 | 1.929 |
| M3 | All | Both | 0.567 | 1.299 | 1.943 |
| M4 | Shorts | Engineered | 0.310 | 1.458 | 2.138 |
| **M5** | **Shorts** | **CLIP** | **0.523** | **1.144** | **1.777** |
| M6 | Shorts | Both | 0.516 | 1.158 | 1.790 |
| M7 | Long-form | Engineered | 0.493 | 1.548 | 2.222 |
| **M8** | **Long-form** | **CLIP** | **0.736** | **1.015** | **1.603** |
| M9 | Long-form | Both | 0.717 | 1.031 | 1.659 |

---

## 🔧 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Core language |
| Google Colab | Development environment |
| Google Drive | Persistent file storage |
| YouTube Data API v3 | Data collection |
| OpenAI CLIP ViT-B/32 | Semantic image embeddings |
| OpenCV | Visual feature extraction |
| XGBoost | Regression model |
| scikit-learn | PCA, StandardScaler, cross-validation |
| SHAP | Model explainability |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Visualisation |
| Anthropic Claude API | AI suggestion generation (work in progress) |

---

## 📚 References

- Anthropic. (2025–2026). Claude [AI assistant]. https://claude.ai
- Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. OpenAI. https://openai.com/research/clip
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. ACM KDD.
- YouTube Data API v3 Documentation. https://developers.google.com/youtube/v3
- Lundberg, S. & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
- Hasler, D. & Suesstrunk, S. (2003). Measuring Colorfulness in Natural Images. SPIE Electronic Imaging.

---

## 👤 Author

**Hana M.**  
Masters in Data Science — Capstone Project
