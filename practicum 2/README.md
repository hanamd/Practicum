
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
