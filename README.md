# IE7500_Group9_FinalProject

This repository contains the code, models, and data pipeline for building a LinkedIn Job Recommender. Given a user’s resume, the system:

- Classifies the resume into a career category
- Recommends relevant job postings from a large dataset of LinkedIn job descriptions

Our goal: help users discover suitable jobs based purely on the text of their resumes.

---

## Project Overview

### 1. Resume Classification

We trained different models to classify resumes into one of ~25 career categories (e.g. Engineer, Finance, Teacher, etc.) using:

- TF-IDF vectorization
- Logistic Regression
- SVM (Selected as final model since it performs the best)
- Naive Bayes
- Hyperparameter tuning via GridSearchCV

---

### 2. Job Recommendation

We compute similarities between resumes and job descriptions to recommend the Top 5 jobs for each resume:

**Vectorization methods:**

- TF-IDF
- CountVectorizer
- spaCy embeddings
- BERT embeddings

**Hyperparameter tuning:**

- max_features
- ngram_range

**Evaluation metric:**

- Mean Top-5 Cosine Similarity

---

## Team Members

- Kalyaan Narnamalpuram
- Justin Magalona
- Yash Sonpal
