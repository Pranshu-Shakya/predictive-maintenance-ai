# AI-Powered Predictive Maintenance & Troubleshooting System

An AI-powered predictive maintenance application that combines Machine
Learning, Retrieval-Augmented Generation (RAG), vector search, and
Google Gemini to predict machine failure and generate domain-grounded
troubleshooting recommendations.

## Features

- Machine failure prediction using Random Forest
- Failure probability estimation
- Machine fault identification
- Maintenance knowledge retrieval using RAG
- FAISS vector database
- Gemini-powered troubleshooting
- FastAPI backend
- React dashboard
- Source attribution for retrieved maintenance documents

## Architecture

React
↓
FastAPI
↓
Random Forest
↓
Failure Prediction
↓
RAG / FAISS
↓
Maintenance Knowledge
↓
Google Gemini
↓
AI Troubleshooting Report

## Tech Stack

### Machine Learning
- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest

### Generative AI
- Google Gemini API
- RAG
- Sentence Transformers
- FAISS
- LangChain

### Backend
- FastAPI
- Pydantic

### Frontend
- React
- Tailwind CSS

## Machine Parameters

The model uses:

- Temperature
- Vibration
- Pressure
- RPM
- Operating Hours
- Flow Rate

## ML Problem

Binary classification:

- 0 → Normal
- 1 → Failure

The system also estimates failure probability using
Random Forest `predict_proba()`.

## RAG Pipeline

Maintenance documents are:

1. Loaded
2. Split into chunks
3. Converted into embeddings
4. Stored in FAISS
5. Retrieved using semantic similarity

The retrieved context is passed to Gemini to generate
grounded troubleshooting recommendations.

## Important Note

The current prototype uses synthetic sensor data generated
from domain-inspired relationships. Model performance on
this dataset should not be interpreted as real-world
industrial model performance.

## Future Improvements

- Real industrial sensor data
- Time-series monitoring
- Multiclass fault prediction
- IoT sensor integration
- Real-time alerts
- Model monitoring
- Cloud deployment