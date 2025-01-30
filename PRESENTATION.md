# AI World Predictions Platform

## A Self-Evolving Multi-Agent Prediction System

---

### The Problem 🤔

- Traditional prediction models are:
  - Single-perspective
  - Slow to adapt
  - Limited by static training data
  - Unable to leverage real-time information

---

### Our Solution 💡

A multi-agent prediction platform that:

- Combines multiple AI experts (Technical, Sentiment, Economic) (GPT-4o)
- Integrates real-time data from multiple sources (Kalshi, Polymarket, Firecrawl)
- Uses RAG for context-aware predictions (Pinecone)

---

### System Architecture 🏗️

┌─────────────────┐ ┌──────────────┐ ┌────────────────┐ ┌────────────┐
│ Data Ingestion  │ │ Vector DB    │ │ Multi-Agent    │ │ REST API   │
│ (RSS/APIs)      │ │ (Pinecone)   │ │ Exchange       │ │            │
└─────────────────┘ └──────────────┘ └────────────────┘ └─────┬──────┘
                                        │
                                        ▼
                                        ┌────────────┐
                                        │ Web UI     │
                                        │ (React/TS) │
                                        └────────────┘

---

### Live Demo 🎮

Key Features:

1. Three prediction modes:
   - Fast (quick analysis)
   - Deep (thorough analysis)
   - Council (multi-expert consensus)
2. Real-time data integration
3. Performance tracking - Future Fine Tuning

---

### Technical Challenge: Multi-Source Data Pipeline 🔄

Most Interesting Implementation:

- Parallel AI web crawlers for diverse data sources:
  - News (World, Finance, Science, Sports)
  - Market data (Polymarket, Kalshi)
  - Social sentiment analysis
- Intelligent data extraction using LLMs
- Automated ETL pipeline with error handling
- Real-time vector database updates

Key Code: Our crawler orchestration system
