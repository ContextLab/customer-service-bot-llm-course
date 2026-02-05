# Assignment 4: Context-Aware Customer Service Chatbot

> **Accept this assignment:** [GitHub Classroom Link](https://classroom.github.com/a/RYMMhVAL)
>
> **Due:** February 16, 2026 at 11:59 PM EST

---

**Timeline: 1 Week**

## Overview

In this assignment, you will build a customer service chatbot that uses **retrieval-augmented generation (RAG)** to answer questions about a fictional company called TechCo. Your system will:

1. **Retrieve** relevant FAQ entries using semantic search (sentence embeddings + cosine similarity)
2. **Compare** semantic search against a TF-IDF keyword baseline
3. **Generate** helpful responses based on the retrieved context

This assignment directly applies concepts from **Lecture 17 (RAG)** and builds on your knowledge of embeddings from **Lectures 11-12**.

## Learning Objectives

- Apply sentence embeddings for semantic search
- Implement cosine similarity retrieval (the same approach from Lecture 17)
- Compare semantic search vs. keyword matching baselines
- Build a working RAG-style chatbot with template-based response generation
- Evaluate retrieval quality using Accuracy at k and MRR

## Dataset

We provide a **50-entry FAQ knowledge base** for TechCo covering: account access, billing, product features, shipping and returns, technical support, and general information.

We also provide **20 test queries** with ground-truth relevant FAQ IDs for evaluation.

The data is automatically downloaded in the notebook.

## Your Tasks

### Required (100 points)

| Component | Points | Description |
|-----------|--------|-------------|
| Semantic Search | 20 | Implement semantic_search() using sentence-transformers |
| TF-IDF Baseline | 10 | Implement tfidf_search() using scikit-learn |
| Response Generation | 15 | Implement generate_response() with confidence logic |
| Evaluation | 20 | Implement evaluate_retrieval() and create bar chart |
| Error Analysis | 10 | 8+ examples with explanations |
| Interactive Demo | 5 | Implement chat() function |
| Code Quality | 10 | Clean, well-commented code |
| Reflection Essay | 10 | 300-500 word reflection |

### Optional Bonus (up to +8 points)

| Bonus | Points | Description |
|-------|--------|-------------|
| Multi-Turn Conversation | +2 | Implement ConversationManager for context tracking |
| LLM-Based Generation | +3 | Use FLAN-T5 for response generation |
| Hybrid Search | +2 | Combine TF-IDF and semantic scores |
| UMAP Visualization | +1 | Visualize the FAQ embedding space |

## Technical Requirements

- All libraries are installed in the notebook
- CPU is sufficient (Colab free tier works)
- No API keys required
- No external servers

## Submission

Push your completed notebook to your GitHub Classroom repository before the deadline.

### Checklist

- All required functions implemented (no NotImplementedError)
- Evaluation metrics computed with bar chart
- Error analysis with 8+ examples
- Reflection written (300-500 words)
- All cells execute without errors

**Deadline:** February 16, 2026 at 11:59 PM EST

## Questions?

1. Check this README and notebook markdown cells
2. Review Lecture 17 slides
3. Post in the course Discord
4. Attend office hours
