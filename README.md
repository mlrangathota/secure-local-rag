# Secure Local RAG for Enterprise Document Analysis

This repository contains the supplementary configuration files, Role-Based Access Control (RBAC) modules, and evaluation scripts for the paper: **"Privacy-Preserving Local Large Language Models for Enterprise Document Analysis"** (Accepted at FRUCT40).

## Repository Contents

* `secure_rag_pipeline.py`: The core local inference pipeline utilizing 8-bit quantized models and deterministic prompt synthesis.
* `rbac_enforcement.py` **[NEW]**: The strict Role-Based Access Control execution module. Cryptographically binds user identity tokens to the HNSW vector search to physically prevent unauthorized context retrieval before it reaches the LLM (Addresses prompt injection and internal threat models).
* `evaluation_pipeline.py` **[NEW]**: The automated scoring protocol (LLM-as-a-judge) used to systematically measure Context Adherence and Hallucination Rates, including the Top-K ablation testing scripts.
* `requirements.txt`: Python dependencies.

## Reproducing Empirical Results
To reproduce the mathematical thresholds mapping hardware optimization to semantic degradation (The INT4 Cliff):
1. Configure your local environment with an NVIDIA RTX 4090 (or equivalent 24GB VRAM GPU).
2. Execute the evaluation suite:
   ```bash
   python evaluation_pipeline.py
   ```

## Security & Privacy Guarantee
By executing this architecture within a zero-trust or air-gapped network, enterprise organizations maintain 100% data sovereignty.

## Contact
For questions regarding the architecture or empirical telemetry, please contact Mahalakshmi Ranga Prasad Thota at SAP America, Inc.
