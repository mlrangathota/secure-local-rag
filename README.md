# Privacy-Preserving Local RAG Pipeline for Enterprise Document Analysis

This repository contains the core simulation framework and reference implementation for the paper titled "Privacy-Preserving Local Large Language Models for Enterprise Document Analysis" submitted to the 40th FRUCT Conference (Helsinki, Finland, November 4–6, 2026).

The proposed framework enables completely isolated, high-performance semantic search and text analysis across sensitive corporate documents without relying on external cloud endpoints or public APIs.

## Key Features
* **Complete Data Isolation:** All operations are executed completely locally inside the on-premise network environment.
* **8-Bit Model Quantization:** Drastically drops GPU memory parameters using INT8 schemas to run heavy 8B models on ordinary consumer grade graphics hardware.
* **Deterministic RAG Guardrails:** Enforces explicit context validation to eliminate model hallucinations and generate factual, reference-grounded responses.

## Project Directory Structure
```text
├── data/
│   └── corporate_vault.json     # Sample configuration data store
├── src/
│   ├── __init__.py
│   └── pipeline.py              # Core execution logic layer
├── requirements.txt             # Mandatory dependency version lock file
└── README.md                    # Repository documentation profile
```
## Prerequisites & Installation
Ensure your target environment possesses a CUDA-compatible NVIDIA graphics card with at least 12 GB of VRAM available.

1.  **Clone the project repository infrastructure:**
    ```bash
git clone https://github.com/mlrangathota/secure-local-rag.git

2.  **Initialize an isolated virtual python environment space:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the precise package dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start Guide
To execute a rapid validation test run across your sample document collection, execute the pipeline directly via terminal command prompts:
```bash
python src/pipeline.py
```
## Citation Information
If you build upon or reference this experimental testing framework in your academic studies, please cite our official conference entry:
```bibtex
@inproceedings{fruct2026privacypreserving,
  author    = {Mahalakshmi Ranga Prasad Thota},
  title     = {Privacy-Preserving Local Large Language Models for Enterprise Document Analysis},
  booktitle = {Proceedings of the 40th Conference of the Open Innovations Association FRUCT},
  year      = {2026},
  address   = {Helsinki, Finland},
  month     = {November}
}
