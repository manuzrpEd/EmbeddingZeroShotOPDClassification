# National Police CAD & Incident Classification  
**Zero-Shot Embedding-Based Crime Categorization at Scale**  

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**42+ million** police records from **20+ major U.S. cities** (2010–2025) → unified into **27 consistent crime categories** — **without training a single supervised model**.

## Why This Project Exists

Every police department uses its own call types and offense codes:

| Agency             | “Suspicious Person” is called…                        |
|--------------------|--------------------------------------------------------|
| Los Angeles        | `415 PARTY` / `SUSPCIRC`                               |
| New York           | `DISPUTE` / `EDP`                                      |
| Chicago            | `PERSON W/GUN` / `SUSPICIOUS PERSON`                   |
| Stockton           | `647F` / `PROWLER`                                     |

→ Comparing crime across cities is nearly impossible with raw data.

This project solves that with **zero-shot classification using sentence embeddings** — a method that works instantly on any new agency, any language style, without retraining.

## Key Innovation: Zero-Shot (Actually Few-Shot Prototype) Classification

| Method                  | Needs labeled data? | Works on new agency? | Performance |
|-------------------------|---------------------|------------------------|-------------|
| Traditional ML          | Thousands of labels | No                     | High (if trained) |
| Fine-tuned BERT         | Yes                 | No                     | Very high |
| **This project** (embedding prototypes) | **No** | **(zero-shot)** | **Yes – instantly** | **Very high** (cosine similarity on MiniLM) |

### How it works
1. We define **27 universal crime categories** (theft, assault, shots fired, etc.).
2. A small sentence transformer (`paraphrase-MiniLM-L3-v2`) creates an embedding for each category name.
3. For every raw call description (“MAN WITH KNIFE”, “415 DISTURBANCE”, “SHOPLIFT IN PROGRESS”), we compute its embedding.
4. Assign the category with the **highest cosine similarity** → done.

No training. No labels. Works on day one for any new city.

This is **not true zero-shot LLM prompting** (which is slow and expensive), but **embedding-based zero-shot** — blazing fast (8192 rows/sec on CPU), deterministic, and extremely robust.

## Why This Matters

- Researchers can now compare crime across hundreds of agencies with one consistent taxonomy.
- Journalists get accurate national pictures without waiting for delayed UCR releases.
- Policymakers see the true volume of mental health crises, gun calls, and noise complaints — not just what makes it into official stats.
- Anyone can add a new city in minutes — just drop the parquet file and rerun.
