# Review Analysis RAG Pipeline

A Python pipeline (built for Google Colab) that loads a dataset of product/customer reviews, runs sentiment analysis on each review, indexes them with FAISS for semantic search, and lets you ask natural-language questions about the reviews using a GPT-powered Retrieval-Augmented Generation (RAG) query loop.

## Features

- **Flexible dataset loading** — accepts `.csv`, `.xlsx`, or prompts for file upload (Colab) if no path is given
- **Automatic encoding detection** for CSV files using `chardet`
- **Memory optimization** — downcasts numeric columns to reduce memory footprint
- **Text cleaning** — lowercases and strips non-alphanumeric characters from review text
- **Batch sentiment analysis** using a Hugging Face `transformers` pipeline
- **Semantic search index** built with `sentence-transformers` embeddings and `FAISS`
- **RAG-based Q&A** — retrieves the most relevant reviews for a query and summarizes themes, complaints, and sentiment using GPT (via the `boltiotai` API wrapper)
- **Interactive query loop** for asking repeated questions about the dataset

## Requirements

- Python 3.8+
- Google Colab environment (uses `google.colab.files` for uploads) — or adapt `load_dataset()` to skip the upload prompt when running locally
- A Bolt IoT OpenAI API key, available as the environment variable `BOLTIOT_API_KEY`

### Dependencies

```
boltiotai
faiss-cpu
sentence-transformers
pandas
numpy
chardet
openpyxl
transformers
torch
```

These are installed automatically by the `pip install` lines at the top of the script.

## Setup

1. Set your API key as an environment variable before running:
   ```bash
   export BOLTIOT_API_KEY="your-api-key-here"
   ```
2. Run the script in a Google Colab notebook (or adapt the file-upload step for local use).
3. When prompted, upload a `.csv` or `.xlsx` file containing reviews. **The first column of the file is assumed to contain the review text.**

## How It Works

1. **`load_dataset()`**
   - Loads the file (auto-detects CSV encoding, or reads Excel directly)
   - Optimizes memory usage for numeric columns
   - Cleans review text (lowercase, strip punctuation/symbols)
   - Randomly samples up to `subset_size` rows (default 10,000) if the dataset is larger
   - Runs batch sentiment analysis (`positive` / `negative` / `neutral`) and prints a summary

2. **`build_faiss_index()`**
   - Encodes cleaned review text into embeddings using the `all-MiniLM-L6-v2` sentence-transformer model
   - Builds a FAISS `IndexFlatL2` index over the embeddings for fast similarity search

3. **`analyze_reviews_with_gpt()`**
   - Embeds the user's query
   - Retrieves the top `k` most similar reviews from the FAISS index
   - Constructs a prompt combining those reviews and the user's question
   - Sends the prompt to GPT-3.5-turbo (via `boltiotai`) to generate a summary of themes, complaints, and sentiment

4. **`process_query()`**
   - Wraps the RAG call in error handling and returns a structured result (`query`, `analysis`, or `error`)

5. **Interactive loop**
   - Prompts the user to enter queries in a terminal-style loop
   - Type `quit` to exit

## Usage Example

```
Enter query (or 'quit'): What are the most common complaints about battery life?
{'query': 'What are the most common complaints about battery life?',
 'analysis': 'Several reviews mention the battery draining quickly...'}

Enter query (or 'quit'): quit
```

## Notes & Limitations

- Assumes the **first column** of the uploaded file is the review text — reorder columns if needed.
- Sentiment analysis uses the default Hugging Face `sentiment-analysis` pipeline, which only returns `POSITIVE`/`NEGATIVE` labels (no built-in `neutral` class), so the "Neutral" count in the summary will typically be 0 unless a custom model is configured.
- Designed for Google Colab; the `files.upload()` call will need to be replaced with a local file path when running outside Colab.
- Large datasets are automatically subsampled to `subset_size` (default 10,000 rows) to control memory and processing time — adjust this parameter as needed.
- Requires a valid `BOLTIOT_API_KEY`; the script will raise an error on startup if it's missing.

## File Structure

This is a single-script pipeline with no external file dependencies beyond the dataset you provide at runtime.
