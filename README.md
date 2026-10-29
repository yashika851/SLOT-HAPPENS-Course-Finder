# SLOT HAPPENS — Course Material Search Engine

A Python information-retrieval project that searches course PDFs using a from-scratch TF-IDF implementation and cosine similarity, with keyword-frequency ranking as a baseline. This version includes a polished Streamlit dashboard.

## Run the dashboard

1. Extract the project ZIP.
2. Make sure the `dataset/` folder is beside `app.py` and contains the PDF files.
3. Open a terminal in the project folder.
4. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Start the dashboard:

   ```bash
   python -m streamlit run app.py
   ```

6. Streamlit will print a local URL (usually `http://localhost:8501`). Open it in your browser.

## Dashboard sections

- **Search:** Search the PDF collection, choose a ranking method, inspect ranked results, compare methods, and export results as CSV.
- **Overview:** Corpus size, subject distribution, vocabulary, token counts, and retrieval pipeline.
- **Algorithm Lab:** Compare keyword-frequency and TF-IDF + cosine retrieval using the relevance labels in `results/relevance_labels.json`.
- **Corpus Explorer:** Browse PDFs and inspect document frequency, IDF, term counts, TF, and TF-IDF.

## Core implementation

- `preprocessing.py`: PDF text extraction and token preprocessing.
- `indexing.py`: vocabulary, term counts, document frequency, and inverted index.
- `tfidf.py`: TF, IDF, and document vector calculations.
- `similarity.py`: query vector and cosine similarity calculations.
- `baseline.py`: keyword-frequency baseline.
- `ranking.py`: top-k ranking.
- `evaluation.py`, `run_experiments.py`: evaluation utilities and experiment runner.
- `app.py`: dashboard interface, using the same project modules for search.

## Evaluation caution

The relevance labels in `results/relevance_labels.json` are currently provisional subject-level judgments. Verify each query's relevant PDFs against their actual contents before reporting evaluation metrics as final. Query-time measurements exclude PDF extraction and initial index construction.

## Terminal version

The original command-line interface is still available:

```bash
python main.py
```

## Dataset

PDFs are included in the supplied project package in `dataset/`. If you distribute the source separately, add the dataset folder alongside `app.py`.
