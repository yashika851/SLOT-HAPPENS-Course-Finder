from pathlib import Path
from collections import Counter
from time import perf_counter
import json
import re

import pandas as pd
import streamlit as st

from preprocessing import load_documents, preprocess_text
from indexing import build_index
from tfidf import build_tfidf
from similarity import calculate_similarities
from ranking import rank_documents
from baseline import keyword_frequency_search

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results"
LABELS_FILE = RESULTS_DIR / "relevance_labels.json"

st.set_page_config(
    page_title="Course Finder · SLOT HAPPENS",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#eef2ff; --muted:#9aa8c7; --panel:#111b31; --panel2:#15223b; --line:rgba(177,196,238,.13); --cyan:#70e1f5; --violet:#a78bfa; }
html, body, [class*="css"] { font-family:'Manrope', sans-serif; }
.stApp { background: radial-gradient(ellipse at 78% 0%, rgba(90,78,190,.20), transparent 34%), radial-gradient(ellipse at 4% 25%, rgba(36,171,197,.10), transparent 32%), #080e1b; color:var(--ink); }
.block-container { max-width: 1450px; padding-top: 1.5rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0e1729,#0a1020); border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:var(--ink); }
.hero { position:relative; overflow:hidden; padding: 2.2rem 2.35rem; border:1px solid rgba(151,170,240,.22); border-radius:26px; background:linear-gradient(120deg,rgba(24,39,69,.98),rgba(24,27,61,.95) 58%,rgba(22,47,68,.95)); margin-bottom:1.25rem; box-shadow:0 22px 70px rgba(0,0,0,.20); }
.hero:after { content:''; position:absolute; width:290px; height:290px; right:-50px; top:-115px; border-radius:50%; border:1px solid rgba(112,225,245,.24); box-shadow:0 0 0 32px rgba(112,225,245,.025),0 0 0 66px rgba(167,139,250,.035); }
.eyebrow { color:var(--cyan); text-transform:uppercase; letter-spacing:.19em; font:500 .72rem 'DM Mono',monospace; margin-bottom:.75rem; }
.hero h1 { font-size:clamp(2.2rem,4vw,3.7rem); letter-spacing:-.055em; line-height:1.02; margin:0; color:#f6f7ff; font-weight:800; }
.hero p { color:#b8c4df; font-size:1.02rem; max-width:720px; line-height:1.7; margin:.95rem 0 0; }
.pill { display:inline-block; border:1px solid rgba(112,225,245,.28); color:#9deafa; background:rgba(112,225,245,.08); border-radius:999px; padding:.38rem .72rem; font: .72rem 'DM Mono',monospace; margin:1.15rem .35rem 0 0; }
.section-title { font-size:1.25rem; font-weight:800; letter-spacing:-.025em; margin:.5rem 0 .2rem; color:#f3f6ff; }
.section-sub { color:var(--muted); font-size:.9rem; margin-bottom:1rem; }
.metric-card { background:linear-gradient(145deg,rgba(21,34,59,.95),rgba(14,23,41,.96)); border:1px solid var(--line); border-radius:18px; padding:1.05rem 1.15rem; min-height:112px; }
.metric-label { color:#9aa8c7; font-size:.77rem; text-transform:uppercase; letter-spacing:.1em; }
.metric-value { color:#f6f8ff; font-size:1.85rem; font-weight:800; letter-spacing:-.05em; margin-top:.28rem; }
.metric-note { color:#7e8eaf; font-size:.75rem; margin-top:.1rem; }
.result-card { border:1px solid var(--line); border-radius:18px; background:linear-gradient(145deg,rgba(19,31,54,.98),rgba(13,22,40,.98)); padding:1.1rem 1.2rem; margin:.65rem 0; }
.result-rank { color:#70e1f5; font:500 .76rem 'DM Mono',monospace; letter-spacing:.08em; }
.result-name { color:#f5f7ff; font-size:1.02rem; font-weight:800; overflow-wrap:anywhere; margin:.35rem 0 .4rem; }
.result-meta { color:#9ba9c6; font-size:.8rem; line-height:1.6; }
.tag { display:inline-block; color:#c9bcff; background:rgba(167,139,250,.10); border:1px solid rgba(167,139,250,.20); border-radius:7px; padding:.22rem .5rem; font-size:.72rem; margin:.2rem .25rem .1rem 0; }
.info-box { padding:1rem 1.1rem; border-radius:14px; border:1px solid rgba(112,225,245,.19); background:rgba(47,142,163,.08); color:#c4d5ed; line-height:1.65; font-size:.88rem; }
.warning-box { padding:1rem 1.1rem; border-radius:14px; border:1px solid rgba(250,204,21,.25); background:rgba(250,204,21,.06); color:#e9ddb0; line-height:1.65; font-size:.86rem; }
div[data-testid="stForm"] { border:1px solid var(--line); background:rgba(17,27,49,.82); padding:1.15rem; border-radius:20px; }
.stButton > button, .stFormSubmitButton > button { border-radius:11px; border:1px solid rgba(112,225,245,.35); background:linear-gradient(110deg,#65d9ed,#9b8afb); color:#07111e; font-weight:800; min-height:2.8rem; }
.stButton > button:hover, .stFormSubmitButton > button:hover { border-color:#d4faff; color:#07111e; box-shadow:0 0 24px rgba(112,225,245,.16); }
.stTextInput input, .stSelectbox [data-baseweb="select"], .stMultiSelect [data-baseweb="select"] { border-radius:11px; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
hr { border-color:var(--line); }
footer { visibility:hidden; }
small, .caption { color:var(--muted); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Reading PDFs and building the search index…")
def get_pipeline():
    if not DATASET_DIR.exists():
        return {}, {}, {}, {}, {}
    docs = load_documents(folder=str(DATASET_DIR))
    if not docs:
        return {}, {}, {}, {}, {}
    index = build_index(docs)
    vectors, idf = build_tfidf(index)
    return docs, index, vectors, idf, {name: Counter(tokens) for name, tokens in docs.items()}


def subject_for(filename):
    stem = Path(filename).stem
    subject = re.sub(r"_part\d+$", "", stem, flags=re.I).strip()
    known = {
        "aws": "AWS", "ml": "Machine Learning", "dsa": "Data Structures & Algorithms",
        "dbms": "Database Management", "mpmc": "Microprocessors & Microcontrollers",
        "operating systems": "Operating Systems", "artificial intelligence": "Artificial Intelligence",
        "embedded systems": "Embedded Systems", "software engineering": "Software Engineering"
    }
    return known.get(subject.lower(), subject.title())


def search(query, method, docs, index, vectors, idf, top_k=5):
    start = perf_counter()
    if method == "TF-IDF + Cosine Similarity":
        scores = calculate_similarities(query, vectors, idf)
        results = rank_documents(scores, top_k=top_k)
    else:
        results = keyword_frequency_search(query, index, top_k=top_k)
    elapsed = (perf_counter() - start) * 1000
    return results, elapsed


def metric_card(label, value, note):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)


def render_result_cards(query, results, method, token_counts, docs):
    q_tokens = list(dict.fromkeys(preprocess_text(query)))
    if not results:
        st.markdown('<div class="info-box">No indexed terms matched this query. Try a subject name or a more specific technical phrase.</div>', unsafe_allow_html=True)
        return
    max_score = max((float(score) for _, score in results), default=0.0)
    for rank, (filename, score) in enumerate(results, start=1):
        counts = token_counts.get(filename, Counter())
        matched = [term for term in q_tokens if counts.get(term, 0) > 0]
        subject = subject_for(filename)
        if method == "TF-IDF + Cosine Similarity":
            score_label = f"Cosine score · {float(score):.6f}"
        else:
            score_label = f"Matched-term frequency · {float(score):.0f}"
        relative = (float(score) / max_score) if max_score else 0
        tags = "".join(f'<span class="tag">{term}</span>' for term in matched[:8]) or '<span class="tag">No exact query-term overlap</span>'
        st.markdown(f'''<div class="result-card"><div class="result-rank">RESULT {rank:02d} &nbsp; / &nbsp; {score_label}</div><div class="result-name">▤ &nbsp; {filename}</div><div class="result-meta">{subject} &nbsp; · &nbsp; {len(docs.get(filename, [])):,} indexed tokens</div><div style="height:7px;background:#202c45;border-radius:20px;margin:.8rem 0 .6rem;overflow:hidden"><div style="height:7px;width:{relative*100:.2f}%;background:linear-gradient(90deg,#65d9ed,#a78bfa);border-radius:20px"></div></div><div class="result-meta">Query terms found</div><div>{tags}</div></div>''', unsafe_allow_html=True)


def read_labels(doc_names):
    if not LABELS_FILE.exists():
        return {}
    try:
        raw = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
        valid = set(doc_names)
        return {q: set(names).intersection(valid) for q, names in raw.items() if names}
    except Exception:
        return {}


def compute_experiments(docs, index, vectors, idf):
    labels = read_labels(docs.keys())
    rows = []
    query_rows = []
    for query, relevant in labels.items():
        if not relevant:
            continue
        for method in ["Keyword Frequency", "TF-IDF + Cosine Similarity"]:
            results, elapsed = search(query, method, docs, index, vectors, idf, 5)
            retrieved = [name for name, _ in results[:5]]
            precision = sum(name in relevant for name in retrieved) / 5
            rr = next((1 / rank for rank, name in enumerate(retrieved, 1) if name in relevant), 0.0)
            rows.append({"Algorithm": method, "Query": query, "Precision@5": precision, "MRR contribution": rr, "Time (ms)": elapsed})
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    detail = pd.DataFrame(rows)
    summary = detail.groupby("Algorithm", as_index=False).agg({"Precision@5":"mean", "MRR contribution":"mean", "Time (ms)":"mean"})
    summary = summary.rename(columns={"MRR contribution":"MRR", "Time (ms)":"Average time (ms)"})
    return summary, detail


docs, index, vectors, idf_values, token_counts = get_pipeline()

with st.sidebar:
    st.markdown("<div style='padding:.5rem 0 1.1rem'><div style='font:500 .7rem DM Mono,monospace;color:#70e1f5;letter-spacing:.18em'>SLOT HAPPENS</div><div style='font-size:1.35rem;font-weight:800;letter-spacing:-.04em;margin-top:.25rem'>Course Finder <span style='color:#a78bfa'>✦</span></div><div style='color:#8292b4;font-size:.8rem;margin-top:.25rem'>AI · Information Retrieval</div></div>", unsafe_allow_html=True)
    page = st.radio("WORKSPACE", ["Search", "Overview", "Algorithm Lab", "Corpus Explorer"], index=0, label_visibility="visible")
    st.markdown("---")
    st.markdown("<div style='font-size:.72rem;color:#8292b4;letter-spacing:.12em;margin-bottom:.5rem'>SYSTEM STATUS</div>", unsafe_allow_html=True)
    if docs:
        st.markdown(f"<div style='color:#8ce8d0;font-size:.86rem'>● Index ready</div><div style='font-size:.8rem;color:#8292b4;margin-top:.25rem'>{len(docs)} PDFs · {len(index['vocabulary']):,} unique terms</div>", unsafe_allow_html=True)
    else:
        st.error("No PDFs found. Put files in the dataset folder.")
    st.markdown("<div style='position:fixed;bottom:1.3rem;color:#5f7093;font-size:.7rem'>Built with Python · TF-IDF · Cosine similarity</div>", unsafe_allow_html=True)

if not docs:
    st.markdown('<div class="hero"><div class="eyebrow">COURSE MATERIAL INTELLIGENCE</div><h1>Your study library,<br>made searchable.</h1><p>No PDF documents were found. Extract your dataset ZIP into the project folder so that <code>dataset/</code> sits beside <code>app.py</code>.</p></div>', unsafe_allow_html=True)
    st.stop()

subjects = sorted({subject_for(name) for name in docs})
total_tokens = sum(len(tokens) for tokens in docs.values())

if page == "Search":
    st.markdown('<div class="hero"><div class="eyebrow">COURSE MATERIAL INTELLIGENCE · 01 / SEARCH</div><h1>Find the signal<br>in your syllabus.</h1><p>Search across your course PDFs in seconds. The engine weights distinctive terms, compares vectors, and ranks the documents most aligned with your query.</p><span class="pill">✦ TF-IDF ENGINE</span><span class="pill">COSINE SIMILARITY</span><span class="pill">34-PDF LIBRARY</span></div>', unsafe_allow_html=True)
    with st.form("search_form", clear_on_submit=False):
        qcol, mcol, kcol = st.columns([5, 2.5, 1.1])
        with qcol:
            query = st.text_input("SEARCH YOUR COURSE MATERIAL", value=st.session_state.get("active_query", ""), placeholder="Try: deadlock memory management, database normalization…")
        with mcol:
            method = st.selectbox("RANKING METHOD", ["TF-IDF + Cosine Similarity", "Keyword Frequency"], index=0)
        with kcol:
            top_k = st.selectbox("RESULTS", [3, 5, 8, 10], index=1)
        submitted = st.form_submit_button("Search library  →", use_container_width=True)
    if submitted:
        st.session_state["active_query"] = query.strip()
        st.session_state["active_method"] = method
        st.session_state["active_top_k"] = top_k
    active_query = st.session_state.get("active_query", "")
    active_method = st.session_state.get("active_method", "TF-IDF + Cosine Similarity")
    active_top_k = st.session_state.get("active_top_k", 5)
    if not active_query:
        st.markdown('<div class="section-title">A good place to start</div><div class="section-sub">Try a query from one of the indexed subjects.</div>', unsafe_allow_html=True)
        suggestions = ["artificial intelligence search algorithms", "database normalization", "process scheduling operating systems", "machine learning classification", "microprocessor instruction set", "aws cloud computing"]
        cols = st.columns(3)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 3]:
                if st.button("⌕  " + suggestion, key="suggest_" + str(i), use_container_width=True):
                    st.session_state["active_query"] = suggestion
                    st.session_state["active_method"] = "TF-IDF + Cosine Similarity"
                    st.session_state["active_top_k"] = 5
                    st.rerun()
        st.markdown('<div class="info-box" style="margin-top:1rem"><b>How it works</b> &nbsp; Your query is cleaned using the same preprocessing rules as the PDFs, converted into TF-IDF weights, compared with document vectors using cosine similarity, then ranked from highest to lowest score.</div>', unsafe_allow_html=True)
    else:
        results, elapsed = search(active_query, active_method, docs, index, vectors, idf_values, active_top_k)
        q_terms = preprocess_text(active_query)
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Results found", len(results), f"Top {active_top_k} requested")
        with c2: metric_card("Query time", f"{elapsed:.2f} ms", "Search phase only · index already built")
        with c3: metric_card("Query terms", len(set(q_terms)), f"{len(set(q_terms).intersection(index['vocabulary']))} in vocabulary")
        st.markdown(f'<div class="section-title" style="margin-top:1.5rem">Ranked documents</div><div class="section-sub">Method: {active_method} · Query: “{active_query}”</div>', unsafe_allow_html=True)
        render_result_cards(active_query, results, active_method, token_counts, docs)
        if results:
            export = pd.DataFrame([{"Rank": rank, "Document": name, "Subject": subject_for(name), "Score": score, "Algorithm": active_method, "Query": active_query} for rank, (name, score) in enumerate(results, 1)])
            st.download_button("↓  Export these results as CSV", export.to_csv(index=False).encode("utf-8"), file_name="course_search_results.csv", mime="text/csv")
        with st.expander("Compare both ranking methods for this query"):
            base_results, base_time = search(active_query, "Keyword Frequency", docs, index, vectors, idf_values, active_top_k)
            tfidf_results, tfidf_time = search(active_query, "TF-IDF + Cosine Similarity", docs, index, vectors, idf_values, active_top_k)
            left, right = st.columns(2)
            with left:
                st.markdown("**TF-IDF + Cosine Similarity**")
                st.caption(f"{tfidf_time:.2f} ms")
                st.dataframe(pd.DataFrame([{"Rank": i, "Document": n, "Cosine score": round(float(s), 6)} for i, (n,s) in enumerate(tfidf_results,1)]), hide_index=True, use_container_width=True)
            with right:
                st.markdown("**Keyword-frequency baseline**")
                st.caption(f"{base_time:.2f} ms")
                st.dataframe(pd.DataFrame([{"Rank": i, "Document": n, "Term frequency": int(s)} for i, (n,s) in enumerate(base_results,1)]), hide_index=True, use_container_width=True)

elif page == "Overview":
    st.markdown('<div class="hero"><div class="eyebrow">SYSTEM OVERVIEW · 02 / LIBRARY</div><h1>One library.<br>Every subject in reach.</h1><p>A live snapshot of the indexed course collection, vocabulary, and retrieval pipeline.</p><span class="pill">INDEX BUILT FROM YOUR PDFS</span></div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("PDF documents", f"{len(docs)}", "Loaded from dataset/")
    with c2: metric_card("Subjects", f"{len(subjects)}", "Grouped from filenames")
    with c3: metric_card("Vocabulary", f"{len(index['vocabulary']):,}", "Unique processed terms")
    with c4: metric_card("Processed tokens", f"{total_tokens:,}", "Across all documents")
    st.markdown('<div class="section-title" style="margin-top:1.6rem">Your subject mix</div><div class="section-sub">Number of PDFs available in each subject.</div>', unsafe_allow_html=True)
    subject_counts = pd.Series([subject_for(name) for name in docs], name="Subject").value_counts().sort_values(ascending=True)
    st.bar_chart(subject_counts, height=360)
    left,right = st.columns([1.15,1])
    with left:
        st.markdown('<div class="section-title">Retrieval pipeline</div><div class="section-sub">From raw PDF to ranked results.</div>', unsafe_allow_html=True)
        for n, title, desc in [("01","Extract & preprocess","PyMuPDF extraction, lowercase, alphabetic tokenization, stop-word removal."),("02","Build the index","Vocabulary, term counts, document frequency, document lengths, inverted index."),("03","Weight the terms","TF × IDF creates a weighted vector for each document and query."),("04","Score & rank","Cosine similarity ranks the closest document vectors; keyword frequency is the baseline.")]:
            st.markdown(f'<div class="result-card" style="display:flex;gap:1rem;align-items:flex-start"><div style="font:500 .85rem DM Mono,monospace;color:#70e1f5;padding-top:.2rem">{n}</div><div><div style="font-weight:800;color:#f5f7ff">{title}</div><div class="result-meta">{desc}</div></div></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-title">Most frequent indexed terms</div><div class="section-sub">Top terms by total count across the collection.</div>', unsafe_allow_html=True)
        total_counts = Counter()
        for counts in token_counts.values(): total_counts.update(counts)
        freq_df = pd.DataFrame(total_counts.most_common(12), columns=["Term", "Occurrences"]).set_index("Term")
        st.bar_chart(freq_df, height=360)

elif page == "Algorithm Lab":
    st.markdown('<div class="hero"><div class="eyebrow">EXPERIMENT STUDIO · 03 / ALGORITHM LAB</div><h1>Measure the ranking.<br>Not just the interface.</h1><p>Compare the project’s two retrieval methods using the relevance labels stored in your project.</p><span class="pill">KEYWORD FREQUENCY</span><span class="pill">TF-IDF + COSINE</span></div>', unsafe_allow_html=True)
    with st.spinner("Running both retrieval methods across labelled test queries…"):
        summary, detail = compute_experiments(docs, index, vectors, idf_values)
    st.markdown('<div class="warning-box"><b>Evaluation note:</b> the current relevance labels in <code>results/relevance_labels.json</code> are provisional subject-level judgments. Treat these numbers as preliminary until the team checks each query’s relevant PDFs against their actual contents.</div>', unsafe_allow_html=True)
    if summary.empty:
        st.info("No non-empty relevance labels were found. Add relevant PDF filenames to results/relevance_labels.json to enable evaluation.")
    else:
        c1,c2 = st.columns(2)
        for i, row in summary.iterrows():
            target = c1 if i == 0 else c2
            with target:
                st.markdown(f'<div class="result-card"><div class="result-rank">METHOD</div><div class="result-name">{row["Algorithm"]}</div><div class="metric-label">Precision@5</div><div class="metric-value">{row["Precision@5"]:.3f}</div><div class="result-meta">MRR: {row["MRR"]:.3f} · Avg. query time: {row["Average time (ms)"]:.2f} ms</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="margin-top:1.3rem">Metric comparison</div><div class="section-sub">Higher Precision@5 and MRR are better. Lower query time is faster.</div>', unsafe_allow_html=True)
        chart_df = summary.set_index("Algorithm")[["Precision@5", "MRR"]]
        st.bar_chart(chart_df, height=300)
        time_df = summary.set_index("Algorithm")[["Average time (ms)"]]
        st.bar_chart(time_df, height=220)
        st.markdown('<div class="section-title" style="margin-top:1rem">Per-query results</div>', unsafe_allow_html=True)
        st.dataframe(detail.rename(columns={"MRR contribution":"Reciprocal rank"}).round(4), hide_index=True, use_container_width=True)
        st.download_button("↓  Download experiment results", detail.to_csv(index=False).encode("utf-8"), file_name="algorithm_comparison.csv", mime="text/csv")
    with st.expander("What do these metrics mean?"):
        st.markdown("**Precision@5** is the fraction of the top five results judged relevant. **MRR (Mean Reciprocal Rank)** rewards a relevant result appearing near the top. **Average query time** measures search execution after the index has been built; it excludes PDF extraction and initial indexing.")

else:  # Corpus Explorer
    st.markdown('<div class="hero"><div class="eyebrow">DOCUMENT COLLECTION · 04 / EXPLORER</div><h1>Know what is<br>inside your library.</h1><p>Browse the indexed PDFs, inspect token counts, and explore the terms represented in the vocabulary.</p><span class="pill">DOCUMENT INDEX</span><span class="pill">VOCABULARY BROWSER</span></div>', unsafe_allow_html=True)
    search_filter = st.text_input("Filter documents", placeholder="Type a subject or filename…")
    rows = []
    for name, tokens in docs.items():
        subj = subject_for(name)
        if search_filter.lower() not in name.lower() and search_filter.lower() not in subj.lower():
            continue
        rows.append({"Document": name, "Subject": subj, "Processed tokens": len(tokens), "Unique terms": len(token_counts[name])})
    st.markdown(f'<div class="section-sub">Showing {len(rows)} of {len(docs)} documents.</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=420)
    st.markdown('<div class="section-title" style="margin-top:1.5rem">Vocabulary lookup</div><div class="section-sub">Search for a term to inspect its document frequency and TF-IDF weight.</div>', unsafe_allow_html=True)
    term = st.text_input("Look up a term", placeholder="e.g. database, scheduling, neural…").strip().lower()
    if term:
        df = index["document_frequency"].get(term, 0)
        idf = idf_values.get(term)
        if idf is None:
            st.warning(f'“{term}” is not in the indexed vocabulary. Try a different spelling or a related term.')
        else:
            a,b,c = st.columns(3)
            with a: metric_card("Document frequency", f"{df}", f"Appears in {df}/{len(docs)} PDFs")
            with b: metric_card("IDF weight", f"{idf:.4f}", "Natural log of N / DF")
            with c: metric_card("Vocabulary status", "Indexed", "Term found in the corpus")
            postings = index["inverted_index"].get(term, {})
            post_df = pd.DataFrame([{"Document": name, "Subject": subject_for(name), "Occurrences": count, "TF": round(count / max(index["document_lengths"].get(name, 1),1), 6), "TF-IDF": round((count / max(index["document_lengths"].get(name, 1),1)) * idf, 6)} for name, count in sorted(postings.items(), key=lambda item: -item[1])])
            st.dataframe(post_df, hide_index=True, use_container_width=True)

st.markdown("<div style='text-align:center;color:#586987;font-size:.74rem;padding:2rem 0 0'>SLOT HAPPENS · Course Material Search Engine · Core retrieval formulas implemented in Python</div>", unsafe_allow_html=True)
