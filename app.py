import streamlit as st
import json
import re
import requests
import numpy as np
import faiss
import pickle
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from groq import Groq
from huggingface_hub import hf_hub_download
import os

st.set_page_config(
    page_title="IMD Climate Fake News Detector",
    page_icon="🌦️",
    layout="wide"
)

if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""
if "page" not in st.session_state:
    st.session_state.page = "api_key"
if "history" not in st.session_state:
    st.session_state.history = []

def api_key_page():
    st.markdown("""
        <div style='text-align:center; padding: 60px 0 20px 0;'>
            <h1>🌦️ IMD Climate Fake News Detector</h1>
            <p style='font-size:18px; color:gray;'>Check climate claims against IMD reports (2008–2024) + live weather data</p>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🔑 Enter your Groq API Key to get started")
        key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
        st.markdown("""
            <small>Don't have a key? Get one free at
            <a href='https://console.groq.com' target='_blank'>console.groq.com</a></small>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("🚀 Continue to App", use_container_width=True, type="primary"):
            if key.strip():
                st.session_state.groq_api_key = key.strip()
                st.session_state.page = "main"
                st.rerun()
            else:
                st.error("⚠️ Please enter a valid Groq API key!")
        st.markdown("---")
        st.markdown("**Built using:**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("- 📄 IMD Annual Reports 2008–2024")
            st.markdown("- 🤗 Fine-tuned SBERT + DeBERTa")
        with col_b:
            st.markdown("- 🌐 Open-Meteo Live API")
            st.markdown("- 🦙 Groq Llama 3")

@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sooriyajs/imd-sbert-finetuned")
    nli_model = pipeline(
        "zero-shot-classification",
        model="sooriyajs/imd-deberta-finetuned",
        device=-1
    )
    chunks_path = hf_hub_download(
        repo_id="sooriyajs/imd-climate-data",
        filename="all_chunks.pkl",
        repo_type="dataset"
    )
    with open(chunks_path, "rb") as f:
        all_chunks = pickle.load(f)
    faiss_path = hf_hub_download(
        repo_id="sooriyajs/imd-climate-data",
        filename="imd_faiss.index",
        repo_type="dataset"
    )
    index = faiss.read_index(faiss_path)
    return embedder, nli_model, all_chunks, index

def formalize_claim(raw_claim, client):
    prompt = f"""Convert this user input into a clear formal factual claim.
Also extract: year mentioned (or null), location (or India), topic.
User input: "{raw_claim}"
Respond ONLY in JSON:
{{"formal_claim": "...", "year": "...", "location": "...", "topic": "..."}}"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    text = response.choices[0].message.content.strip()
    try:
        text = re.sub(r"```json|```", "", text).strip()
        return json.loads(text)
    except:
        return {"formal_claim": raw_claim, "year": None,
                "location": "India", "topic": "other"}

def retrieve_evidence(claim, embedder, index, all_chunks, year_filter=None, top_k=3):
    claim_vec = embedder.encode([claim], convert_to_numpy=True)
    D, I = index.search(claim_vec, top_k * 3)
    results = []
    for idx in I[0]:
        if idx < len(all_chunks):
            chunk = all_chunks[idx]
            if year_filter and chunk["year"] != str(year_filter):
                continue
            results.append(chunk)
            if len(results) == top_k:
                break
    if not results:
        for idx in I[0][:top_k]:
            if idx < len(all_chunks):
                results.append(all_chunks[idx])
    return results

def fetch_climate_data(year, location="India"):
    coords = {
        "India":   (20.5937, 78.9629),
        "Delhi":   (28.6139, 77.2090),
        "Mumbai":  (19.0760, 72.8777),
        "Chennai": (13.0827, 80.2707),
    }
    lat, lon = coords.get(location, coords["India"])
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "Asia/Kolkata"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        d = r.json().get("daily", {})
        temps_max = [t for t in d.get("temperature_2m_max", []) if t is not None]
        temps_min = [t for t in d.get("temperature_2m_min", []) if t is not None]
        rain = [p for p in d.get("precipitation_sum", []) if p is not None]
        return {
            "year": year,
            "avg_max_temp": round(sum(temps_max)/len(temps_max), 2) if temps_max else None,
            "avg_min_temp": round(sum(temps_min)/len(temps_min), 2) if temps_min else None,
            "total_rain_mm": round(sum(rain), 2) if rain else None,
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_nli_verdict(claim, evidence_text, nli_model):
    result = nli_model(
        evidence_text[:512],
        candidate_labels=["entailment", "contradiction", "neutral"],
        hypothesis_template="This text supports the claim: {}"
    )
    verdict_map = dict(zip(result["labels"], result["scores"]))
    ent = verdict_map.get("entailment", 0)
    con = verdict_map.get("contradiction", 0)
    neu = verdict_map.get("neutral", 0)
    if ent > 0.5: return "TRUE", ent
    if con > 0.5: return "FAKE", con
    return "UNCERTAIN", neu

def multi_vote(claim, evidence_chunks, nli_model):
    verdicts = []
    for chunk in evidence_chunks:
        v, score = get_nli_verdict(claim, chunk["text"], nli_model)
        verdicts.append((v, score, chunk))
    counts = {"TRUE": 0, "FAKE": 0, "UNCERTAIN": 0}
    for v, s, _ in verdicts:
        counts[v] += 1
    final = max(counts, key=counts.get)
    confidence = counts[final] / len(verdicts)
    return final, confidence, verdicts

def generate_explanation(claim, verdict, evidence_chunks, client, api_data=None):
    evidence_text = "\n".join([f"- [{c['year']}] {c['text'][:200]}" for c in evidence_chunks])
    api_summary = ""
    if api_data and api_data.get("status") == "success":
        api_summary = f"""
Live API Data ({api_data['year']}):
- Avg Max Temp: {api_data['avg_max_temp']}C
- Avg Min Temp: {api_data['avg_min_temp']}C
- Total Rainfall: {api_data['total_rain_mm']} mm"""
    prompt = f"""A user made this climate claim: "{claim}"
Our system verdict: {verdict}
Evidence from IMD reports:
{evidence_text}
{api_summary}
Write a short 2-3 sentence explanation. Be factual and cite the data."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def claim_verification_page(client):
    st.title("🔍 Claim Verification")
    st.markdown("Check any climate claim about India against IMD reports (2008–2024) + live weather data!")
    embedder, nli_model, all_chunks, index = load_models()
    st.markdown("### 💡 Try these examples:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("2020 was hottest year in India"):
            st.session_state.claim = "2020 was hottest year in India"
        if st.button("Cyclone Amphan hit West Bengal 2020"):
            st.session_state.claim = "Cyclone Amphan hit West Bengal 2020"
    with col2:
        if st.button("India had no rainfall in 2019"):
            st.session_state.claim = "India had no rainfall in 2019"
        if st.button("2015 had worst heatwave in India"):
            st.session_state.claim = "2015 had worst heatwave in India"
    claim = st.text_area(
        "Enter your climate claim:",
        value=st.session_state.get("claim", ""),
        placeholder="e.g. 2020 was the hottest year in India",
        height=100
    )
    if st.button("✅ Check Claim", type="primary"):
        if claim.strip():
            with st.spinner("Analyzing claim..."):
                meta = formalize_claim(claim, client)
                formal = meta.get("formal_claim", claim)
                year = meta.get("year")
                location = meta.get("location", "India")
                evidence = retrieve_evidence(formal, embedder, index, all_chunks, year_filter=year)
                api_data = fetch_climate_data(year, location) if year else None
                final_verdict, confidence, verdicts = multi_vote(formal, evidence, nli_model)
                if confidence < 0.5:
                    final_verdict = "UNCERTAIN"
                explanation = generate_explanation(formal, final_verdict, evidence, client, api_data)
                st.markdown("---")
                emoji = {"TRUE": "✅", "FAKE": "❌", "UNCERTAIN": "⚠️"}.get(final_verdict, "❓")
                color = {"TRUE": "green", "FAKE": "red", "UNCERTAIN": "orange"}.get(final_verdict, "gray")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"### {emoji} Verdict: :{color}[{final_verdict}]")
                    st.markdown(f"**Confidence:** {confidence:.0%}")
                    st.markdown(f"**Formal claim:** {formal}")
                with col2:
                    st.markdown("### 📝 Explanation")
                    st.write(explanation)
                st.markdown("---")
                st.markdown("### 📚 Evidence Used")
                for v, s, chunk in verdicts:
                    with st.expander(f"[{chunk['year']}] {v} ({s:.0%})"):
                        st.write(chunk["text"][:300])
                if api_data and api_data.get("status") == "success":
                    st.markdown("---")
                    st.markdown("### 🌡️ Live Climate Data")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Avg Max Temp", f"{api_data['avg_max_temp']}°C")
                    c2.metric("Avg Min Temp", f"{api_data['avg_min_temp']}°C")
                    c3.metric("Total Rainfall", f"{api_data['total_rain_mm']} mm")
                st.session_state.history.append({
                    "claim": claim,
                    "verdict": final_verdict,
                    "time": datetime.now().strftime("%H:%M:%S")
                })
        else:
            st.error("Please enter a claim!")
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 🕓 Verdict History")
        for h in reversed(st.session_state.history[-10:]):
            emoji = {"TRUE": "✅", "FAKE": "❌", "UNCERTAIN": "⚠️"}.get(h["verdict"], "❓")
            st.markdown(f"{emoji} `[{h['time']}]` **{h['verdict']}** — {h['claim'][:60]}")
    st.markdown("---")
    st.markdown("### 💬 Was this result helpful?")
    rating = st.slider("⭐ Rate this result", 1, 5, 3, key="quick_rating")
    stars = "⭐" * rating
    st.markdown(f"Your rating: **{stars} ({rating}/5)**")
    comment = st.text_input(
        "Any comments?",
        placeholder="Was the verdict accurate? Any suggestions?",
        key="quick_comment"
    )
    if st.button("📨 Submit Feedback", key="quick_feedback_btn", type="primary"):
        if comment.strip() or rating:
            with open("feedback.txt", "a") as f:
                f.write(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Rating: {rating}/5 | Comment: {comment}\n"
                )
            st.success("🙏 Thanks for your feedback!")
            st.balloons()
        else:
            st.error("Please write a comment before submitting!")

def feedback_page():
    st.title("💬 Feedback")
    st.markdown("We'd love to hear your thoughts on the IMD Climate Fake News Detector!")
    st.markdown("---")
    name = st.text_input("Your Name (optional)", placeholder="e.g. Durga")
    rating = st.slider("⭐ Rate this app", 1, 5, 3, help="1 = Poor, 5 = Excellent")
    stars = "⭐" * rating
    st.markdown(f"Your rating: **{stars} ({rating}/5)**")
    category = st.selectbox("What is your feedback about?", [
        "Overall Experience",
        "Claim Verification Accuracy",
        "UI / Design",
        "Speed / Performance",
        "Other"
    ])
    comment = st.text_area("Your Comments", placeholder="Tell us what you think...", height=150)
    if st.button("📨 Submit Feedback", type="primary"):
        if comment.strip() or rating:
            feedback_entry = (
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Name: {name or 'Anonymous'} | "
                f"Rating: {rating}/5 | "
                f"Category: {category} | "
                f"Comment: {comment}\n"
            )
            with open("feedback.txt", "a") as f:
                f.write(feedback_entry)
            st.success("🙏 Thank you for your feedback!")
            st.balloons()
        else:
            st.error("Please write a comment or give a rating before submitting!")
    if os.path.exists("feedback.txt"):
        with open("feedback.txt", "r") as f:
            lines = f.readlines()
        st.markdown(f"---\n*{len(lines)} people have submitted feedback so far!*")

def main():
    if not st.session_state.groq_api_key:
        api_key_page()
        return
    client = Groq(api_key=st.session_state.groq_api_key)
    with st.sidebar:
        st.markdown("## 🌦️ IMD Detector")
        st.markdown("---")
        page = st.radio("Navigate", ["🔍 Claim Verification", "💬 Feedback"])
        st.markdown("---")
        st.markdown("### About")
        st.markdown("Built using:")
        st.markdown("- IMD Annual Reports 2008–2024")
        st.markdown("- Fine-tuned SBERT + DeBERTa")
        st.markdown("- Open-Meteo Live API")
        st.markdown("- Groq Llama 3")
        st.markdown("---")
        if st.button("🔄 Change API Key"):
            st.session_state.groq_api_key = ""
            st.session_state.page = "api_key"
            st.rerun()
    if page == "🔍 Claim Verification":
        claim_verification_page(client)
    elif page == "💬 Feedback":
        feedback_page()

if __name__ == "__main__":
    main()
