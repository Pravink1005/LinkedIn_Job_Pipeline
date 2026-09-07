import os
import re
import time
import random
import hashlib
import csv
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import joblib
from scrapling import Fetcher, StealthyFetcher
from ml_predictor import predict_job_details

# ============================================================
# LOAD TRAINED ML ARTIFACTS
# ============================================================

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEGREE_VECTORIZER_FILE = os.path.join(MODEL_DIR, "degree_vectorizer.pkl")
DEGREE_MODEL_FILE = os.path.join(MODEL_DIR, "degree_model.pkl")
SPEC_VECTORIZER_FILE = os.path.join(MODEL_DIR, "specialization_vectorizer.pkl")
SPEC_MODEL_FILE = os.path.join(MODEL_DIR, "specialization_model.pkl")

try:
    VECTORIZER_DEGREE = joblib.load(DEGREE_VECTORIZER_FILE)
    MODEL_DEGREE = joblib.load(DEGREE_MODEL_FILE)
    VECTORIZER_SPEC = joblib.load(SPEC_VECTORIZER_FILE)
    MODEL_SPEC = joblib.load(SPEC_MODEL_FILE)
    print("[ML Models] Successfully loaded degree and specialization models.")
except Exception as e:
    print(f"[ML Models Error] Failed to load models: {e}")
    VECTORIZER_DEGREE, MODEL_DEGREE = None, None
    VECTORIZER_SPEC, MODEL_SPEC = None, None

# ============================================================
# USER-AGENT ROTATION POOL
# ============================================================

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]

def get_random_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

# ============================================================
# PERSISTENT DEDUPLICATION TRACKER
# ============================================================

CSV_OUTPUT_DIR = "csv_output"
CURRENT_JOBS_CSV = os.path.join(CSV_OUTPUT_DIR, "current_jobs.csv")
SEEN_JOBS_FILE = os.path.join(CSV_OUTPUT_DIR, "seen_job_ids.json")

def load_seen_jobs():
    if os.path.exists(SEEN_JOBS_FILE):
        try:
            with open(SEEN_JOBS_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()

def save_seen_jobs(seen_set):
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
    with open(SEEN_JOBS_FILE, "w", encoding="utf-8") as f:
        json.dump(list(seen_set), f, indent=2)

# ============================================================
# FEATURE ENGINEERING & QUALIFICATION EXTRACTION
# ============================================================

SKILL_PATTERNS = {
    "Python": r"\bpython\b",
    "SQL": r"\bsql\b|\bpostgresql\b|\bmysql\b",
    "R": r"(?:\b|_)r(?:-lang|lang)?(?:\b|_)",
    "Java": r"\bjava\b",
    "Spark": r"\bspark\b|\bpyspark\b",
    "Airflow": r"\bairflow\b",
    "AWS": r"\baws\b|\bamazon web services\b",
    "Azure": r"\bazure\b",
    "GCP": r"\bgcp\b|\bgoogle cloud\b",
    "Power BI": r"\bpower bi\b|\bpowerbi\b",
    "Tableau": r"\btableau\b",
}

def clean_text_for_ml(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s\.-]", " ", text)
    return " ".join(text.split())

def extract_skills(text):
    if not text or text == "N/A":
        return "Not Specified"
    found = [skill for skill, pattern in SKILL_PATTERNS.items() if re.search(pattern, text.lower())]
    return ", ".join(found) if found else "Not Specified"

def extract_experience_years(text):
    if not text or text == "N/A":
        return "Not Specified", "Not Specified"
    patterns = [
        r"(\d+)\s*(?:to|-|–)\s*(\d+)\s*(?:years?|yrs?)",       # e.g., "8-12 years"
        r"(\d+)\+\s*(?:years?|yrs?)",                         # e.g., "5+ years"
        r"(\d+)\s*(?:years?|yrs?)\s+of\s+experience",         # e.g., "4 years of experience"
        r"(?:at least|minimum|min)\s*(\d+)\s*(?:years?|yrs?)", # e.g., "minimum 3 years"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) == 2 and groups[1]:
                return groups[0], groups[1]
            elif len(groups) >= 1 and groups[0]:
                return groups[0], f"{groups[0]}+"
    return "Not Specified", "Not Specified"

def extract_qualification_text(text):
    if not text or text == "N/A":
        return text

    # Step 1: Search for Minimum / Basic / Broad Qualifications
    min_pattern = r"(?:minimum qualifications|basic qualifications|who you are and what you bring|who you are|qualifications|requirements|candidate profile)[:\s]+(.*?)(?=(?:preferred qualifications|preferred|responsibilities|about the job|what you will do|$))"
    min_match = re.search(min_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if min_match and len(min_match.group(1).strip()) > 15:
        return min_match.group(1).strip()

    # Step 2: Fallback to Preferred Qualifications if Minimum is missing
    pref_pattern = r"(?:preferred qualifications|preferred)[:\s]+(.*?)(?=(?:responsibilities|about the job|what you will do|$))"
    pref_match = re.search(pref_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if pref_match and len(pref_match.group(1).strip()) > 15:
        return pref_match.group(1).strip()

    # Step 3: Fallback to full description
    return text

def extract_specialization_text(text):
    """Remove degree-focused lines so education fields do not become job domains."""
    if not text or text == "N/A":
        return text

    lines = re.split(r"\r?\n+|(?<=[.!?])\s+", text)
    degree_line_pattern = (
        r"\b(?:degree|qualification|education|bachelor|master|phd|doctorate|"
        r"undergraduate|postgraduate|academic|major|field of study)\b"
    )
    return " ".join(
        line.strip()
        for line in lines
        if line.strip() and not re.search(degree_line_pattern, line, re.IGNORECASE)
    )

def predict_education_ml(text, min_confidence=0.20):
    """
    Hybrid Prediction Logic:
    1. Parses Minimum/Preferred sections.
    2. Applies Regex Rules for broad statements ("Any Bachelor's", "Any Master's").
    3. Uses ML TF-IDF models as a fallback if no direct rules match.
    """
    predictions = predict_job_details(text)
    return predictions["predicted_degree"], predictions["predicted_specialization"]

    if not text or text == "N/A":
        return "Not Specified", "Not Specified"
    
    target_qual_text = extract_qualification_text(text)
    target_qual_text = target_qual_text.replace("’", "'").replace("‘", "'")
    specialization_text = extract_specialization_text(text)
    cleaned_degree_text = clean_text_for_ml(target_qual_text)
    cleaned_specialization_text = clean_text_for_ml(specialization_text)
    
    rule_degree = None
    rule_spec = None
    
    # -------------------------------------------------------------
    # RULE-BASED MATCHING FOR BROAD DEGREE STATEMENTS
    # -------------------------------------------------------------
    # Catch: "Any Master's", "Master's degree or higher", "Postgraduate"
    if re.search(r"\bany\s+master(?:'s)?\b|\bmaster(?:'s)?\s+degree\b|\bpost\s*graduat(?:e|ion)\b", target_qual_text, re.I):
        rule_degree = "Master's Degree"
        
    # Catch: "Any Bachelor's", "Bachelor's degree or equivalent", "Graduate"
    elif re.search(
        r"\bany\s+bachelor(?:'s)?\b|\bbachelor(?:'s)?\s+(?:or\s+associate\s+)?degree\b|\bunder\s*graduat(?:e|ion)\b",
        target_qual_text,
        re.I,
    ):
        rule_degree = "Bachelor's Degree"
        
    # Catch: PhD requirements
    elif re.search(r"\bphd\b|\bdoctorate\b", target_qual_text, re.I):
        rule_degree = "PhD"

    # -------------------------------------------------------------
    # RULE-BASED MATCHING FOR SPECIALIZATIONS / DOMAINS
    # -------------------------------------------------------------
    if re.search(
        r"\bbusiness\s+analytics\b|\bdata\s+analytics\b|\bresearch\s+analyst\b.*\b(?:business\s+)?operations?\b|\b(?:business\s+)?operations?\b.*\bresearch\s+analyst\b",
        specialization_text,
        re.I | re.DOTALL,
    ):
        rule_spec = "Business Analytics"

    # -------------------------------------------------------------
    # ML PREDICTION (FALLBACK)
    # -------------------------------------------------------------
    try:
        # --- DEGREE PREDICTION ---
        if not rule_degree:
            degree_vec = VECTORIZER_DEGREE.transform([cleaned_degree_text])
            deg_probs = MODEL_DEGREE.predict_proba(degree_vec)[0]
            max_deg_prob = np.max(deg_probs)
            top_deg_class = MODEL_DEGREE.classes_[np.argmax(deg_probs)]
            
            if max_deg_prob >= min_confidence:
                rule_degree = top_deg_class
            else:
                rule_degree = "Certification / Not Specified"

        # --- SPECIALIZATION PREDICTION ---
        if not rule_spec:
            specialization_vec = VECTORIZER_SPEC.transform([cleaned_specialization_text])
            rule_spec = MODEL_SPEC.predict(specialization_vec)[0]

        return rule_degree, rule_spec

    except Exception as e:
        print(f"[ML Prediction Error]: {e}")
        return rule_degree or "Not Specified", rule_spec or "Not Specified"
    
# ============================================================
# LINKEDIN URL HELPERS
# ============================================================

def normalize_linkedin_job_url(url):
    if not url:
        return "N/A"
    match = re.search(r"/jobs/view/(\d{6,})(?:[/?#]|$)", str(url), flags=re.I)
    if match:
        return f"https://www.linkedin.com/jobs/view/{match.group(1)}/"
    return url.split("?")[0].split("#")[0].rstrip("/") + "/"

def get_canonical_job_id(url):
    normalized = normalize_linkedin_job_url(url)
    match = re.search(r"/jobs/view/(\d+)(?:/|$)", normalized, flags=re.I)
    if match:
        return f"linkedin_{match.group(1)}"
    return f"linkedin_unknown_{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:15]}"

# ============================================================
# HTTP FETCH & WORKER PIPELINE
# ============================================================

def safe_fetch_get(url, is_stealth=False, max_retries=3):
    for attempt in range(max_retries):
        try:
            headers = get_random_headers()
            if is_stealth:
                response = StealthyFetcher.fetch(url, headless=True, network_idle=True, headers=headers)
            else:
                response = Fetcher.get(url, headers=headers)
            if getattr(response, "status", 200) == 429:
                time.sleep(2.0 * (2 ** attempt))
                continue
            return response
        except Exception:
            time.sleep(1.5)
    return None

def fetch_single_job_hybrid(job):
    full_description = ""
    company = job.get("company", "")
    
    # 1. Fetch Job Details from LinkedIn
    try:
        time.sleep(random.uniform(1.0, 2.5))
        detail_response = safe_fetch_get(job["link"], is_stealth=False)
        if detail_response:
            container = detail_response.css(".show-more-less-html__markup") or detail_response.css(".description__text")
            if container:
                full_description = " ".join(str(t).strip() for t in container.css("::text").getall() if str(t).strip())
    except Exception as e:
        print(f"Error fetching job {job.get('job_id')}: {e}")

    if not full_description.strip():
        return None

    # 2. Feature Engineering & ML Predictions
    skills = extract_skills(full_description)
    min_exp, max_exp = extract_experience_years(full_description)
    predictions = predict_job_details(full_description)
    pred_degree = predictions["predicted_degree"]
    pred_spec = predictions["predicted_specialization"]

    return {
        "job_id": job.get("job_id"),
        "category": job.get("category", ""),
        "title": job.get("title", ""),
        "company": company or "N/A",
        "city": job.get("city", ""),
        "state": job.get("state", ""),
        "country": job.get("country", ""),
        "source": "LinkedIn",
        "timestamp": job.get("timestamp", ""),
        "link": job.get("link", ""),
        "full_description": full_description or "N/A",
        "skills": skills,
        "degree_required": pred_degree,
        "specialization_required": pred_spec,
        "min_experience_years": min_exp,
        "max_experience_years": max_exp,
    }

# ============================================================
# CSV EXPORT
# ============================================================

CSV_FIELDS = [
    "Job ID",
    "Category",
    "Job Title",
    "Company",
    "City",
    "State",
    "Country",
    "Source",
    "Extracted Skills",
    "Degree Required",
    "Specialization Required",
    "Min Exp (Years)",
    "Max Exp (Years)",
    "Posted Time",
    "Job Link",
    "Full Description",
]

def export_current_jobs_to_csv(jobs):
    if not jobs:
        print("\nNo new jobs to append.")
        return False

    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
    file_exists = os.path.exists(CURRENT_JOBS_CSV)

    with open(CURRENT_JOBS_CSV, "a", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists or os.stat(CURRENT_JOBS_CSV).st_size == 0:
            writer.writeheader()

        for job in jobs:
            writer.writerow({
                "Job ID": job.get("job_id", ""),
                "Category": job.get("category", ""),
                "Job Title": job.get("title", ""),
                "Company": job.get("company", ""),
                "City": job.get("city", ""),
                "State": job.get("state", ""),
                "Country": job.get("country", ""),
                "Source": job.get("source", "LinkedIn"),
                "Extracted Skills": job.get("skills", ""),
                "Degree Required": job.get("degree_required", ""),
                "Specialization Required": job.get("specialization_required", ""),
                "Min Exp (Years)": job.get("min_experience_years", ""),
                "Max Exp (Years)": job.get("max_experience_years", ""),
                "Posted Time": job.get("timestamp", ""),
                "Job Link": job.get("link", ""),
                "Full Description": job.get("full_description", ""),
            })

    print(f"\n[Pipeline Complete] Appended {len(jobs)} jobs with ML classifications to: {CURRENT_JOBS_CSV}")
    return True

# ============================================================
# MAIN EXECUTION PIPELINE
# ============================================================

if __name__ == "__main__":
    search_keywords = ["Data Analyst", "Data Engineer", "Data Science"]
    location = "India"
    jobs_buffer = []
    seen_listings = load_seen_jobs()
    queued_job_ids = set()

    print("[Pipeline Started] Searching LinkedIn postings...")

    for keyword in search_keywords:
        formatted_keyword = keyword.replace(" ", "%20")
        formatted_location = location.replace(" ", "%20")
        search_url = f"https://www.linkedin.com/jobs/search/?keywords={formatted_keyword}&location={formatted_location}&f_TPR=r3600&start=0"
        
        response = safe_fetch_get(search_url, is_stealth=False)
        if not response:
            continue

        job_cards = response.css(".job-search-card")
        for card in job_cards[:5]:  # Limit cards per query during test run
            link_el = card.css("a.base-card__full-link")
            if not link_el:
                continue
            
            link = normalize_linkedin_job_url(link_el[0].attrib.get("href", ""))
            job_id = get_canonical_job_id(link)
            
            if not job_id or job_id in seen_listings or job_id in queued_job_ids:
                continue

            queued_job_ids.add(job_id)
            jobs_buffer.append({
                "job_id": job_id,
                "category": keyword,
                "title": card.css(".base-search-card__title::text").get(default="").strip(),
                "company": card.css("h4.base-search-card__subtitle a::text").get(default="").strip(),
                "city": "India",
                "state": "",
                "country": "India",
                "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M"),
                "link": link,
            })

    # Multithreaded Processing
    processed_jobs = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(fetch_single_job_hybrid, job): job for job in jobs_buffer}
        for future in as_completed(futures):
            try:
                res = future.result()
                if res is not None:
                    processed_jobs.append(res)
            except Exception as exc:
                print(f"Error processing job thread: {exc}")

    # Export to CSV and save deduplication state
    if export_current_jobs_to_csv(processed_jobs):
        seen_listings.update(job["job_id"] for job in processed_jobs)
        save_seen_jobs(seen_listings)