from pathlib import Path
import re
import warnings
import joblib


warnings.filterwarnings("ignore", category=Warning)


# ==============================
# MODEL FOLDER PATH
# ==============================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# ==============================
# SAFELY LOAD MODELS
# ==============================

def safe_load_model(path):
    try:
        return joblib.load(path)
    except Exception:
        return None


degree_vectorizer = safe_load_model(MODEL_DIR / "degree_vectorizer.pkl")
degree_model = safe_load_model(MODEL_DIR / "degree_model.pkl")
specialization_vectorizer = safe_load_model(MODEL_DIR / "specialization_vectorizer.pkl")
specialization_model = safe_load_model(MODEL_DIR / "specialization_model.pkl")


# ==============================
# EXPLICIT DEGREE PATTERNS
# ==============================

DEGREE_PATTERNS = {
    "B.E/B.Tech": [
        r"\bB\.\s*E\.?\b",
        r"\bB\.?\s*Tech\.?\b",
        r"\bBachelor\s+of\s+Engineering\b",
        r"\bBachelor\s+of\s+Technology\b",
    ],
    "B.Sc": [
        r"\bB\.?\s*Sc\.?\b",
        r"\bBachelor\s+of\s+Science\b",
        r"\bBS\b",
    ],
    "BCA": [
        r"\bBCA\b",
        r"\bBachelor\s+of\s+Computer\s+Applications\b",
    ],
    "B.Com": [
        r"\bB\.?\s*Com\.?\b",
        r"\bBachelor\s+of\s+Commerce\b",
    ],
    "BBA": [
        r"\bBBA\b",
        r"\bBachelor\s+of\s+Business\s+Administration\b",
    ],
    "M.E/M.Tech": [
        r"\bM\.\s*E\.?\b",
        r"\bM\.?\s*Tech\.?\b",
        r"\bMaster\s+of\s+Engineering\b",
        r"\bMaster\s+of\s+Technology\b",
    ],
    "M.Sc": [
        r"\bM\.?\s*Sc\.?\b",
        r"\bMaster\s+of\s+Science\b",
        r"\bMS\b",
    ],
    "MCA": [
        r"\bMCA\b",
        r"\bMaster\s+of\s+Computer\s+Applications\b",
    ],
    "MBA": [
        r"\bMBA\b",
        r"\bMaster\s+of\s+Business\s+Administration\b",
    ],
    "PhD": [
        r"\bPh\.?\s*D\.?\b",
        r"\bPhD\b",
        r"\bDoctorate\b",
    ],
    "Diploma": [
        r"\bDiploma\b",
        r"\bEngineering\s+Diploma\b",
    ],
    "Any Bachelor's Degree": [
        r"\bBachelor'?s\s+or\s+Master'?s\s+degree\b",
        r"\bBachelor'?s\s+or\s+associate\s+degree\b",
        r"\bBachelor'?s\s+degree\b",
        r"\bBachelor'?s\s+qualification\b",
        r"\bAny\s+recognized\s+Bachelor'?s\s+degree\b",
    ],
    "Any Master's Degree": [
        r"\bMaster'?s\s+degree\b",
        r"\bMaster'?s\s+qualification\b",
        r"\bAny\s+recognized\s+Master'?s\s+degree\b",
    ],
}


# ==============================
# EXPLICIT SPECIALIZATION PATTERNS
# ==============================

SPECIALIZATION_PATTERNS = [
    ("AI/ML", [
        r"\bAI\s*/\s*ML\b",
        r"\b(?:AI|ML)\s*(?:engineer|developer|scientist|specialist|role|team|model|pipeline|architect)\b",
        r"\b(?:artificial\s+intelligence|machine\s+learning|deep\s+learning|computer\s+vision|natural\s+language\s+processing|generative\s+AI|LLM|large\s+language\s+model)\b",
        r"\b(?:artificial\s+intelligence|machine\s+learning|deep\s+learning|computer\s+vision|natural\s+language\s+processing|generative\s+AI|LLM)\s+(?:engineer|developer|scientist|specialist|role|team)\b",
        r"\b(?:PyTorch|TensorFlow|scikit-learn|Keras|Hugging\s+Face|LangChain|OpenCV|transformers)\b",
    ]),
    ("Data Science", [
        r"\bdata\s+scientist\b",
        r"\bdata\s+science\b",
        r"\banalytics\b",
        r"\bstatistical\s+analysis\b",
        r"\bexperimental\s+design\b",
        r"\bpredictive\s+modeling\b",
        r"\bforecasting\b",
        r"\bpower\s+bi\b",
        r"\bpython\s+and\s+sql\b",
        r"\bpython\s*[,\)]?\s*and\s*sql\b",
        r"\bSQL\s+and\s+Python\b",
    ]),
    ("Data Engineering", [
        r"\bdata engineering\b",
        r"\bdata engineer\b",
        r"\betl\b",
        r"\bdata pipelines\b",
        r"\bdbt\b",
        r"\bAzure\s+Data\s+Factory\b",
        r"\bKusto\b",
        r"\bdata\s+warehouse\b",
        r"\bstreaming\s+data\b",
    ]),
    ("IoT", [
        r"\bInternet\s+of\s+Things\b",
        r"\bIoT\b",
        r"\bIOT\b",
        r"\bembedded\s+systems\b",
        r"\bsmart\s+devices\b",
        r"\bsensor\s+networks\b",
        r"\bindustrial\s+IoT\b",
        r"\bedge\s+computing\b",
        r"\bdevice\s+integration\b",
    ]),
    ("Software Engineering", [
        r"\bsoftware engineering\b",
        r"\bsoftware engineer\b",
        r"\bfull stack\b",
        r"\bbackend\b",
        r"\bfrontend\b",
        r"\bJavaScript\b",
        r"\bPython\b",
        r"\bAPI\s+development\b",
        r"\bweb\s+development\b",
    ]),
    ("Computer Science", [
        r"\bcomputer science\b",
        r"\bCS\b",
        r"\bsoftware development\b",
        r"\balgorithm\b",
        r"\bdata structures\b",
        r"\bcomputer\s+applications\b",
    ]),
    ("Cyber Security", [
        r"\bcyber security\b",
        r"\bsecurity engineering\b",
        r"\bcybersecurity\b",
        r"\bnetwork security\b",
        r"\binformation\s+security\b",
    ]),
    ("Electrical Engineering", [
        r"\belectrical engineering\b",
        r"\belectronics\b",
        r"\bpower systems\b",
        r"\bcontrol systems\b",
        r"\bembedded\s+hardware\b",
    ]),
    ("Mechanical Engineering", [
        r"\bmechanical engineering\b",
        r"\bmanufacturing\b",
        r"\bindustrial engineering\b",
        r"\bproduction\s+engineering\b",
    ]),
    ("Finance", [
        r"\bfinance\b",
        r"\bfinancial\b",
        r"\bfinancial\s+services\b",
        r"\bfinancial\s+data\b",
        r"\binvestment\s+management\b",
        r"\binvestment\s+industry\b",
    ]),
    ("Business Administration", [
        r"\bbusiness administration\b",
        r"\bMBA\b",
        r"\boperations\b",
        r"\bstrategy\b",
        r"\bfinance\s+analyst\b",
    ]),
]


# ==============================
# HELPER FUNCTIONS
# ==============================

def detect_explicit_degree(job_description):
    if not job_description or not job_description.strip():
        return []

    normalized_description = job_description.replace("’", "'").replace("‘", "'")
    found = []
    for degree, patterns in DEGREE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, normalized_description, re.IGNORECASE):
                found.append(degree)
                break
    return list(dict.fromkeys(found))


GENERIC_AI_EXCLUDES = [
    "ai tools",
    "use ai tools",
    "ai-assisted",
    "artificial intelligence (ai) tools",
    "we may use artificial intelligence (ai) tools",
    "ai included",
    "ai tool",
    "ai-powered",
    "ai-driven",
]


def extract_specialization_text(job_description):
    if not job_description or not job_description.strip():
        return ""

    sentences = re.split(r"\r?\n+|(?<=[.!?])\s+", job_description)
    degree_pattern = re.compile(
        r"\b(?:degree|qualification|education|bachelor|master|phd|doctorate|"
        r"undergraduate|postgraduate|academic|major|field of study)\b",
        re.IGNORECASE,
    )
    return " ".join(sentence.strip() for sentence in sentences if sentence.strip() and not degree_pattern.search(sentence))


def detect_explicit_specialization(job_description):
    if not job_description or not job_description.strip():
        return []

    cleaned = extract_specialization_text(job_description).lower()
    for phrase in GENERIC_AI_EXCLUDES:
        cleaned = cleaned.replace(phrase, " ")

    if re.search(r"\blooking\s+for\s+(?:a\s+)?data\s+analyst\b", cleaned):
        return ["Data Analyst"]

    if re.search(r"\bdata\s+engineering\b|\bdata\s+engineer\b", cleaned):
        return ["Data Engineering"]

    if re.search(r"\bresearch\s+analyst\b", cleaned) and re.search(
        r"\b(?:business\s+)?operations?\b|\bdata\s+analysis\b|\bdata\s+modeling\b",
        cleaned,
    ):
        return ["Business Analytics"]

    if re.search(
        r"\bdata\s+pipelines?\b|\b(?:PySpark|Spark\s+SQL)\b|\bdata\s+warehous(?:e|ing)\b|\bELT\b",
        cleaned,
        re.IGNORECASE,
    ):
        return ["Data Engineering"]

    scores = []
    for idx, (specialization, patterns) in enumerate(SPECIALIZATION_PATTERNS):
        score = 0
        for pattern in patterns:
            if re.search(pattern, cleaned, re.IGNORECASE):
                score += 1
        if score > 0:
            scores.append((specialization, score, idx))

    if not scores:
        return []

    scores.sort(key=lambda item: (-item[1], item[2]))
    return [specialization for specialization, _, _ in scores]


# ==============================
# DEGREE PREDICTION
# ==============================

def predict_degree(job_description):
    if not job_description or not job_description.strip():
        return "Not Specified"

    explicit = detect_explicit_degree(job_description)
    if explicit:
        return " / ".join(explicit)

    if degree_model is not None and degree_vectorizer is not None:
        try:
            text_vector = degree_vectorizer.transform([job_description])
            probabilities = degree_model.predict_proba(text_vector)[0]
            best_index = probabilities.argmax()
            predicted_degree = str(degree_model.classes_[best_index])
            confidence = probabilities[best_index] * 100
            if confidence >= 60:
                return predicted_degree
        except Exception:
            pass

    return "Not Specified"


# ==============================
# SPECIALIZATION PREDICTION
# ==============================

def predict_specialization(job_description):
    if not job_description or not job_description.strip():
        return "Not Specified"

    explicit = detect_explicit_specialization(job_description)
    if explicit:
        return explicit[0]

    if specialization_model is not None and specialization_vectorizer is not None:
        try:
            cleaned_description = extract_specialization_text(job_description)
            text_vector = specialization_vectorizer.transform([cleaned_description])
            probabilities = specialization_model.predict_proba(text_vector)[0]
            best_index = probabilities.argmax()
            predicted_specialization = str(specialization_model.classes_[best_index])
            confidence = probabilities[best_index] * 100
            if confidence >= 60:
                return predicted_specialization
        except Exception:
            pass

    return "Not Specified"


# ==============================
# COMPLETE JOB PREDICTION
# ==============================

def predict_job_details(job_description):
    degree = predict_degree(job_description)
    specialization = predict_specialization(job_description)

    return {
        "predicted_degree": degree,
        "predicted_specialization": specialization,
    }