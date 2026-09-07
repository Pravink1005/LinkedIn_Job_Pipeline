# LinkedIn Job Pipeline

A Python pipeline that collects recent LinkedIn job postings and extracts:

- Required degree
- Job specialization
- Technical and professional skills
- Minimum and maximum experience
- Job metadata and full description

The project uses TF-IDF and Logistic Regression models for degree and specialization prediction, plus rule-based extraction for skills and experience.

## Project Structure

```text
Main.py                              LinkedIn scraping and CSV export pipeline
ml_predictor.py                      Shared prediction and extraction logic
test_ml.py                           Quick local prediction test
train_models.py                      Retrain aligned ML models
ML_Model.ipynb                       Original model-training notebook
Degree_Prediction_Training_5000_Corrected.csv
                                     Training dataset
models/                              Saved model artifacts
csv_output/                          Scraped CSV and deduplication state
requirements.txt                     Python dependencies
```

## Requirements

- Windows, macOS, or Linux
- Python 3.10+
- Internet access for LinkedIn scraping

The saved models were trained with scikit-learn `1.6.1`. Keep that version installed to avoid model compatibility warnings.

## Setup on Windows

Open PowerShell in the project folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the commands directly with the virtual-environment interpreter:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Test Predictions

Edit the `job_description` value in `test_ml.py`, then run:

```powershell
.venv\Scripts\python.exe test_ml.py
```

The script prints:

- Predicted degree
- Predicted specialization
- Extracted skills
- Categorized skills with evidence
- Minimum and maximum experience

Example output:

```text
Predicted Degree: Any Bachelor's Degree
Predicted Specialization: Software Engineering
Extracted Skills: Python, JavaScript, React
Experience (Min, Max): ('3', '5')
```

## Retrain Models

The training script applies the same degree and specialization text preprocessing used during production inference:

```powershell
.venv\Scripts\python.exe train_models.py
```

It reads `Degree_Prediction_Training_5000_Corrected.csv`, validates both models, and writes these files into `models/`:

```text
degree_vectorizer.pkl
degree_model.pkl
specialization_vectorizer.pkl
specialization_model.pkl
```

## Run the Scraper

Run:

```powershell
.venv\Scripts\python.exe Main.py
```

The current configuration searches these keywords in India:

```text
Full Stack Developer
Java Developer
Spring Boot Developer
Node.js Developer
react developer
```

It searches postings from the last 12 hours and collects up to 100 new jobs per keyword, using pages of 25 results. The theoretical maximum is 500 jobs per run, subject to LinkedIn results, duplicates, expired postings, and access restrictions.

Scraped data is appended to:

```text
csv_output/current_jobs.csv
```

Processed job IDs are stored in:

```text
csv_output/seen_job_ids.json
```

The pipeline uses job IDs and CSV recovery to avoid exporting the same job repeatedly. IDs are saved only after successful processing and CSV export.

## Extraction Features

### Skills

Known skills are normalized through aliases, for example:

```text
Postgres -> PostgreSQL
K8s -> Kubernetes
NLP -> Natural Language Processing
JS -> JavaScript
```

The extractor also discovers uncatalogued technology names from skill-oriented lines, such as:

```text
Snowflake
Databricks
Palantir Foundry
SAP UDF
Apache Flink
```

Use the structured API when category and evidence are needed:

```python
from ml_predictor import extract_skills_structured

skills = extract_skills_structured(job_description)
```

### Experience

Supported formats include:

```text
3-5 years                    -> ('3', '5')
3-5 years of experience     -> ('3', '5')
at least 2 years            -> ('2', '2+')
5+ years                    -> ('5', '5+')
```

Company history is filtered so phrases such as `over 60 years of experience` in a company description do not replace the candidate requirement.

## Outputs

The CSV includes:

- Job ID
- Search category
- Job title
- Company
- Location fields
- Extracted skills
- Required degree
- Required specialization
- Minimum experience
- Maximum experience
- Collection timestamp
- Job link
- Full description

## Validation

Compile the project files:

```powershell
.venv\Scripts\python.exe -m py_compile ml_predictor.py Main.py test_ml.py train_models.py
```

Check installed dependencies:

```powershell
.venv\Scripts\python.exe -m pip check
```

## Notes

- LinkedIn may limit, block, or change search results and HTML selectors.
- `Main.py` records the collection timestamp; it does not currently extract LinkedIn's original posting timestamp.
- Location fields are populated from the current scraper configuration and may require additional parsing for exact city/state values.
- Unknown skills can be added to `SKILL_CATALOG` in `ml_predictor.py` or discovered dynamically when they appear in skill-oriented text.
