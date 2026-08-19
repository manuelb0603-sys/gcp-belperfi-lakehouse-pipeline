# GCP PersonalFi BigQuery Data Lakehouse Pipeline

An automated, modern ELT (Extract, Load, Transform) data pipeline engineered inside Google Cloud Platform utilizing **BigQuery** and **Dataform** to build a structured, production-grade Data Lakehouse architecture.

## 🏗️ Architecture Overview
This pipeline implements a multi-layer Lakehouse medallion architecture designed to process, clean, and standardize financial transaction data:
* **Bronze Layer (Raw):** Raw data ingested from external sources and staged dynamically in BigQuery.
* **Silver Layer (Cleaned):** Dataform models (SQLX) enforce strict schema constraints, apply high-precision financial data typing (`NUMERIC`), handle time-zone standardization, and inject metadata processing logs (`processed_at`).
* **Gold Layer (Curated):** Business-ready views optimized for analytics, BI tools, and stakeholder dashboards.

## 🛠️ Tech Stack & Tooling
* **Cloud Provider:** Google Cloud Platform (GCP)
* **Data Warehouse:** Google BigQuery
* **Data Transformation:** Dataform (SQLX)
* **File Uploads:** Python, pandas, and Google Cloud Storage
* **Version Control:** Git & GitHub Integration
* **Security:** GCP authentication

## 📂 Project Structure
```text
├── Scripts/
│   ├── upload_csv.py          # Uploads issuer-specific CSV files to GCS and archives them
│   ├── run_all_issuers.py     # Runs the uploader for every issuer in config.json
│   ├── memory.py              # SQLite-backed short-term and long-term memory helpers
│   └── email_weekly_report.py # Generates weekly financial insights via BigQuery + Ollama LLM, emails HTML report
├── agent/
│   └── memory/
│       └── memory.sqlite3     # Local runtime database for report memory (ignored by Git)
├── definitions/
│   ├── bronze/                # Raw landing dependency declarations
│   ├── silver/                # Cleaned SQLX models (e.g., chase_transactions.sqlx)
│   └── gold/                  # Curated analytical views
├── config.json                # Runtime configuration for issuer-specific upload settings
├── config.example.json       # Example template for creating config.json
├── workflow_settings.yaml     # Dataform project configuration
└── README.md
```

## 📤 CSV Upload Workflow
The repository also includes a small CSV upload helper for moving local transaction files into Google Cloud Storage.

Each issuer can be configured independently in the config file with its own:
* local input directory
* backup directory
* GCS bucket and prefix
* GCP project name
* PII columns to remove before upload

The upload script reads the configured issuer, removes any listed PII columns, uploads the cleaned CSV to the target bucket/prefix, and then moves the processed file into the backup folder.

## ▶️ How to Run the CSV Upload Script
1. Copy config.example.json to config.json and update the values for your environment.
2. Make sure your Google Cloud credentials are available. A typical setup is:
   ```bash
   gcloud auth application-default login
   ```
3. Install the required Python packages:
   ```bash
   pip install pandas google-cloud-storage
   ```
4. Upload files for one issuer:
   ```bash
   python Scripts/upload_csv.py amex
   ```
   You can also specify the config file explicitly:
   ```bash
   python Scripts/upload_csv.py --config config.json --issuer amex
   ```
5. Upload files for every issuer defined in config.json:
   ```bash
   python Scripts/run_all_issuers.py
   ```

This workflow is useful for automating local CSV ingestion into GCS while keeping a backup copy of each processed file.

## 📧 Weekly Financial Report Automation

The `email_weekly_report.py` script generates an intelligent weekly financial insights email that:
* Queries your **BigQuery Gold Layer** (`fact_monthly_spending`) to extract spending trends, category analysis, and key metrics  
* Processes data through a local **Ollama LLM** (configurable via `model_server_url` & `model_name` in config) to generate contextual insights with playful Japanese food puns from your assistant Mogumogu-chan 🍣🍤
* Uses local SQLite memory to include relevant long-term facts and recent report history in each prompt
* Saves each completed report and extracts reusable financial facts for future reports
* Outputs an attractive HTML report styled for email readability

### Report Memory

The `Scripts/memory.py` module manages the report's local SQLite memory database. The database is created automatically at `agent/memory/memory.sqlite3` when the weekly report runs. Set the `MEMORY_DB_PATH` environment variable to use a different location.

Memory is organized into three tables:
* `threads` stores stable internal identifiers for report conversations. External keys are hashed before storage.
* `short_term` stores recent prompts and report responses for conversational context. Entries are trimmed to the newest 12 per thread by default.
* `long_facts` stores reusable facts extracted from completed reports, such as spending trends or financial goals.

The SQLite database and SQL scratch files are excluded from Git. Delete the local database only if you want to reset the report's memory.

### Setup Instructions

1. **Configure additional settings in `config.json`:**
   ```json  
   {
     "model_server_url": "http://localhost:11434",  // URL of your Ollama server
     "model_name": "mistral",                       // Model name to use for report generation (e.g., mistral, llama2)
     "Monthly_budget": 5000,                        // Your monthly deposit amount  
     "Fixed_costs": 1500,                           // Total fixed expenses per month
     "Vehicle_Info": "Sedan - Gas & Insurance included", 
     "Savings_goal": 1000
   }
   ```

2. **Ensure Ollama is running locally** (or configure a remote server URL):
   * Download from [ollama.com](https://ollama.com) if not installed  
   * Install your preferred model: `ollama pull mistral` (or another supported LLM)

3. **Set up Gmail App Password:** 
   * Generate an app-specific password for your sending email in Google Account settings (2FA required on sender account). Do NOT use your main password!  

4. **Configure recipients** via the `email_listings` array:
   ```json  
   {
     "email_sender": "sender@gmail.com",  // Your Gmail address with app password set
     "Gmail_app_credentials": "your_16_char_password_here", 
     "email_listings": ["subscriber1@gmail.com", "subscriber2@gmail.com"] 
   }
   ```

5. **Schedule automatic weekly runs:**
   
   On Windows: Open Task Scheduler → Create Basic Task (trigger = Weekly) → Action = Start Program with your Python executable and script path
   
   On Linux/macOS: Add crontab entry for automated execution (e.g., `0 9 * * 1 /usr/bin/python3 email_weekly_report.py`)

### Report Features
* ✅ Monthly snapshot with spending trends & category analysis  
* ✅ KPI cards showing total net expense, Mom % change, and disposable cash calculation  
* ✅ Top discretionary categories with goal suggestions based on your budget allocation  
* ✅ Personalized character tips for fun financial habits advice
* ✅ Interactive HTML formatting designed to look beautiful in email clients
