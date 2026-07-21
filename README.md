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
│   └── run_all_issuers.py     # Runs the uploader for every issuer in config.json
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