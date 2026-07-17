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
* **Version Control:** Git & GitHub Integration
* **Security:** GCP Secret Manager (for secure repository authentication)

## 📂 Project Structure
```text
├── definitions/
│   ├── staging/      # Raw bronze tables and schema configurations
│   ├── silver/       # Cleaned SQLX models (e.g., chase_transactions.sqlx)
│   └── gold/         # Curated analytical views
├── includes/         # Reusable javascript macros and constants
├── .gitignore        # Git exclusion rules
└── workflow_settings.yaml # Dataform project configurations