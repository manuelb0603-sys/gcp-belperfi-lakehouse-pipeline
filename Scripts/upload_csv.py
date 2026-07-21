import io
import os
import json
import argparse
from pathlib import Path
import pandas as pd
from google.cloud import storage
from google.api_core import exceptions as google_exceptions

DEFAULT_ISSUER = "amex"


def resolve_config_path(config_path="config.json"):
    if os.path.isabs(config_path):
        return config_path

    project_root = Path(__file__).resolve().parent.parent
    return str(project_root / config_path)


def load_config(config_path="config.json", issuer=DEFAULT_ISSUER):
    resolved_config_path = resolve_config_path(config_path)

    if not os.path.exists(resolved_config_path):
        raise FileNotFoundError(f"Configuration file '{resolved_config_path}' not found.")

    with open(resolved_config_path, "r") as f:
        full_config = json.load(f)

    if "issuers" not in full_config or issuer not in full_config["issuers"]:
        raise KeyError(f"Issuer '{issuer}' not found in configuration file '{resolved_config_path}'.")

    return full_config["issuers"][issuer]


def transform_and_upload(config_path="config.json", issuer=DEFAULT_ISSUER):
    if not issuer:
        raise ValueError("An issuer must be provided. Use a positional issuer argument or --issuer.")

    config = load_config(config_path, issuer)

    local_dir = config["local_dir"]
    backup_dir = config.get("backup_dir")
    bucket_name = config["bucket_name"]
    gcs_prefix = config.get("gcs_prefix")
    pii_columns = config.get("pii_columns", [])

    storage_client_kwargs = {}
    project_name = config.get("project_name")
    if project_name:
        storage_client_kwargs["project"] = project_name

    try:
        storage_client = storage.Client(**storage_client_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Unable to initialize Google Cloud Storage client. Set a project in config.json or GOOGLE_CLOUD_PROJECT."
        ) from exc

    bucket = storage_client.bucket(bucket_name)

    for filename in os.listdir(local_dir):
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(local_dir, filename)
            print(f"Processing: {filename}")

            df = pd.read_csv(file_path)

            if pii_columns:
                df = df.drop(columns=[col for col in pii_columns if col in df.columns])

            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)

            if not gcs_prefix:
                raise KeyError(f"Missing 'gcs_prefix' for issuer '{issuer}' in the config file.")

            gcs_blob_name = f"{gcs_prefix}/{filename}"
            blob = bucket.blob(gcs_blob_name)

            upload_success = False
            for attempt in range(1, 4):
                try:
                    blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")
                    upload_success = True
                    break
                except (google_exceptions.TimeoutError, google_exceptions.ServiceUnavailable,
                        google_exceptions.TooManyRequests, ConnectionError, OSError) as exc:
                    if attempt == 3:
                        raise RuntimeError(f"Upload failed for {filename} after 3 attempts") from exc
                    print(f"Upload attempt {attempt} failed for {filename}: {exc}. Retrying...")

            if not upload_success:
                raise RuntimeError(f"Upload did not complete for {filename}")

            print(f"Successfully uploaded clean file to gs://{bucket_name}/{gcs_blob_name}")

            if backup_dir:
                os.makedirs(backup_dir, exist_ok=True)
                destination_path = os.path.join(backup_dir, filename)
                os.replace(file_path, destination_path)
                print(f"Moved file to backup directory: {destination_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strip PII and upload CSVs to Cloud Storage.")
    parser.add_argument("--config", default="config.json", help="Path to config JSON file")
    parser.add_argument("issuer", nargs="?", default=None, help="Issuer key from the config file")
    parser.add_argument("--issuer", dest="issuer_flag", default=None, help="Issuer key from the config file")
    args = parser.parse_args()

    issuer = args.issuer_flag or args.issuer or DEFAULT_ISSUER
    transform_and_upload(args.config, issuer)
