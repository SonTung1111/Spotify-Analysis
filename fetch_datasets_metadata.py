import csv
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from datetime import datetime

api = KaggleApi()
api.authenticate()

KEYWORD = "spotify"
OUTPUT_FILE = Path("datasets/spotify_datasets_metadata.csv")
CSV_COLUMNS = [
    "id",
    "ref",
    "subtitle",
    "creator_name",
    "creator_url",
    "total_bytes",
    "url",
    "last_updated",
    "download_count",
    "is_private",
    "is_featured",
    "license_name",
    "description",
    "owner_name",
    "owner_ref",
    "kernel_count",
    "title",
    "topic_count",
    "view_count",
    "vote_count",
    "current_version_number",
    "usability_rating",
    "tags",
    "files",
    "versions",
]

raw_datasets = []
# Fetch first 10 pages (20 * 10 ~ 200 datasets metadata)
for page_num in range(1, 11):
    try:
        page_results = api.dataset_list(search=KEYWORD, sort_by="votes", page=page_num)

        if not page_results:
            break

        raw_datasets.extend(page_results)
    except Exception as e:
        print(f"Could not fetch page {page_num}: {e}")
        break


with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(CSV_COLUMNS)

    for ds in raw_datasets:
        row_data = []
        for col in CSV_COLUMNS:
            val = getattr(ds, col, None)

            if col == "last_updated" and isinstance(val, datetime):
                val = val.strftime("%Y-%m-%d %H:%M:%S")

            elif col == "tags" and isinstance(val, list):
                val = ", ".join([getattr(t, "name", str(t)) for t in val])

            elif col in ["files", "versions"] and isinstance(val, list):
                val = ", ".join([str(f) for f in val])

            if val is None:
                val = ""

            row_data.append(val)

        writer.writerow(row_data)
