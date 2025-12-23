import os
import json

# Define paths
raw_dir = "/Users/aakarsh/Desktop/ProjectRoot/phone_videos/raw/"
annotations_dir = "/Users/aakarsh/Desktop/ProjectRoot/phone_videos/annotations/"

# Ensure annotations folder exists
os.makedirs(annotations_dir, exist_ok=True)

# Template metadata structure
template = {
    "video_filename": "",
    "record_date": "",
    "participants": [],
    "location": "",
    "tags": [],
    "device_model": "",
    "notes": ""
}

# Iterate through video files in raw folder
for filename in os.listdir(raw_dir):
    if filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        base_name = os.path.splitext(filename)[0]
        json_path = os.path.join(annotations_dir, f"{base_name}.json")

        if not os.path.exists(json_path):
            metadata = template.copy()
            metadata["video_filename"] = filename

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)

            print(f"🆕 Created metadata file: {json_path}")
        else:
            print(f"✅ Metadata already exists for: {filename}")
