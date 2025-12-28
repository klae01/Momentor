import argparse
import os
import shutil

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

DEFAULT_JSON_FILES = [
    "Moment-10M_0.json",
    "Moment-10M_1.json",
    "GESM_data.json",
]


def download_json(repo_id, raw_dir, repo_type):
    os.makedirs(raw_dir, exist_ok=True)
    for filename in tqdm(DEFAULT_JSON_FILES, ncols=120, desc="Downloading JSON"):
        remote_path = f"data/{filename}"
        local_path = os.path.join(raw_dir, filename)
        cached = hf_hub_download(repo_id, remote_path, repo_type=repo_type)
        shutil.copy(cached, local_path)
    print("Finished.")


def upload_json(repo_id, raw_dir, repo_type):
    api = HfApi()
    api.create_repo(repo_id, repo_type=repo_type, exist_ok=True)
    for filename in tqdm(DEFAULT_JSON_FILES, ncols=120, desc="Uploading JSON"):
        local_path = os.path.join(raw_dir, filename)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Missing JSON file: {local_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"data/{filename}",
            repo_id=repo_id,
            repo_type=repo_type,
        )
    print("Finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=str)
    parser.add_argument("--raw-dir", required=True, type=str)
    parser.add_argument("--repo-type", default="dataset", type=str)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--download", action="store_true")
    group.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    if args.download:
        download_json(args.repo, args.raw_dir, args.repo_type)
    else:
        upload_json(args.repo, args.raw_dir, args.repo_type)


if __name__ == "__main__":
    main()
