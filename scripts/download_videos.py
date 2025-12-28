import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
from tqdm import tqdm


def load_remote_index(repo_id, repo_type, index_path):
    try:
        cached = hf_hub_download(repo_id, index_path, repo_type=repo_type)
    except EntryNotFoundError:
        return []
    except HfHubHTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return []
        raise
    with open(cached, "r") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = list(data.keys())
    if not isinstance(data, list):
        raise ValueError(f"Remote index {index_path} is not a JSON array.")
    return data


def split_repo_files(repo_files):
    tar_numbers = []
    index_files = []
    for path in repo_files:
        if path.startswith("videos/part_") and path.endswith(".tar"):
            number_text = path[len("videos/part_") : -len(".tar")]
            if number_text.isdigit():
                tar_numbers.append(int(number_text))
            continue
        if path.startswith("index/") and path.endswith(".json"):
            index_files.append(path)
    return tar_numbers, index_files


def next_part_number(tar_numbers):
    return max(tar_numbers) + 1 if tar_numbers else 0


def upload_tar_and_index(api, repo_id, repo_type, index_dir, tar_path, file_names):
    part_name = os.path.basename(tar_path)
    index_path = os.path.join(index_dir, part_name.replace(".tar", ".json"))
    index_data = load_remote_index(repo_id, repo_type, index_path)

    api.upload_file(
        path_or_fileobj=tar_path,
        path_in_repo=f"videos/{part_name}",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    if not index_data:
        index_data = []
    existing = set(index_data)
    for name in file_names:
        if name not in existing:
            index_data.append(name)
            existing.add(name)

    with tempfile.NamedTemporaryFile("w", delete=False) as handle:
        json.dump(index_data, handle)
        temp_index = handle.name
    api.upload_file(
        path_or_fileobj=temp_index,
        path_in_repo=index_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    os.remove(temp_index)


def build_tar(tar_path, file_paths):
    with tarfile.open(tar_path, "w") as tar:
        for path in file_paths:
            tar.add(path, arcname=os.path.basename(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", required=True, type=str)
    parser.add_argument("--video_path", required=True, type=str)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--hf-repo", required=True, type=str)
    parser.add_argument("--repo-type", default="dataset", type=str)
    parser.add_argument("--upload-threshold-gb", type=float, default=8.0)
    parser.add_argument("--index-dir", default="index", type=str)
    args = parser.parse_args()

    if shutil.which("yt-dlp") is None:
        print("yt-dlp is required to download videos. Please install it first.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.video_path, exist_ok=True)
    threshold_bytes = int(args.upload_threshold_gb * 1024 * 1024 * 1024)
    jobs = args.jobs if args.jobs and args.jobs > 0 else 1
    api = HfApi()
    api.create_repo(args.hf_repo, repo_type=args.repo_type, exist_ok=True)

    print("Loading data.")
    with open(args.source_path, "r") as handle:
        packed_data = json.load(handle)

    video_names = list(packed_data.keys())
    youtube_video_format = "https://www.youtube.com/watch?v={}"

    repo_files = api.list_repo_files(args.hf_repo, repo_type=args.repo_type)
    tar_numbers, index_files = split_repo_files(repo_files)
    uploaded_files = set()
    if index_files:
        for index_path in tqdm(index_files, ncols=120, desc="Loading indexes"):
            index_data = load_remote_index(args.hf_repo, args.repo_type, index_path)
            uploaded_files.update(index_data)

    batch_paths = []
    batch_names = []
    batch_size = 0

    def flush_batch():
        nonlocal batch_paths, batch_names, batch_size
        if not batch_paths:
            return
        repo_files = api.list_repo_files(args.hf_repo, repo_type=args.repo_type)
        tar_numbers, _ = split_repo_files(repo_files)
        part_number = next_part_number(tar_numbers)
        tar_name = f"part_{part_number:04d}.tar"
        tar_path = os.path.join(args.video_path, tar_name)
        build_tar(tar_path, batch_paths)
        upload_tar_and_index(
            api,
            args.hf_repo,
            args.repo_type,
            args.index_dir,
            tar_path,
            batch_names,
        )
        os.remove(tar_path)
        uploaded_files.update(batch_names)
        batch_paths = []
        batch_names = []
        batch_size = 0

    def add_to_batch(path):
        nonlocal batch_size
        if not os.path.exists(path):
            return
        size = os.path.getsize(path)
        if size <= 0:
            return
        batch_paths.append(path)
        batch_names.append(os.path.basename(path))
        batch_size += size
        if batch_size >= threshold_bytes:
            flush_batch()

    def download_one(video_name):
        url = youtube_video_format.format(video_name)
        file_path = os.path.join(args.video_path, f"{video_name}.mp4")
        if os.path.exists(file_path):
            return video_name, "skipped", file_path
        try:
            subprocess.run(
                ["yt-dlp", "-c", "-o", file_path, "-f", "134", url],
                check=True,
            )
            return video_name, "downloaded", file_path
        except subprocess.CalledProcessError as exc:
            return video_name, "failed", exc

    success_count = 0
    fail_count = 0
    skip_count = 0

    to_download = []
    with tqdm(total=len(video_names), ncols=120, desc="Downloading videos") as progress:
        for name in video_names:
            filename = f"{name}.mp4"
            file_path = os.path.join(args.video_path, filename)
            if filename in uploaded_files:
                skip_count += 1
                progress.update(1)
                progress.set_postfix(success=success_count, failed=fail_count, skipped=skip_count)
                continue
            if os.path.exists(file_path):
                skip_count += 1
                add_to_batch(file_path)
                progress.update(1)
                progress.set_postfix(success=success_count, failed=fail_count, skipped=skip_count)
                continue
            to_download.append(name)

        if jobs == 1:
            for name in to_download:
                _, status, result = download_one(name)
                if status == "downloaded":
                    success_count += 1
                    add_to_batch(result)
                elif status == "failed":
                    fail_count += 1
                else:
                    skip_count += 1
                    if isinstance(result, str):
                        add_to_batch(result)
                progress.update(1)
                progress.set_postfix(success=success_count, failed=fail_count, skipped=skip_count)
        else:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = {executor.submit(download_one, name): name for name in to_download}
                for future in as_completed(futures):
                    _, status, result = future.result()
                    if status == "downloaded":
                        success_count += 1
                        add_to_batch(result)
                    elif status == "failed":
                        fail_count += 1
                    else:
                        skip_count += 1
                        if isinstance(result, str):
                            add_to_batch(result)
                    progress.update(1)
                    progress.set_postfix(success=success_count, failed=fail_count, skipped=skip_count)

    if batch_paths:
        flush_batch()

    print(f"Finished. Success: {success_count}. Failed: {fail_count}. Skipped: {skip_count}.")


if __name__ == "__main__":
    main()
