import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import random

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    are_progress_bars_disabled,
    disable_progress_bars,
    enable_progress_bars,
)
from tqdm import tqdm

DEFAULT_DISPATCH_INTERVAL_SECONDS = 2.0
INDEX_REFRESH_SECONDS = 60.0
DEFAULT_UPLOAD_REFERENCE_GB = 7.0
TQDM_FORMAT = "{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"


@contextmanager
def hf_quiet_upload():
    previously_disabled = are_progress_bars_disabled()
    disable_progress_bars()
    try:
        yield
    finally:
        if not previously_disabled:
            enable_progress_bars()


def load_remote_index(repo_id, repo_type, index_path, revision):
    try:
        cached = hf_hub_download(
            repo_id,
            index_path,
            repo_type=repo_type,
            revision=revision,
        )
    except EntryNotFoundError:
        return set()
    except HfHubHTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return set()
        raise
    with open(cached, "r") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return set(data)
    if isinstance(data, dict):
        return set(data.keys())
    raise ValueError(f"Remote index {index_path} must be a JSON array or object.")


def split_repo_files(repo_files):
    tar_numbers = []
    index_numbers = []
    index_files = []
    for path in repo_files:
        if path.startswith("videos/part_") and path.endswith(".tar"):
            number_text = path[len("videos/part_") : -len(".tar")]
            if number_text.isdigit():
                tar_numbers.append(int(number_text))
            continue
        if path.startswith("index/") and path.endswith(".json"):
            index_files.append(path)
            if path.startswith("index/part_"):
                number_text = path[len("index/part_") : -len(".json")]
                if number_text.isdigit():
                    index_numbers.append(int(number_text))
    return tar_numbers, index_numbers, index_files


def next_part_number(tar_numbers, index_numbers):
    all_numbers = tar_numbers + index_numbers
    return max(all_numbers) + 1 if all_numbers else 0


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
    parser.add_argument("--upload-reference-gb", type=float, default=DEFAULT_UPLOAD_REFERENCE_GB)
    parser.add_argument("--dispatch-interval", type=float, default=DEFAULT_DISPATCH_INTERVAL_SECONDS)
    parser.add_argument("--index-dir", default="index", type=str)
    args = parser.parse_args()

    if shutil.which("yt-dlp") is None:
        print("yt-dlp is required to download videos. Please install it first.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.video_path, exist_ok=True)
    threshold_bytes = int(args.upload_threshold_gb * 1024 * 1024 * 1024)
    reference_bytes = int(args.upload_reference_gb * 1024 * 1024 * 1024)
    if reference_bytes > threshold_bytes:
        reference_bytes = threshold_bytes
    jobs = args.jobs if args.jobs and args.jobs > 0 else 1
    api = HfApi()
    api.create_repo(args.hf_repo, repo_type=args.repo_type, exist_ok=True)

    print("Loading data.")
    with open(args.source_path, "r") as handle:
        packed_data = json.load(handle)

    video_names = list(packed_data.keys())
    youtube_video_format = "https://www.youtube.com/watch?v={}"

    batch_entries = []
    batch_names = set()
    batch_size = 0
    post_skip_count = 0
    last_index_refresh = 0.0
    last_index_sha = None
    last_repo_files = []
    last_remote_files = set()

    def add_to_batch(path):
        nonlocal batch_size
        if not os.path.exists(path):
            return False
        name = os.path.basename(path)
        if name in batch_names:
            return False
        size = os.path.getsize(path)
        if size <= 0:
            return False
        batch_entries.append({"name": name, "path": path, "size": size})
        batch_names.add(name)
        batch_size += size
        return True

    def remove_entries(names, count_post_skip=False):
        nonlocal batch_entries, batch_size, post_skip_count
        if not names or not batch_entries:
            return 0
        remaining = []
        removed = 0
        for entry in batch_entries:
            if entry["name"] in names:
                removed += 1
                batch_size -= entry["size"]
                batch_names.discard(entry["name"])
            else:
                remaining.append(entry)
        if removed:
            batch_entries = remaining
            if count_post_skip:
                post_skip_count += removed
        return removed

    def refresh_indexes(force=False):
        nonlocal last_remote_files, last_index_refresh, last_index_sha, last_repo_files
        now = time.monotonic()
        if not force and now - last_index_refresh < INDEX_REFRESH_SECONDS:
            return False
        curr_index_sha = api.repo_info(args.hf_repo, repo_type=args.repo_type).sha
        curr_repo_files = api.list_repo_files(
            args.hf_repo,
            repo_type=args.repo_type,
            revision=curr_index_sha,
        )
        _, _, index_files = split_repo_files(curr_repo_files)

        curr_remote_files = set()
        with hf_quiet_upload():
            for index_path in tqdm(
                index_files,
                ncols=120,
                desc="Loading indexes",
                mininterval=10,
                bar_format=TQDM_FORMAT,
            ):
                index_data = load_remote_index(
                    args.hf_repo,
                    args.repo_type,
                    index_path,
                    curr_index_sha,
                )
                curr_remote_files.update(index_data)

        changed = curr_remote_files != last_remote_files
        removed = remove_entries(curr_remote_files, count_post_skip=True)

        last_index_refresh = now
        last_index_sha = curr_index_sha
        last_repo_files = curr_repo_files
        last_remote_files = curr_remote_files
        return changed or removed > 0

    def select_entries(target_bytes, size_lowerbound=0):
        selected = []
        total = 0
        for entry in batch_entries:
            if total >= target_bytes:
                break
            selected.append(entry)
            total += entry["size"]
        if total < size_lowerbound:
            return [], 0
        return selected, total

    def is_conflict_error(exc):
        if getattr(exc, "response", None) is None:
            return False
        return exc.response.status_code in (409, 412)

    def attempt_upload(final=False):
        nonlocal post_skip_count
        while True:
            refresh_indexes(force=True)
            if not batch_entries:
                return False
            tar_numbers, index_numbers, index_files = split_repo_files(last_repo_files)
            part_number = next_part_number(tar_numbers, index_numbers)
            tar_name = f"part_{part_number:04d}.tar"
            index_path = os.path.join(args.index_dir, tar_name.replace(".tar", ".json"))
            if index_path in index_files:
                continue

            size_lowerbound=(0 if final else reference_bytes)
            selected, _ = select_entries(threshold_bytes, size_lowerbound)
            if not selected:
                return False
            selected_names = [entry["name"] for entry in selected]
            selected_sizes = {entry["name"]: entry["size"] for entry in selected}

            temp_dir = tempfile.mkdtemp(prefix="hf_upload_")
            temp_videos = os.path.join(temp_dir, "videos")
            temp_index_dir = os.path.join(temp_dir, "index")
            os.makedirs(temp_videos, exist_ok=True)
            os.makedirs(temp_index_dir, exist_ok=True)

            tar_path = os.path.join(temp_videos, tar_name)
            build_tar(tar_path, [entry["path"] for entry in selected])

            temp_index = os.path.join(temp_index_dir, os.path.basename(index_path))
            with open(temp_index, "w") as handle:
                json.dump(selected_sizes, handle, indent=0, ensure_ascii=False)

            try:
                parent_sha = last_index_sha
                with hf_quiet_upload():
                    api.upload_folder(
                        folder_path=temp_dir,
                        path_in_repo="",
                        repo_id=args.hf_repo,
                        repo_type=args.repo_type,
                        parent_commit=parent_sha,
                        commit_message=f"Upload videos/{tar_name} and {index_path}",
                    )
            except HfHubHTTPError as exc:
                shutil.rmtree(temp_dir, ignore_errors=True)
                if is_conflict_error(exc):
                    continue
                raise

            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Upload complete: videos/{tar_name} -> {index_path}")
            remove_entries(set(selected_names))
            last_remote_files.update(selected_names)
            return True

    def maybe_flush():
        if batch_size >= threshold_bytes:
            attempt_upload(final=False)

    def download_one(video_name):
        url = youtube_video_format.format(video_name)
        file_path = os.path.join(args.video_path, f"{video_name}.mp4")
        if os.path.exists(file_path):
            return video_name, "skipped", file_path
        try:
            subprocess.run(
                [
                    "yt-dlp",
                    "-c",
                    "--no-progress",
                    "--no-warnings",
                    "--quiet",
                    "-o",
                    file_path,
                    "-f",
                    "134",
                    url,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return video_name, "downloaded", file_path
        except subprocess.CalledProcessError as exc:
            return video_name, "failed", exc

    success_count = 0
    fail_count = 0
    skip_count = 0
    dispatch_interval = max(0.0, args.dispatch_interval)
    next_dispatch_time = time.monotonic() + dispatch_interval

    def status_text():
        pending_gb = batch_size / (1024 * 1024 * 1024)
        return (
            f"success={success_count} "
            f"fail={fail_count} "
            f"skip={skip_count} "
            f"post_skip={post_skip_count} "
            f"pending={pending_gb:.2f}"
        )

    def refresh_if_due(progress):
        if refresh_indexes(force=False):
            progress.set_postfix_str(status_text(), refresh=False)

    def wait_for_dispatch(collect_fn=None, refresh_fn=None):
        nonlocal next_dispatch_time
        if dispatch_interval <= 0:
            return
        while True:
            now = time.monotonic()
            if now >= next_dispatch_time:
                break
            if collect_fn is not None:
                collect_fn()
            if refresh_fn is not None:
                refresh_fn()
            time.sleep(0.2)
        next_dispatch_time = max(
            next_dispatch_time + dispatch_interval,
            time.monotonic() + dispatch_interval,
        )

    refresh_indexes(force=True)
    random.shuffle(video_names)
    with tqdm(
        total=len(video_names),
        ncols=120,
        desc="Downloading videos",
        mininterval=10,
        bar_format=TQDM_FORMAT,
    ) as progress:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            pending = []
            max_pending = max(1, jobs * 2)

            def collect_completed():
                nonlocal success_count, fail_count, skip_count, post_skip_count
                completed = []
                for future in pending:
                    if not future.done():
                        continue
                    completed.append(future)
                    name, status, result = future.result()
                    filename = f"{name}.mp4"
                    if status == "downloaded":
                        if filename in last_remote_files:
                            post_skip_count += 1
                        else:
                            success_count += 1
                            add_to_batch(result)
                    elif status == "failed":
                        fail_count += 1
                    else:
                        skip_count += 1
                        if isinstance(result, str):
                            if filename in last_remote_files:
                                post_skip_count += 1
                            else:
                                add_to_batch(result)
                    progress.update(1)
                    progress.set_postfix_str(status_text(), refresh=False)
                for future in completed:
                    pending.remove(future)
                if completed:
                    maybe_flush()
                    progress.set_postfix_str(status_text(), refresh=False)

            for name in video_names:
                refresh_if_due(progress)
                filename = f"{name}.mp4"
                file_path = os.path.join(args.video_path, filename)
                if filename in last_remote_files:
                    skip_count += 1
                    progress.update(1)
                    progress.set_postfix_str(status_text(), refresh=False)
                    continue
                if os.path.exists(file_path):
                    skip_count += 1
                    add_to_batch(file_path)
                    maybe_flush()
                    progress.update(1)
                    progress.set_postfix_str(status_text(), refresh=False)
                    continue

                while len(pending) >= max_pending:
                    collect_completed()
                    refresh_if_due(progress)
                    if len(pending) >= max_pending:
                        time.sleep(0.2)

                pending.append(executor.submit(download_one, name))
                wait_for_dispatch(collect_completed, lambda: refresh_if_due(progress))
                collect_completed()

            while pending:
                collect_completed()
                refresh_if_due(progress)
                if pending:
                    time.sleep(0.5)

            while batch_entries:
                if attempt_upload(final=True):
                    progress.set_postfix_str(status_text(), refresh=False)
                    continue
                refresh_if_due(progress)
                if not batch_entries:
                    break
                time.sleep(0.5)

    print(f"Finished. Success: {success_count}. Failed: {fail_count}. Skipped: {skip_count}.")


if __name__ == "__main__":
    main()
