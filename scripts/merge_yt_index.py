import argparse
import json
import os
import tempfile

import ijson
from huggingface_hub import HfApi

YT_INDEX_NAME = "yt_index.json"


def iter_ids_from_json(path):
    top_level = None
    with open(path, "r") as handle:
        for prefix, event, value in ijson.parse(handle):
            if top_level is None:
                if prefix == "" and event == "start_map":
                    top_level = "map"
                elif prefix == "" and event == "start_array":
                    top_level = "array"
            if top_level == "map" and prefix == "" and event == "map_key":
                yield value
            elif top_level == "array" and prefix == "item.id" and event in ("string", "number"):
                yield str(value)


def build_yt_index(sources):
    ids = []
    seen = set()
    for source in sources:
        for video_id in iter_ids_from_json(source):
            if video_id in seen:
                continue
            seen.add(video_id)
            ids.append(video_id)
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--hf-repo", required=True, type=str)
    parser.add_argument("--repo-type", default="dataset", type=str)
    args = parser.parse_args()

    yt_index = build_yt_index(args.source)
    if not yt_index:
        raise ValueError("No ids found in source files.")

    api = HfApi()
    api.create_repo(args.hf_repo, repo_type=args.repo_type, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="yt_index_") as temp_dir:
        index_path = os.path.join(temp_dir, YT_INDEX_NAME)
        with open(index_path, "w") as handle:
            json.dump(yt_index, handle)
        api.upload_file(
            path_or_fileobj=index_path,
            path_in_repo=YT_INDEX_NAME,
            repo_id=args.hf_repo,
            repo_type=args.repo_type,
            commit_message=f"Upload {YT_INDEX_NAME}",
        )
    print(f"Uploaded {YT_INDEX_NAME} with {len(yt_index)} ids.")


if __name__ == "__main__":
    main()
