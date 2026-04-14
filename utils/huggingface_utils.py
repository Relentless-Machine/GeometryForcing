from pathlib import Path

from huggingface_hub import hf_hub_download


def _local_hfd_download_path(repo_id: str, filename: str) -> str | None:
    repo_name = repo_id.split("/")[-1]
    repo_slug = repo_id.replace("/", "--")
    repo_key = repo_id.replace("/", "_")
    candidates = [
        Path("./huggingface") / filename,
        Path("./huggingface") / "metrics_models" / Path(filename).name,
        Path("./huggingface") / repo_name / filename,
        Path("./huggingface") / repo_id / filename,
        Path("./huggingface") / repo_slug / filename,
        Path("./huggingface") / repo_key / filename,
        Path("./huggingface") / f"models--{repo_slug}" / "snapshots",
    ]

    for candidate in candidates[:-1]:
        if candidate.is_file():
            return str(candidate)

    snapshots_root = candidates[-1]
    if snapshots_root.is_dir():
        for snapshot in sorted(snapshots_root.iterdir(), reverse=True):
            snap_file = snapshot / filename
            if snap_file.is_file():
                return str(snap_file)

    return None


def local_hfd_repo_dir(repo_id: str) -> str | None:
    repo_name = repo_id.split("/")[-1]
    repo_slug = repo_id.replace("/", "--")
    repo_key = repo_id.replace("/", "_")
    candidates = [
        Path("./huggingface") / repo_slug,
        Path("./huggingface") / repo_key,
        Path("./huggingface") / repo_name,
        Path("./huggingface") / repo_id,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)

    snapshots_root = Path("./huggingface") / f"models--{repo_slug}" / "snapshots"
    if snapshots_root.is_dir():
        for snapshot in sorted(snapshots_root.iterdir(), reverse=True):
            if snapshot.is_dir() and (snapshot / "config.json").is_file():
                return str(snapshot)

    hf_root = Path("./huggingface")
    if (hf_root / "config.json").is_file() and (
        (hf_root / "model.safetensors").is_file()
        or (hf_root / "pytorch_model.bin").is_file()
    ):
        return str(hf_root)

    return None


def download_from_hf(
    repo_id: str,
    filename: str,
) -> str:
    """
    Download a file from a Hugging Face model hub.
    If the file was already downloaded by hfd into ./huggingface, reuse it first.
    """
    local_path = _local_hfd_download_path(repo_id, filename)
    if local_path is not None:
        return local_path

    return hf_hub_download(
        repo_id=repo_id,
        cache_dir="./huggingface",
        filename=filename,
    )
