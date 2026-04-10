from hashlib import sha256
from pathlib import Path


def resource_files(resource_path: Path) -> list[Path]:
    files = list(resource_path.glob("**/*.txt")) + list(resource_path.glob("**/*.md"))
    return sorted(files, key=lambda path: str(path.relative_to(resource_path)))


def hash_file(file_path: Path) -> str:
    digest = sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filesystem_resource_manifest(resource_path: Path) -> dict[str, str]:
    return {
        str(file_path.relative_to(resource_path)): hash_file(file_path)
        for file_path in resource_files(resource_path)
    }
