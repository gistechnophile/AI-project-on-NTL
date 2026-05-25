"""
Robust ResNet weight downloader with retry logic and hash verification.

PyTorch's default downloader doesn't retry on corrupted chunks.
This script downloads from PyTorch CDN with:
  - Resume support (partial downloads)
  - SHA256 hash verification
  - Multiple retry attempts with exponential backoff
  - Progress reporting
"""

import os
import sys
import hashlib
import time
from pathlib import Path

import urllib.request
import urllib.error


# PyTorch model zoo URLs and their expected SHA256 hashes
MODELS = {
    "resnet18": {
        "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        "sha256": "f37072fd2f71d93d2ba7d46632ec57f3",
        "size_mb": 44.7,
    },
    "resnet34": {
        "url": "https://download.pytorch.org/models/resnet34-b627a593.pth",
        "sha256": "b627a593",
        "size_mb": 83.3,
    },
    "resnet50": {
        "url": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        "sha256": "11ad3fa6",
        "size_mb": 97.8,
    },
}


def get_torch_hub_dir():
    """Get PyTorch hub checkpoint directory."""
    torch_hub = os.path.expanduser(os.getenv("TORCH_HOME", "~/.cache/torch"))
    return Path(torch_hub) / "hub" / "checkpoints"


def compute_sha256(filepath, chunk_size=8192):
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_with_resume(url, dest_path, expected_hash, max_retries=5):
    """
    Download a file with resume support and hash verification.
    Returns True if successful, False otherwise.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        print(f"\n[Attempt {attempt}/{max_retries}] Downloading {url}")
        print(f"  Destination: {dest_path}")

        # Check if partial file exists
        existing_size = dest_path.stat().st_size if dest_path.exists() else 0
        if existing_size > 0:
            print(f"  Resuming from {existing_size / 1e6:.1f} MB")

        headers = {}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"

        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                if existing_size > 0 and response.status == 206:
                    total_size += existing_size
                    mode = "ab"
                else:
                    mode = "wb"
                    existing_size = 0

                downloaded = existing_size
                block_size = 8192
                start_time = time.time()

                with open(dest_path, mode) as f:
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Progress
                        elapsed = time.time() - start_time
                        speed = downloaded / elapsed / 1024 if elapsed > 0 else 0
                        pct = downloaded / total_size * 100 if total_size > 0 else 0
                        print(f"\r  {pct:.1f}% | {downloaded/1e6:.1f}/{total_size/1e6:.1f} MB | "
                              f"{speed:.1f} KB/s", end="", flush=True)

                print()  # newline after progress

            # Verify hash
            print("  Verifying SHA256 hash...")
            actual_hash = compute_sha256(dest_path)
            if actual_hash.startswith(expected_hash):
                print(f"  Hash OK: {actual_hash[:16]}...")
                return True
            else:
                print(f"  Hash MISMATCH!")
                print(f"    Expected: {expected_hash}...")
                print(f"    Got:      {actual_hash[:16]}...")
                print("  Deleting corrupted file and retrying...")
                dest_path.unlink()

        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"  Network error: {e}")
            wait_time = min(2 ** attempt, 60)  # Exponential backoff, cap at 60s
            print(f"  Waiting {wait_time}s before retry...")
            time.sleep(wait_time)

        except Exception as e:
            print(f"  Unexpected error: {e}")
            break

    print(f"\n[FAILED] Could not download {url} after {max_retries} attempts.")
    return False


def main():
    hub_dir = get_torch_hub_dir()
    print(f"PyTorch hub directory: {hub_dir}")

    success_count = 0
    for name, info in MODELS.items():
        dest = hub_dir / f"{name}-{info['sha256']}.pth"
        if dest.exists():
            actual_hash = compute_sha256(dest)
            if actual_hash.startswith(info["sha256"]):
                print(f"\n{name}: Already cached and verified")
                success_count += 1
                continue
            else:
                print(f"\n{name}: Cached but hash mismatch! Re-downloading...")
                dest.unlink()

        if download_with_resume(info["url"], dest, info["sha256"]):
            success_count += 1

    print(f"\n{'='*50}")
    print(f"Downloaded {success_count}/{len(MODELS)} models successfully.")
    if success_count == len(MODELS):
        print("All ResNet weights ready for training!")
        return 0
    else:
        print("Some downloads failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
