"""Upload sandbox_scripts/remediate.py and project assets to Daytona Sandboxes."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from daytona_sdk import Daytona, DaytonaConfig


def sync_to_daytona_sandboxes():
    api_key = os.getenv("DAYTONA_API_KEY")
    if not api_key:
        print("ERROR: DAYTONA_API_KEY is not set in environment or .env file.")
        sys.exit(1)

    print(f"Connecting to Daytona with API key: {api_key[:10]}...")
    client = Daytona(DaytonaConfig(api_key=api_key))

    sandboxes = list(client.list())
    print(f"Found {len(sandboxes)} total sandboxes.")

    if not sandboxes:
        print("No sandboxes found in Daytona account.")
        return

    # Files and scripts to upload
    files_to_sync = [
        ("sandbox_scripts/remediate.py", "sandbox_scripts/remediate.py"),
        ("eval/eval_set.json", "eval/eval_set.json"),
        ("eval/judge.py", "eval/judge.py"),
        ("eval/compare.py", "eval/compare.py"),
        ("pipeline/chunker.py", "pipeline/chunker.py"),
        ("pipeline/embedder.py", "pipeline/embedder.py"),
        ("pipeline/index.py", "pipeline/index.py"),
        ("pipeline/retriever.py", "pipeline/retriever.py"),
        ("pipeline/generator.py", "pipeline/generator.py"),
        ("pipeline/llm_client.py", "pipeline/llm_client.py"),
        ("indexes/active.json", "indexes/active.json"),
    ]

    # Add all active corpus files
    corpus_active_dir = PROJECT_ROOT / "corpus" / "active"
    if corpus_active_dir.exists():
        for doc_file in corpus_active_dir.rglob("*.md"):
            rel_p = doc_file.relative_to(PROJECT_ROOT).as_posix()
            files_to_sync.append((rel_p, rel_p))

    # Sort sandboxes by created_at (most recent first)
    sandboxes_sorted = sorted(sandboxes, key=lambda s: getattr(s, "created_at", "") or "", reverse=True)

    for i, sb in enumerate(sandboxes_sorted[:3]):  # Sync to the 3 most recent sandboxes
        sb_id = getattr(sb, "id", "unknown")
        sb_name = getattr(sb, "name", "unknown")
        sb_state = str(getattr(sb, "state", "unknown"))

        print(f"\n[{i+1}/3] Processing Sandbox: {sb_name} ({sb_id}) - State: {sb_state}")

        try:
            # If stopped or paused, start sandbox to allow file upload
            if "stopped" in sb_state.lower() or "paused" in sb_state.lower():
                print(f"  Starting sandbox {sb_id}...")
                sb.start()
                sb.wait_for_sandbox_start(timeout=60)
            elif "archived" in sb_state.lower():
                print(f"  Skipping archived sandbox {sb_id}")
                continue

            work_dir = getattr(sb, "get_work_dir", lambda: "/workspace")()
            print(f"  Target work directory: {work_dir}")

            for local_rel, remote_rel in files_to_sync:
                local_path = PROJECT_ROOT / local_rel
                if not local_path.exists():
                    continue

                # Upload both to root/relative and work_dir/relative for max compatibility
                dest_paths = [
                    remote_rel,
                    f"{work_dir.rstrip('/')}/{remote_rel}",
                    f"/home/daytona/{remote_rel}",
                ]

                # Ensure parent folder exists
                parent_dir = str(Path(remote_rel).parent).replace("\\", "/")
                if parent_dir and parent_dir != ".":
                    try:
                        sb.fs.create_folder(parent_dir)
                    except Exception:
                        pass
                    try:
                        sb.fs.create_folder(f"{work_dir.rstrip('/')}/{parent_dir}")
                    except Exception:
                        pass

                # Upload file
                try:
                    file_bytes = local_path.read_bytes()
                    sb.fs.upload_file(file_bytes, remote_rel)
                    sb.fs.upload_file(file_bytes, f"{work_dir.rstrip('/')}/{remote_rel}")
                    print(f"  [OK] Uploaded: {remote_rel}")
                except Exception as e:
                    print(f"  [!] Warning uploading {remote_rel}: {e}")

        except Exception as e:
            print(f"  ERROR syncing to sandbox {sb_id}: {e}")

    print("\n[SUCCESS] Daytona sandbox synchronization complete!")


if __name__ == "__main__":
    sync_to_daytona_sandboxes()
