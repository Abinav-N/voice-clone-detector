"""
Deploy the Voice Clone Detector Gradio Demo to Hugging Face Spaces (Task 24).

Usage:
    # 1. Using huggingface_hub API:
    python deploy_hf.py --space your-username/voice-clone-detector --token YOUR_HF_TOKEN

    # 2. Or using Git:
    git remote add space https://huggingface.co/spaces/your-username/voice-clone-detector
    git push space main
"""
import argparse
import os
import sys
from pathlib import Path


def deploy_to_spaces(space_id: str, token: str = None, checkpoint_path: str = "checkpoints/best.pt"):
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("huggingface_hub is not installed. Installing it now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)

    print(f"Creating / verifying Hugging Face Space: {space_id}...")
    try:
        create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
            token=token,
        )
        print(f"✓ Space ready at https://huggingface.co/spaces/{space_id}")
    except Exception as e:
        print(f"[Notice] Repository access check: {e}")

    # Files to upload
    files_to_upload = [
        "app.py",
        "README.md",
        "requirements.txt",
        "packages.txt",
    ]

    # Directories to upload
    dirs_to_upload = ["src"]

    print("\nUploading core files...")
    for f in files_to_upload:
        if os.path.isfile(f):
            print(f"  Uploading {f}...")
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f,
                repo_id=space_id,
                repo_type="space",
            )

    print("\nUploading src directory...")
    for d in dirs_to_upload:
        if os.path.isdir(d):
            api.upload_folder(
                folder_path=d,
                path_in_repo=d,
                repo_id=space_id,
                repo_type="space",
            )

    # Optional checkpoint upload if present
    if os.path.isfile(checkpoint_path):
        print(f"\nUploading trained model checkpoint: {checkpoint_path}...")
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=checkpoint_path,
            repo_id=space_id,
            repo_type="space",
        )
    else:
        print(f"\n[Note] No checkpoint found at '{checkpoint_path}'. App will run in demonstration mode.")

    print(f"\n🚀 Deployment complete! Your live app is available at:")
    print(f"👉 https://huggingface.co/spaces/{space_id}")


def main():
    parser = argparse.ArgumentParser(description="Deploy Gradio app to Hugging Face Spaces.")
    parser.add_argument("--space", required=True, help="HF Space repo ID (e.g., username/voice-clone-detector)")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"), help="Hugging Face access token (or set HF_TOKEN env var)")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="Path to checkpoint to upload")
    args = parser.parse_args()

    deploy_to_spaces(space_id=args.space, token=args.token, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
