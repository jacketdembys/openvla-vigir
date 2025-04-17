from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="openvla/openvla-7b",
    local_dir="/root/huggingface_models/openvla-7b",
    local_dir_use_symlinks=False,  # Set to False to get real files (not symlinks)
    revision="main"  # or a specific tag/commit if needed
)

print(f"Model downloaded to: {local_dir}")
