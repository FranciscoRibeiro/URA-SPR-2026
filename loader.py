import json
import os


# -----------------------------
def load_jsonl(file_path):
    
    tasks = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            
            line = line.strip()
            if not line:
                continue
                
            raw = json.loads(line)
            task_id = raw.get("task_id", "")
            impls = raw.get("implementations", [])
            
            canonical_code = None
            incorrects = []
            
            for impl in impls:
                if not isinstance(impl, dict):
                    continue
                    
                label = impl.get("label", 0)
                code = impl.get("code", "")
                
                if label == 1:
                    canonical_code = code
                else:
                    incorrects.append(code)
            
            if canonical_code is not None and incorrects:
                tasks.append({
                    "task_id": task_id,
                    "canonical": canonical_code,
                    "incorrects": incorrects
                })

    return tasks


# -----------------------------
def extract_archives(dataset_root):
    import subprocess
    
    print("Checking for unextracted dataset archives...")
    for root, _, files in os.walk(dataset_root):
        for file in files:
            if file.endswith(".tar.zst"):
                archive_path = os.path.join(root, file)
                # Check if we already extracted it. We'll look for a directory with the same name (minus .tar.zst)
                folder_name = file.replace(".tar.zst", "")
                folder_path = os.path.join(root, folder_name)
                
                has_folds = any(os.path.isdir(os.path.join(root, d)) for d in os.listdir(root) if d.startswith("fold_"))
                
                if not has_folds:
                    print(f"Extracting {archive_path}...")
                    try:
                        subprocess.run(["tar", "-xf", archive_path, "-C", root], check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Failed to extract {archive_path}: {e}")

# -----------------------------
def get_dataset_files(dataset_root="dataset"):
    
    extract_archives(dataset_root)
    
    files = []
    
    # ICSE26 benchmarks
    icse26_dir = os.path.join(dataset_root, "icse26")
    if os.path.isdir(icse26_dir):
        for bench in ["bigcodebench", "humaneval"]:
            bench_root = os.path.join(icse26_dir, bench, "icse26")
            if os.path.isdir(bench_root):
                for fold_idx in range(10):
                    fold_dir = os.path.join(bench_root, f"fold_{fold_idx}")
                    if os.path.isdir(fold_dir):
                        for split_type in ["fit", "validate", "test"]:
                            path = os.path.join(fold_dir, f"{split_type}.jsonl")
                            if os.path.isfile(path):
                                files.append({
                                    "source": f"icse26/{bench}",
                                    "fold": fold_idx,
                                    "split": split_type,
                                    "path": path
                                })

    # Phase2 splits
    phase2_dir = os.path.join(dataset_root, "bigcodebench", "phase2")
    if os.path.isdir(phase2_dir):
        for split_name in sorted(os.listdir(phase2_dir)):
            split_root = os.path.join(phase2_dir, split_name)
            if os.path.isdir(split_root) and not split_name.startswith("."):
                for fold_idx in range(10):
                    fold_dir = os.path.join(split_root, f"fold_{fold_idx}")
                    if os.path.isdir(fold_dir):
                        for split_type in ["fit", "validate", "test"]:
                            path = os.path.join(fold_dir, f"{split_type}.jsonl")
                            if os.path.isfile(path):
                                files.append({
                                    "source": f"phase2/{split_name}",
                                    "fold": fold_idx,
                                    "split": split_type,
                                    "path": path
                                })
                                
    return files
