import os
import csv
from loader import get_dataset_files, load_jsonl
import metrics


# -----------------------------
def initialize_models():
    
    print(f"\nHardware Detection: Using device '{metrics.get_device()}'")
    if str(metrics.get_device()) == "mps":
        print("✅ SUCCESS: Apple Silicon GPU (Metal Performance Shaders) is active and will be used!\n")
    
    print("Loading CodeBERT...")
    cb_tokenizer, cb_model, cb_device = metrics.load_embedding_model("microsoft/codebert-base")
    
    print("Loading UniXcoder...")
    ux_tokenizer, ux_model, ux_device = metrics.load_embedding_model("microsoft/unixcoder-base")
    
    return {
        "codebert": (cb_tokenizer, cb_model, cb_device),
        "unixcoder": (ux_tokenizer, ux_model, ux_device)
    }


# -----------------------------
def get_embeddings(code, models, cache):
    
    if code in cache:
        return cache[code]
        
    cb_tokenizer, cb_model, cb_device = models["codebert"]
    cb_emb = metrics.get_embedding_codebert(code, cb_tokenizer, cb_model, cb_device)
    
    ux_tokenizer, ux_model, ux_device = models["unixcoder"]
    ux_emb = metrics.get_embedding_unixcoder(code, ux_tokenizer, ux_model, ux_device)
    
    cache[code] = (cb_emb, ux_emb)
    return cache[code]


# -----------------------------
def process_tasks(files, models):
    from tqdm import tqdm
    import os
    
    # Initialize the output file
    output_file = "results/all_metrics.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Wipe the existing file if it exists so we start fresh
    if os.path.exists(output_file):
        os.remove(output_file)
        
    embedding_cache = {}
    
    total_files = len(files)
    
    # Use tqdm to give a nice progress bar with an ETA
    for file_idx, file_info in enumerate(tqdm(files, desc="Processing Files", unit="file"), start=1):
        
        file_results = []
        path = file_info["path"]
        
        tasks = load_jsonl(path)
        
        for task in tasks:
            
            task_id = task["task_id"]
            canonical = task["canonical"]
            incorrects = task["incorrects"]
            
            cb_canon, ux_canon = get_embeddings(canonical, models, embedding_cache)
            
            for impl_idx, impl in enumerate(incorrects, start=1):
                
                # Levenshtein
                char_raw, char_norm = metrics.compute_char_levenshtein(canonical, impl)
                tok_raw, tok_norm = metrics.compute_token_levenshtein(canonical, impl)
                
                # AST
                ast_raw, ast_norm = metrics.compute_ast_diff(canonical, impl)
                
                # Embeddings
                cb_impl, ux_impl = get_embeddings(impl, models, embedding_cache)
                cb_cos = metrics.compute_cosine_similarity(cb_canon, cb_impl)
                ux_cos = metrics.compute_cosine_similarity(ux_canon, ux_impl)
                
                file_results.append({
                    "source_dataset": file_info["source"],
                    "fold": file_info["fold"],
                    "split_type": file_info["split"],
                    "task_id": task_id,
                    "impl_index": impl_idx,
                    "char_lev_raw": char_raw,
                    "char_lev_norm": char_norm,
                    "token_lev_raw": tok_raw,
                    "token_lev_norm": tok_norm,
                    "ast_changed_nodes": ast_raw,
                    "ast_norm_ratio": ast_norm,
                    "codebert_cosine": cb_cos,
                    "unixcoder_cosine": ux_cos
                })
                
        # --- Incremental Save ---
        if file_results:
            file_exists = os.path.exists(output_file)
            headers = list(file_results[0].keys())
            with open(output_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(file_results)


# -----------------------------
def save_results(results, output_file="results/all_metrics.csv"):
    
    if not results:
        print("No results to save.")
        return
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    headers = list(results[0].keys())
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Results saved to {output_file}")


# -----------------------------
def main():
    
    print("Discovering dataset files...")
    files = get_dataset_files("dataset")
    print(f"Found {len(files)} files to process.")
    
    if not files:
        print("No files found. Ensure the 'dataset' folder is present.")
        return
        
    models = initialize_models()
    
    print("Starting processing. Results will be saved incrementally to results/all_metrics.csv")
    process_tasks(files, models)
    print("Finished processing all files!")


# -----------------------------
if __name__ == "__main__":
    main()
