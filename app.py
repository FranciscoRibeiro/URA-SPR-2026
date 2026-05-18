import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
import glob


st.set_page_config(page_title="Distance Analysis Dashboard", layout="wide")


# -----------------------------
@st.cache_data
def load_data():
    
    all_files = glob.glob("results/*.csv")
    if not all_files:
        return pd.DataFrame()
    
    df_list = []
    for f in all_files:
        df_list.append(pd.read_csv(f))
    
    df = pd.concat(df_list, ignore_index=True)
    
    def minmax(s):
        s_clean = s.dropna()
        if len(s_clean) == 0:
            return s
        diff = s_clean.max() - s_clean.min()
        if diff == 0:
            return s * 0
        return (s - s_clean.min()) / diff

    metrics = [
        "char_lev_raw", "char_lev_norm", 
        "token_lev_raw", "token_lev_norm", 
        "ast_changed_nodes", "ast_norm_ratio",
        "codebert_cosine", "unixcoder_cosine"
    ]
    
    for m in metrics:
        if m not in df.columns:
            df[m] = pd.NA
        df[f"{m}_scaled"] = minmax(df[m])
        
    if "source_dataset" in df.columns:
        df["main_folder"] = df["source_dataset"].apply(lambda x: x.split('/')[0] if isinstance(x, str) and '/' in x else 'unknown')
        df["sub_dataset"] = df["source_dataset"].apply(lambda x: x.split('/', 1)[1] if isinstance(x, str) and '/' in x else x)
        
    return df


# -----------------------------
def get_metrics(scale, core_metrics):
    
    if scale:
        return [f"{m}_scaled" for m in core_metrics]
    return core_metrics


# -----------------------------
def main():
    
    df = load_data()

    if df.empty:
        st.error("No CSV files found in the 'results/' directory. Please run the analysis script first.")
        st.stop()

    core_metrics = [
        "char_lev_raw", "char_lev_norm", 
        "token_lev_raw", "token_lev_norm", 
        "ast_changed_nodes", "ast_norm_ratio"
    ]

    if "codebert_cosine" in df.columns and not df["codebert_cosine"].isna().all():
        core_metrics.extend(["codebert_cosine", "unixcoder_cosine"])

    st.title("Dataset Distance & Similarity Analysis")

    # --- Global Controls ---
    col_scale, col_main = st.columns(2)
    with col_scale:
        scale = st.checkbox("Use min-max scaling for metrics", value=False)

    all_main_folders = sorted(df["main_folder"].dropna().unique()) if "main_folder" in df.columns else []

    main_opts = ["All"] + all_main_folders
    main_idx = main_opts.index("icse26") if "icse26" in main_opts else 0

    with col_main:
        selected_main_folder = st.selectbox("Main Category (Phase2 / ICSE26)", main_opts, index=main_idx)

    if selected_main_folder != "All":
        global_df = df[df["main_folder"] == selected_main_folder]
    else:
        global_df = df

    all_datasets = sorted(global_df["sub_dataset"].dropna().unique()) if "sub_dataset" in global_df.columns else []
    all_folds  = sorted(global_df["fold"].dropna().unique()) if "fold" in global_df.columns else []
    all_splits = sorted(global_df["split_type"].dropna().unique()) if "split_type" in global_df.columns else []

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    dataset_opts = ["All"] + all_datasets
    dataset_idx = dataset_opts.index("bigcodebench") if "bigcodebench" in dataset_opts else 0

    with col1:
        selected_dataset = st.selectbox("Dataset", dataset_opts, index=dataset_idx)

    fold_opts = ["All"] + all_folds
    fold_idx = fold_opts.index(0) if 0 in fold_opts else 0

    with col2:
        selected_fold = st.selectbox("Fold", fold_opts, index=fold_idx) if all_folds else None

    split_opts = ["All"] + all_splits
    split_idx = split_opts.index("test") if "test" in split_opts else 0

    with col3:
        selected_split = st.selectbox("Split", split_opts, index=split_idx) if all_splits else None

    with col4:
        max_rows = max(10, len(global_df))
        n_rows = st.slider("Number of rows", 10, max_rows, max_rows, key="df_rows")

    # --- Filter ---
    filtered = global_df.copy()
    if selected_dataset and selected_dataset != "All":
        filtered = filtered[filtered["sub_dataset"] == selected_dataset]
    if selected_fold is not None and selected_fold != "All":
        filtered = filtered[filtered["fold"] == selected_fold]
    if selected_split is not None and selected_split != "All":
        filtered = filtered[filtered["split_type"] == selected_split]

    total_records = len(filtered)
    filtered = filtered.head(n_rows)

    st.write(f"Showing {len(filtered)} of {total_records} records")

    # --- Table ---
    st.dataframe(filtered, use_container_width=True)

    st.markdown("---")

    # --- Distribution ---
    st.header("Distribution")

    col1, col2, col3 = st.columns(3)

    with col1:
        dist_def_dataset = ["bigcodebench"] if "bigcodebench" in all_datasets else all_datasets
        dist_datasets = st.multiselect("Datasets", all_datasets, default=dist_def_dataset, key="dist_datasets")
    with col2:
        dist_def_fold = [0] if 0 in all_folds else all_folds
        dist_folds = st.multiselect("Folds", all_folds, default=dist_def_fold, key="dist_folds") if all_folds else []
    with col3:
        dist_def_split = ["test"] if "test" in all_splits else all_splits
        dist_splits = st.multiselect("Splits", all_splits, default=dist_def_split, key="dist_splits") if all_splits else []

    dist_filtered = global_df[global_df["sub_dataset"].isin(dist_datasets)]
    if dist_folds:
        dist_filtered = dist_filtered[dist_filtered["fold"].isin(dist_folds)]
    if dist_splits:
        dist_filtered = dist_filtered[dist_filtered["split_type"].isin(dist_splits)]

    bins = st.slider("Bins", 10, 100, 30, key="dist_bins")
    metrics_to_plot = get_metrics(scale, core_metrics)

    cols_per_row = 3
    for i in range(0, len(metrics_to_plot), cols_per_row):
        row_metrics = metrics_to_plot[i:i+cols_per_row]
        fig, axes = plt.subplots(1, len(row_metrics), figsize=(5 * len(row_metrics), 4))
        if len(row_metrics) == 1:
            axes = [axes]
        
        for j, metric in enumerate(row_metrics):
            metric_data = dist_filtered[metric].dropna()
            if not metric_data.empty:
                axes[j].hist(metric_data, bins=bins, edgecolor="black")
                axes[j].set_title(metric)
                axes[j].set_xlabel(metric)
                axes[j].set_ylabel("Count")
        
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    # --- Stats Table ---
    st.header("Statistics per Dataset and Split")

    stats_split_opts = ["All"] + all_splits
    stats_split_idx = stats_split_opts.index("test") if "test" in stats_split_opts else 0
    stats_split = st.selectbox("Split", stats_split_opts, index=stats_split_idx, key="stats_split") if all_splits else None

    for metric in get_metrics(scale, core_metrics):
        st.subheader(f"{metric} Statistics")
        rows = []
        for dataset in all_datasets:
            subset = global_df[(global_df["sub_dataset"] == dataset)]
            if stats_split and stats_split != "All":
                subset = subset[subset["split_type"] == stats_split]
                
            metric_subset = subset[metric].dropna()
            if not metric_subset.empty:
                rows.append({
                    "dataset": dataset,
                    "mean":   round(metric_subset.mean(),   4),
                    "median": round(metric_subset.median(), 4),
                    "min":    round(metric_subset.min(),    4),
                    "max":    round(metric_subset.max(),    4),
                    "std":    round(metric_subset.std(),    4),
                    "count":  len(metric_subset)
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")

    # --- Correlation ---
    st.header("Correlation Analysis")

    col1, col2, col3 = st.columns(3)
    with col1:
        corr_def_dataset = ["bigcodebench"] if "bigcodebench" in all_datasets else all_datasets
        corr_datasets = st.multiselect("Datasets", all_datasets, default=corr_def_dataset, key="corr_datasets")
    with col2:
        corr_def_fold = [0] if 0 in all_folds else all_folds
        corr_folds = st.multiselect("Folds", all_folds, default=corr_def_fold, key="corr_folds") if all_folds else []
    with col3:
        corr_def_split = ["test"] if "test" in all_splits else all_splits
        corr_splits = st.multiselect("Splits", all_splits, default=corr_def_split, key="corr_splits") if all_splits else []

    col_x, col_y = st.columns(2)
    with col_x:
        x_metric = st.selectbox("X axis", get_metrics(scale, core_metrics), index=0, key="corr_x")
    with col_y:
        y_metric = st.selectbox("Y axis", get_metrics(scale, core_metrics), index=min(1, len(get_metrics(scale, core_metrics))-1), key="corr_y")

    corr_filtered = global_df[global_df["sub_dataset"].isin(corr_datasets)]
    if corr_folds:
        corr_filtered = corr_filtered[corr_filtered["fold"].isin(corr_folds)]
    if corr_splits:
        corr_filtered = corr_filtered[corr_filtered["split_type"].isin(corr_splits)]

    corr_filtered = corr_filtered.dropna(subset=[x_metric, y_metric])

    if len(corr_filtered) > 1:
        pearson_r,  pearson_p  = scipy_stats.pearsonr(corr_filtered[x_metric], corr_filtered[y_metric])
        spearman_r, spearman_p = scipy_stats.spearmanr(corr_filtered[x_metric], corr_filtered[y_metric])
        kendall_r,  kendall_p  = scipy_stats.kendalltau(corr_filtered[x_metric], corr_filtered[y_metric])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pearson r",  f"{pearson_r:.4f}",  f"p = {pearson_p:.2e}")
        with col2:
            st.metric("Spearman r", f"{spearman_r:.4f}", f"p = {spearman_p:.2e}")
        with col3:
            st.metric("Kendall τ",  f"{kendall_r:.4f}",  f"p = {kendall_p:.2e}")

        sample_size = st.slider("Sample size", min_value=min(10, len(corr_filtered)), max_value=len(corr_filtered), value=min(2000, len(corr_filtered)), key="corr_sample")
        
        sample = corr_filtered.sample(n=sample_size, random_state=42)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(sample[x_metric], sample[y_metric], alpha=0.3, s=10)
        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        ax.set_title(f"{x_metric} vs {y_metric}  |  Pearson: {pearson_r:.4f}")
        
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        
        # --- Correlation Heatmap ---
        st.subheader("Correlation Matrix Heatmap")
        st.write("This matrix shows the Pearson correlation across all selected metrics simultaneously.")
        
        # Get whichever metrics are active (scaled or unscaled) based on the global checkbox
        active_metrics = get_metrics(scale, core_metrics)
        
        # Compute correlation matrix
        corr_matrix = global_df[active_metrics].corr(method='pearson')
        
        # Display as a styled dataframe with a color gradient
        st.dataframe(
            corr_matrix.style.background_gradient(cmap="coolwarm", axis=None).format("{:.3f}"),
            use_container_width=True
        )

    else:
        st.warning("Not enough data points to compute correlation.")


# -----------------------------
if __name__ == "__main__":
    main()