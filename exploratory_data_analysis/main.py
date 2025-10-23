import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from load_data import load_annotations, compute_class_stats, objects_to_df
from collections import Counter
from utils import compute_visibility, cooccurrence_matrix
# --- Default file paths (can be overridden by user) ---
DEFAULT_TRAIN_JSON = "../dataset/train/labels.json"
DEFAULT_VAL_JSON   = "../dataset/val/labels.json"
DEFAULT_TRAIN_IMG  = "../dataset/train/images"
DEFAULT_VAL_IMG    = "../dataset/val/images"

if __name__ == "__main__":
    st.set_page_config(page_title="BDD100K Exploratory Data Analysis", layout="wide")
    st.title("BDD100K Object Detection — EDA Dashboard")

    #############################################################################    
    st.sidebar.header("Data Source Configuration")
    train_json = st.sidebar.text_input("Train JSON path:", value=DEFAULT_TRAIN_JSON)
    val_json   = st.sidebar.text_input("Validation JSON path:", value=DEFAULT_VAL_JSON)
    train_img_root = st.sidebar.text_input("Train images directory:", value=DEFAULT_TRAIN_IMG)
    val_img_root   = st.sidebar.text_input("Validation images directory:", value=DEFAULT_VAL_IMG)

    st.sidebar.markdown("You can modify these paths if your dataset is in a different location.")

    if not (train_json and val_json and os.path.exists(train_json) and os.path.exists(val_json)):
        st.warning("Please provide valid train and validation JSON paths.")
        st.stop()

    train_objs, classes = load_annotations(train_json)
    val_objs, _ = load_annotations(val_json)
    st.success(f"Loaded {len(train_objs)} train and {len(val_objs)} validation images, {len(classes)} classes.")

    #############################################################################
    st.subheader(" Class Frequency Analysis")

    train_counts, _ = compute_class_stats(train_objs)
    val_counts, _ = compute_class_stats(val_objs)

    df_compare = pd.DataFrame({
        "train": train_counts,
        "val": val_counts
    }).fillna(0)

    df_compare["avg"] = (df_compare["train"] + df_compare["val"]) / 2
    df_compare = df_compare.sort_values("train", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Show top-K classes", 5, len(df_compare), 10)
    with col2:
        log_scale = st.checkbox("Log scale (helps small classes)", value=False)

    df_top = df_compare.head(top_k)
    x = np.arange(len(df_top))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_train = ax.bar(x - width/2, df_top["train"], width, label="Train", color="skyblue")
    bars_val = ax.bar(x + width/2, df_top["val"], width, label="Validation", color="salmon")

    ax.set_xticks(x)
    ax.set_xticklabels(df_top.index, rotation=45, ha="right")
    ax.set_yscale("log" if log_scale else "linear")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Objects")
    ax.set_title("Class Frequency: Train vs Validation")
    ax.legend()
    plt.tight_layout()

    # Add numeric labels on bars
    for bar in bars_train + bars_val:
        height = bar.get_height()
        ax.annotate(f"{int(height)}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    st.pyplot(fig)

    #############################################################################
    st.subheader("Average Object Frequency (Mean of Train + Val)")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    bars_avg = ax2.bar(df_top.index, df_top["avg"], color="mediumseagreen")
    ax2.set_xticklabels(df_top.index, rotation=45, ha="right")
    ax2.set_yscale("log" if log_scale else "linear")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Average Frequency")
    ax2.set_title("Average Object Frequency Across Splits")

    for bar in bars_avg:
        height = bar.get_height()
        ax2.annotate(f"{int(height)}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig2)

    #############################################################################
    st.subheader("Class Coverage (Train Set)")

    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ax3.pie(df_top["train"], labels=df_top.index, autopct='%1.1f%%', startangle=90)
    ax3.set_title("Train Set Class Coverage (Top-K)")
    st.pyplot(fig3)

    imbalance_ratio = df_compare["train"].max() / (df_compare["train"].min() + 1e-6)
    st.markdown(f"⚖️ **Imbalance Ratio (max/min frequency)**: `{imbalance_ratio:.2f}`")
    if imbalance_ratio > 10:
        st.warning("Severe class imbalance detected — consider weighted loss or augmentation.")
    elif imbalance_ratio > 3:
        st.info("Moderate imbalance — could affect minority classes.")
    else:
        st.success("Balanced dataset distribution.")

    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    st.download_button(
        label="⬇️ Download Train–Val Frequency Chart",
        data=buf.getvalue(),
        file_name="class_frequency_train_val.png",
        mime="image/png"
    )

    #############################################################################
    st.subheader("Average Class Frequency (Mean of Train + Val)")

    df_compare["avg"] = (df_compare["train"] + df_compare["val"]) / 2
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(df_compare.index, df_compare["avg"], color="mediumseagreen")
    ax2.set_xticklabels(df_compare.index, rotation=45, ha="right")
    ax2.set_title("Average Object Frequency Across Splits")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Average Frequency")
    plt.tight_layout()
    st.pyplot(fig2)

    # Combine train and val counts
    combined_counts = Counter(train_counts)
    for c, v in val_counts.items():
        combined_counts[c] += v

    df_combined = pd.DataFrame({
        "class": list(combined_counts.keys()),
        "total_objects": list(combined_counts.values())
    }).sort_values("total_objects", ascending=False)

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df_combined["class"], df_combined["total_objects"])
    ax.set_title("Object Frequency Distribution (All Splits Combined)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Total Objects")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)


    #############################################################################   
    if train_img_root and val_img_root:
        st.subheader("Mean Visibility Comparison")
        df_train_vis = compute_visibility(train_objs, train_img_root)
        df_val_vis = compute_visibility(val_objs, val_img_root)
        merged = df_train_vis.merge(df_val_vis, on="class", suffixes=("_train", "_val"))
        st.dataframe(merged, use_container_width=True)

        # Smart difficulty
        st.subheader("Detection Difficulty Ranking (Combined Train + Val)")
        merged["mean_visibility"] = merged[["mean_visibility(%)_train", "mean_visibility(%)_val"]].mean(axis=1)
        merged["total_samples"] = merged["sample_count_train"] + merged["sample_count_val"]
        vis_mean = merged["mean_visibility"].mean()
        count_median = merged["total_samples"].median()
        easy, moderate, hard = [], [], []
        for _, r in merged.iterrows():
            if r["mean_visibility"] >= vis_mean and r["total_samples"] >= count_median:
                easy.append(r["class"])
            elif r["mean_visibility"] < vis_mean and r["total_samples"] < count_median:
                hard.append(r["class"])
            else:
                moderate.append(r["class"])
        st.markdown(f"Easy: {', '.join(easy)}")
        st.markdown(f"Moderate: {', '.join(moderate)}")
        st.markdown(f"Hard: {', '.join(hard)}")

    #############################################################################
    st.subheader("Object Density vs Visibility Correlation")
    df_train = objects_to_df(train_objs)
    objs_per_img = df_train.groupby("image").size().mean()
    corr = df_train_vis["mean_visibility(%)"].corr(df_train_vis["sample_count"])
    st.write(f"Average objects per image: {objs_per_img:.2f}")
    st.write(f"Correlation between visibility and sample count: {corr:.2f}")

    #############################################################################
    st.subheader("Aspect Ratio Distribution (Train)")
    ratios = []
    for labels in train_objs.values():
        for o in labels:
            w, h = o["x_max"] - o["x_min"], o["y_max"] - o["y_min"]
            if h > 0:
                ratios.append(w / h)
    plt.figure(figsize=(6, 4))
    plt.hist(ratios, bins=30)
    plt.xlabel("Aspect Ratio (w/h)")
    plt.ylabel("Count")
    plt.title("Aspect Ratio Distribution")
    st.pyplot(plt)

    #############################################################################
    st.subheader("Inter-Class Confusion Potential")
    co_train = cooccurrence_matrix(train_objs, classes)
    similar_classes = []
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            c1, c2 = classes[i], classes[j]
            if co_train.loc[c1, c2] > 500:  # arbitrary threshold
                similar_classes.append((c1, c2))
    st.write(f"Highly co-occurring pairs (possible confusion): {similar_classes[:10]}")

    #############################################################################
    st.subheader("Interactive Filters")
    min_vis, max_vis = st.slider("Visibility range (%)", 0.0, 10.0, (0.0, 5.0))
    filtered = df_train_vis[(df_train_vis["mean_visibility(%)"] >= min_vis) &
                            (df_train_vis["mean_visibility(%)"] <= max_vis)]
    st.dataframe(filtered, use_container_width=True)

    #############################################################################
    st.subheader("Automated Summary Insights")
    st.write(f"- Dominant classes: {', '.join(df_compare['train'].sort_values(ascending=False).head(3).index)}")
    st.write(f"- Rare classes: {', '.join(df_compare['train'].sort_values().head(3).index)}")
    st.write(f"- Easy detection classes: {', '.join(easy)}")
    st.write(f"- Hard detection classes: {', '.join(hard)}")
    st.write("Dataset balanced across splits; visibility variation mostly mild.")

    st.markdown("---")
    st.caption("EDA Dashboard • Mritunjoy Halder")


