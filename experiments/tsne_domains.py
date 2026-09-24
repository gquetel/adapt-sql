"""t-SNE of Sentence-BERT embeddings, coloured by target domain.

We sample `--n-per-domain` attack queries and `--n-per-domain` normal queries
from the test split of each in-domain dataset (A-A, B-B, C-C, D-D), embed them
with the Sentence-BERT model of diversity_metric.py and project each label
separately with t-SNE. If the model captures the domain, the points of one
domain form a group.

t-SNE plots can show groups that do not exist in the embedding space. Thus we
also compute the silhouette score of the domain labels on the original
embeddings (cosine distance): a value near 0 means no grouping by domain, a
value near 1 means well-separated domains.

Run from experiments/ (outputs go to ../output):
    python tsne_domains.py
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from diversity_metric import (
    DATASETS_DIR,
    DTYPES,
    INDOMAIN_DATASETS,
    SBERT_MODEL,
    compute_embeddings,
)

# Letter of each in-domain dataset -> project name (see the thesis, Superviz26).
DOMAIN_NAMES = {
    "A": "OurAirports",
    "B": "Sakila",
    "C": "AdventureWorks",
    "D": "Oracle HR",
}

# Categorical slots 1-4 of the reference palette, in fixed order. Two slots have
# less than 3:1 contrast on white, so each domain also has its own marker.
DOMAIN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DOMAIN_MARKERS = ["o", "s", "^", "D"]

LABELS = {0: "normal", 1: "attack"}


def sample_queries(n_per_domain: int, seed: int) -> pd.DataFrame:
    """Sample n_per_domain queries per (domain, label) from the test splits."""
    samples = []
    for name, filename in INDOMAIN_DATASETS.items():
        df = pd.read_csv(
            DATASETS_DIR / filename,
            dtype=DTYPES,
            usecols=["full_query", "label", "split"],
        )
        df = df[df["split"] == "test"]
        for label in LABELS:
            pool = df[df["label"] == label]
            sample = pool.sample(n=n_per_domain, random_state=seed)
            samples.append(sample.assign(domain=name.split("-")[1]))
        print(f"[{name}] sampled {n_per_domain} queries per label")
    return pd.concat(samples, ignore_index=True)


def plot_panel(ax, points: np.ndarray, domains: np.ndarray, title: str):
    for i, domain in enumerate(DOMAIN_NAMES):
        mask = domains == domain
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=18,
            alpha=0.8,
            color=DOMAIN_COLORS[i],
            marker=DOMAIN_MARKERS[i],
            edgecolors="white",
            linewidths=0.4,
            label=f"{domain} ({DOMAIN_NAMES[domain]})",
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "t-SNE of Sentence-BERT embeddings of normal and attack queries, "
            "coloured by target domain."
        )
    )
    parser.add_argument("--n-per-domain", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path("../output")
    out_dir.mkdir(exist_ok=True, parents=True)

    df = sample_queries(args.n_per_domain, args.seed)
    embeddings = compute_embeddings(df)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    results = {"model": SBERT_MODEL, "seed": args.seed}
    for ax, (label, label_name) in zip(axes, LABELS.items()):
        mask = (df["label"] == label).to_numpy()
        emb = embeddings[mask]
        domains = df.loc[mask, "domain"].to_numpy()

        silhouette = silhouette_score(emb, domains, metric="cosine")
        print(f"Silhouette score by domain ({label_name}): {silhouette:.4f}")

        # Perplexity 30 is the default value; it must stay below the number of
        # points.
        tsne = TSNE(
            n_components=2,
            metric="cosine",
            init="pca",
            perplexity=min(30, len(emb) - 1),
            random_state=args.seed,
        )
        points = tsne.fit_transform(emb)

        plot_panel(
            ax,
            points,
            domains,
            f"{label_name.capitalize()} queries "
            f"(n={len(emb)}, silhouette={silhouette:.3f})",
        )
        results[label_name] = {
            "queries": df.loc[mask, "full_query"].tolist(),
            "domains": domains,
            "embeddings": emb,
            "tsne": points,
            "silhouette": silhouette,
        }

    # One legend for both panels: the domains are the same.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"t-SNE of {SBERT_MODEL} embeddings by target domain")
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"tsne-domains.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "tsne-domains.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"Figure and data saved to {out_dir}/tsne-domains.{{png,pdf,pkl}}")


if __name__ == "__main__":
    main()
