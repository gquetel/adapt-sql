import argparse
import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import sqlglot
import sqlglot.errors
import sqlparse
import sys
import torch
from scipy.stats import gmean
from scipy.spatial.distance import pdist

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer


# Resolve the dataset directory relative to the current user's home. 
# Change to actual location of datasets.
DATASETS_DIR = Path.home() / "datasets" / "superviz26-fsl"

# dtype is specified to prevent a DtypeWarning when reading the CSVs.
DTYPES = {
    "full_query": str,
    "label": int,
    "user_inputs": str,
    "attack_stage": str,
    "tamper_method": str,
    "attack_status": str,
    "statement_type": str,
    "query_template_id": str,
    "attack_id": str,
    "attack_technique": str,
    "split": str,
}


def print_vocab_size(queries, type: str, name: str) -> dict:
    """Lexical diversity: vocabulary size and Type-Token Ratio.

    Returns a dict of the metrics so the caller can aggregate them into the
    per-mode results CSV (the per-vocab .txt dump is kept as a side artifact).
    """
    if not queries:
        print(f"Vocabulary for {name} {type} queries: skipped (empty pool).")
        return {"vocab_size": 0, "token_count": 0, "ttr": float("nan")}

    v = CountVectorizer()
    X = v.fit_transform(queries)
    vocab_size = len(v.vocabulary_)
    print(f"Vocabulary size for {name} {type} queries: {vocab_size}")

    token_count = int(X.sum())
    ttr = vocab_size / token_count if token_count else 0
    print(f"Type-Token Ratio (TTR) for {name} {type} queries: {ttr:.4f}")

    with open(f"vocab-{name}-{type}.txt", "w") as f:
        for word, idx in sorted(v.vocabulary_.items(), key=lambda x: x[1]):
            f.write(f"{idx}: {word}\n")

    return {"vocab_size": vocab_size, "token_count": token_count, "ttr": ttr}


def print_unique_pts(queries: list, type: str, name: str) -> dict:
    pts = {}
    cnt_prserr = 0

    logging.disable(sys.maxsize)
    for q in tqdm(queries):
        try:
            glot_trees = sqlglot.parse(q, dialect="mysql")
            for glot_tree in glot_trees:
                if glot_tree == None or isinstance(glot_tree, sqlglot.exp.Command):
                    # A Command is returned, the tool didn't manage to parse the query
                    # correctly, ignore those.
                    cnt_prserr += 1
                    continue

                # Replace all literals or identifier to get a canonical representation.
                # "Normalize" parse trees.
                for i in glot_tree.find_all(
                    sqlglot.exp.Identifier | sqlglot.exp.Literal | sqlglot.exp.Comment
                ):
                    i.set("this", "I")

                for i in glot_tree.find_all(sqlglot.exp.HexString):
                    i.set("this", "0")

                # print(repr(glot_tree))
                canon_tree = glot_tree.sql(comments=False)
                if canon_tree not in pts:
                    pts[canon_tree] = 1
                else:
                    pts[canon_tree] += 1
        except sqlglot.errors.ParseError as e:
            cnt_prserr += 1
        except sqlglot.errors.TokenError as e:
            cnt_prserr += 1
        except KeyError as e:
            cnt_prserr += 1

    logging.disable(logging.NOTSET)

    if cnt_prserr > 0:
        print(f"There were {cnt_prserr} parsing errors during processing.")
    s_keys = sorted(pts)
    with open(f"parse-trees-{name}-{type}.txt", "w") as f:
        for e in s_keys:
            f.write(f"{e}: {pts[e]}\n")
    print(f"Number of unique parse trees for {name} {type} queries: {len(pts)}")

    return {"n_unique_parse_trees": len(pts), "n_parse_errors": cnt_prserr}


# Sentence-BERT model with the best average score in the SBERT pretrained models
# table: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"


def compute_embeddings(df: pd.DataFrame):
    """Compute Sentence-BERT embeddings of queries (column 'full_query').

    This is a one-time experiment, so embeddings are recomputed every call (no
    caching).

    The model applies mean pooling over the token embeddings. Inputs longer
    than the model limit (384 tokens) are truncated.

    Args:
        df (pd.DataFrame): frame with a 'full_query' column to embed.
    """
    queries = df["full_query"].to_list()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(SBERT_MODEL, device=device)

    # We compute embeddings by batches, they should not be too big because
    # they might be bigger than memory.
    return model.encode(
        queries,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def print_dataset_tsne(
    df: pd.DataFrame, type: str, name: str, n_sampling: None | int = None
):
    if n_sampling:
        df = df.sample(n_sampling, random_state=42)

    queries = df["full_query"].to_list()
    embeddings = compute_embeddings(df)

    # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    # Let's use default params as much as possible.
    # We set perplexity to 50 as the doc states that higher dimensions requires
    # higher values.
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(50, len(queries) - 1),
        verbose=1,
        n_jobs=-1,
    )
    tsne_embeddings = tsne.fit_transform(embeddings)

    # Save the results to allow to compute the figure with all datasets later.
    results = {
        "queries": queries,
        "embeddings": embeddings,
        "tsne_embeddings": tsne_embeddings,
        "type": type,
        "name": name,
    }

    print(f"t-SNE results saved to tsne-{name}-{type}.pkl")

    with open(f"../output/tsne-{name}-{type}.pkl", "wb") as f:
        pickle.dump(results, f)

    # Now plot individual results.
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        tsne_embeddings[:, 0],
        tsne_embeddings[:, 1],
        alpha=0.6,
        s=20,
    )

    plt.title(
        f"t-SNE Visualization of {name} {type} \n"
        f"Using Sentence-BERT ({SBERT_MODEL}) Embeddings (n={len(queries)})"
    )
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, alpha=0.3)
    legend_label = f"{type.capitalize()} Queries"
    plt.legend([scatter], [legend_label])

    plt.tight_layout()
    plt.savefig(f"../output/tsne-{name}-{type}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved to ../output/tsne-{name}-{type}.png")


def print_div_sem(
    df: pd.DataFrame,
    type: str,
    name: str,
    sample_size: int = 5000,
    n_repeats: int = 10,
):
    """Diversity metric from: https://aclanthology.org/2024.findings-naacl.228.pdf

    div_sem is the mean pairwise cosine distance over the embeddings. Estimating
    it from a single random sample is noisy: the value swings depending on which
    queries are drawn. To stabilise it we embed the whole pool *once* (the
    expensive step) and then average the metric over `n_repeats` random subsamples
    of `sample_size` queries drawn from that cached pool. Averaging shrinks the
    standard deviation of the estimate by ~sqrt(n_repeats), and we also report
    the std so a difference between datasets can be told apart from sampling noise.

    Args:
        df (pd.DataFrame): _description_
        type (str): _description_
        name (str): _description_
        sample_size (int): number of queries per subsample.
        n_repeats (int): number of subsamples to average over.
    """

    # A pairwise distance needs at least two points. A pool can be empty (e.g. a
    # dataset without attacks) -- skip it cleanly (before the costly embedding
    # step) instead of crashing on a 0-row array.
    if len(df) < 2:
        print(
            f"Semantic Diversity of {type} for dataset {name}: skipped "
            f"(only {len(df)} queries available, need >= 2)."
        )
        return {
            "div_sem_mean": float("nan"),
            "div_sem_std": float("nan"),
            "div_sem_n": len(df),
        }

    _embeddings = compute_embeddings(df=df)
    n = len(_embeddings)

    # If the pool is no larger than the sample size there is nothing to resample,
    # so just compute the metric once on everything.
    if n <= sample_size:
        div_sem = float(np.mean(pdist(_embeddings, metric="cosine")))
        print(
            f"Semantic Diversity of {type} for dataset {name} using cosine distance "
            f"(n={n}, single pass): {div_sem}"
        )
        return {"div_sem_mean": div_sem, "div_sem_std": 0.0, "div_sem_n": n}

    # Fixed rng so the reported value is reproducible; averaging is what reduces
    # the variance across the (upstream) data seed.
    rng = np.random.default_rng(0)
    values = []
    for _ in range(n_repeats):
        idx = rng.choice(n, size=sample_size, replace=False)
        values.append(np.mean(pdist(_embeddings[idx], metric="cosine")))

    values = np.array(values)
    print(
        f"Semantic Diversity of {type} for dataset {name} using cosine distance "
        f"({n_repeats}x{sample_size} from pool of {n}): "
        f"{values.mean():.6f} +/- {values.std():.6f}"
    )
    return {
        "div_sem_mean": float(values.mean()),
        "div_sem_std": float(values.std()),
        "div_sem_n": n,
    }


def load_wafamole_samples(fp_sane: str, fp_attacks: str):
    # This is too long to parse each time, let's also save them as pickles.
    fp_patks = "../output/parsed-wafamole-attacks.pkl"
    fp_psane = "../output/parsed-wafamole-sane.pkl"

    if os.path.isfile(fp_patks):
        attacks = pd.read_pickle(fp_patks)
    else:
        attack = open(fp_attacks, "r").read()
        attacks = sqlparse.split(attack)
        pd.to_pickle(attacks, fp_patks)

    if os.path.isfile(fp_psane):
        sanes = pd.read_pickle(fp_psane)
    else:
        sane = open(fp_sane, "r").read()
        sanes = sqlparse.split(sane)
        pd.to_pickle(sanes, fp_psane)

    df_sane = pd.DataFrame(sanes, columns=["full_query"])
    df_attack = pd.DataFrame(attacks, columns=["full_query"])

    df_sane = df_sane.assign(label=0)
    df_attack = df_attack.assign(label=1)

    return pd.concat([df_sane, df_attack])


def load_kaggle_samples(fp: Path) -> pd.DataFrame:
    """Load the Kaggle SQL injection dataset (columns 'Query' and 'Label')."""
    df = pd.read_csv(fp)
    return df.rename(columns={"Query": "full_query", "Label": "label"})


def process_dataset(
    df: pd.DataFrame,
    name: str,
    query_column: str = "full_query",
    label_column: str = "label",
    split_column: str | None = "split",
    sem_sample_size: int = 20000,
    vocab: bool = True,
    parse_trees: bool = True,
    div_sem: bool = True,
) -> list:
    """Compute lexical (vocab), syntactic (parse trees) and semantic (div_sem)
    diversity for one dataset.

    Source layout (see experiments/build_big_trainsets.py): the *train* split is
    benign-only and the *test* split holds both normal queries and attacks of the
    target domain. We only measure the target domain, so we draw both the normal
    and the attack pools from the test split. The test split of a LODO dataset
    (e.g. BCD-A) is the same as the test split of the in-domain dataset (A-A),
    thus the in-domain datasets are sufficient.

    Lexical and syntactic metrics are cheap enough to run on the *full* normal and
    attack pools. Semantic diversity (the expensive Sentence-BERT embedding step) is
    run on a random sub-pool of `sem_sample_size` queries per label.

    All three metrics default to on so a single call reports lex + synt + sem.

    External datasets (Kaggle, WAF-A-MoLE) have no split: set `split_column` to
    None to draw each pool from all the queries with that label.

    Returns one row dict per label (normal, attack) with every computed metric, so
    the caller can aggregate the rows into the per-mode results CSV.
    """
    # (label type, source split, frame) for the two pools we measure.
    if split_column is None:
        pools = [
            ("normal", "all", df[df[label_column] == 0]),
            ("attack", "all", df[df[label_column] == 1]),
        ]
    else:
        pools = [
            ("normal", "test", df[(df[split_column] == "test") & (df[label_column] == 0)]),
            ("attack", "test", df[(df[split_column] == "test") & (df[label_column] == 1)]),
        ]

    rows = []
    for qtype, src_split, pool in pools:
        queries = pool[query_column].tolist()
        print(f"[{name}] {qtype} pool ({src_split} split): {len(queries)} queries")

        row = {
            "dataset": name,
            "type": qtype,
            "source_split": src_split,
            "n_queries": len(queries),
        }

        # Lexical and syntactic metrics run on the full pool.
        if vocab:
            row.update(print_vocab_size(queries, qtype, name))
        if parse_trees:
            row.update(print_unique_pts(queries, qtype, name))

        # Semantic metric runs on a (capped) random sub-pool. print_div_sem
        # further resamples within this pool to stabilise the estimate.
        if div_sem:
            pool_sem = pool.rename(columns={query_column: "full_query"})
            if len(pool_sem) > sem_sample_size:
                pool_sem = pool_sem.sample(n=sem_sample_size, random_state=2)
            row.update(print_div_sem(pool_sem, qtype, name))

        rows.append(row)

    return rows


# In-domain scenario -> filename map. We do not process the LODO datasets: their
# test split is the same as the in-domain one (see process_dataset).
INDOMAIN_DATASETS = {
    "A-A": "a-a.csv",
    "B-B": "b-b.csv",
    "C-C": "c-c.csv",
    "D-D": "d-d.csv",
}

# External datasets. Change to actual location of datasets.
KAGGLE_PATH = Path.home() / "datasets" / "kaggle" / "Modified_SQL_Dataset.csv"
WAFAMOLE_DIR = Path.home() / "repos" / "wafamole_dataset"

METRICS = ["lex", "synt", "sem"]


def write_results(rows: list, results_filename: str):
    results = pd.DataFrame(rows)
    out_path = Path("../output") / results_filename
    results.to_csv(out_path, index=False)
    print(f"Wrote aggregated metrics ({len(results)} rows) to {out_path}")


def process_datasets(datasets: dict, results_filename: str, metrics: list):
    """Compute the selected diversity metrics for each dataset and write one
    aggregated CSV (`results_filename`, under ../output), one row per
    (dataset, label type).

    For every CSV, process_dataset draws normal queries and attacks from the test
    split, computes lexical/syntactic metrics on the full pools and semantic
    diversity on a 20k sub-pool per label (see process_dataset for the why).
    """
    rows = []
    for name, filename in datasets.items():
        df = pd.read_csv(DATASETS_DIR / filename, dtype=DTYPES)
        rows.extend(process_dataset(df=df, name=name, **metric_flags(metrics)))

    write_results(rows, results_filename)


def metric_flags(metrics: list) -> dict:
    return {
        "vocab": "lex" in metrics,
        "parse_trees": "synt" in metrics,
        "div_sem": "sem" in metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute lexical (vocab), syntactic (parse trees) and semantic "
            "(div_sem) diversity of the SQLIA datasets. Normal queries and "
            "attacks come from the test split (target domain); lexical/syntactic "
            "run on the full pools, semantic on a 20k sub-pool per label. Choose "
            "'indomain' for A->A, B->B, ... 'kaggle' and 'wafamole' process the "
            "external datasets, which have no split."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["indomain", "kaggle", "wafamole"],
        nargs="+",
        required=True,
        help="Which datasets to process (one or more).",
    )
    parser.add_argument(
        "--metrics",
        choices=METRICS,
        nargs="+",
        default=METRICS,
        help="Which metrics to compute (default: all).",
    )
    args = parser.parse_args()

    Path("../output").mkdir(exist_ok=True, parents=True)

    # A partial run must not overwrite the results of a full run.
    suffix = "" if set(args.metrics) == set(METRICS) else "_" + "-".join(args.metrics)

    for mode in args.mode:
        results_filename = f"results_{mode}{suffix}.csv"
        if mode == "indomain":
            process_datasets(INDOMAIN_DATASETS, results_filename, args.metrics)
        elif mode == "kaggle":
            df = load_kaggle_samples(KAGGLE_PATH)
            rows = process_dataset(
                df=df, name="Kaggle", split_column=None, **metric_flags(args.metrics)
            )
            write_results(rows, results_filename)
        elif mode == "wafamole":
            df = load_wafamole_samples(
                fp_sane=WAFAMOLE_DIR / "sane.sql",
                fp_attacks=WAFAMOLE_DIR / "attacks.sql",
            )
            rows = process_dataset(
                df=df, name="WAF-A-MoLE", split_column=None, **metric_flags(args.metrics)
            )
            write_results(rows, results_filename)


if __name__ == "__main__":
    main()
