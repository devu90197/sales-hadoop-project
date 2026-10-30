"""Data steps for the project: cleaning, MapReduce and K-Means. The website is in app.py."""
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_FILE = "data/online_retail.csv.gz"


# ---------- Step 1: cleaning ----------

def load_raw():
    return pd.read_csv(DATA_FILE, dtype={"InvoiceNo": str, "StockCode": str}, parse_dates=["InvoiceDate"])


def clean(raw):
    """Applies each cleaning rule in turn and records how many rows it removed."""
    steps = []
    df = raw

    def apply(name, reason, keep):
        nonlocal df
        before = len(df)
        df = df[keep(df)]
        steps.append({"Rule": name, "Why": reason, "Rows removed": before - len(df), "Rows left": len(df)})

    apply("Remove exact duplicates", "The same line recorded twice would count sales twice.",
          lambda d: ~d.duplicated())
    apply("Remove rows with no customer ID", "Customers cannot be segmented without an ID.",
          lambda d: d["CustomerID"].notna())
    apply("Remove cancelled invoices", "Invoice numbers starting with C are refunds, not sales.",
          lambda d: ~d["InvoiceNo"].str.startswith("C"))
    apply("Remove zero or negative quantity and price", "These cannot be real sales.",
          lambda d: (d["Quantity"] > 0) & (d["UnitPrice"] > 0))
    apply("Remove non-product codes", "Codes like POST and BANK CHARGES are fees, not products.",
          lambda d: d["StockCode"].str.match(r"^\d"))

    df = df.copy()
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    df["Description"] = df["Description"].str.strip()
    df["Revenue"] = (df["Quantity"] * df["UnitPrice"]).round(2)
    df["Month"] = df["InvoiceDate"].dt.strftime("%Y-%m")
    df["Weekday"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour
    return df, pd.DataFrame(steps)


# ---------- Step 2: MapReduce ----------

JOBS = {
    "Revenue by country": ("Country", "Revenue"),
    "Revenue by month": ("Month", "Revenue"),
    "Units sold by product": ("Description", "Quantity"),
}


def split_input(records, n_splits):
    """Input split: divides the records into nearly equal parts."""
    size = len(records) // n_splits + 1
    return [records[i:i + size] for i in range(0, len(records), size)]


def mapper(record):
    """Map: one record in, one (key, value) pair out."""
    key, value = record
    return key, value


def combiner(pairs):
    """Combine: a mini-reducer that adds values inside one split, so less data is shuffled."""
    totals = defaultdict(float)
    for key, value in pairs:
        totals[key] += value
    return list(totals.items())


def shuffle_sort(all_pairs):
    """Shuffle and sort: puts every value with the same key together, sorted by key."""
    groups = defaultdict(list)
    for key, value in all_pairs:
        groups[key].append(value)
    return dict(sorted(groups.items()))


def reducer(key, values):
    """Reduce: adds the values for one key."""
    return key, round(sum(values), 2)


def run_mapreduce(df, job, n_splits=4, use_combiner=True):
    key_col, value_col = JOBS[job]
    records = list(zip(df[key_col].tolist(), df[value_col].tolist()))
    splits = split_input(records, n_splits)

    map_outputs = [[mapper(r) for r in split] for split in splits]
    map_count = sum(len(p) for p in map_outputs)
    if use_combiner:
        map_outputs = [combiner(p) for p in map_outputs]
    shuffled_pairs = [pair for pairs in map_outputs for pair in pairs]

    groups = shuffle_sort(shuffled_pairs)
    results = [reducer(k, v) for k, v in groups.items()]
    return {
        "results": pd.DataFrame(results, columns=[key_col, value_col]),
        "split_sizes": [len(s) for s in splits],
        "map_output": map_count,
        "shuffled": len(shuffled_pairs),
        "reduced": len(results),
    }


# ---------- Step 4: machine learning (K-Means on RFM) ----------

def build_rfm(df):
    """RFM: days since last purchase, number of orders, and total money spent."""
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda d: (snapshot - d.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
    )
    return rfm


def scale_rfm(rfm):
    # Spending is very skewed, so log values stop a few big buyers controlling the clusters
    return StandardScaler().fit_transform(np.log1p(rfm))


def evaluate_k(rfm, k_values=range(2, 9)):
    X = scale_rfm(rfm)
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        rows.append({"k": k, "Inertia": model.inertia_, "Silhouette": silhouette_score(X, model.labels_)})
    return pd.DataFrame(rows)


def segment_customers(rfm, k):
    """Clusters customers, then names each cluster by its rank on a combined RFM score."""
    X = scale_rfm(rfm)
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    out = rfm.copy()
    scores = pd.Series(X[:, 1] + X[:, 2] - X[:, 0]).groupby(labels).mean().sort_values(ascending=False)
    names = {
        2: ["Valuable", "Low value"],
        3: ["Best", "Regular", "At risk"],
        4: ["Best", "Loyal", "Occasional", "At risk"],
        5: ["Best", "Loyal", "Regular", "Occasional", "At risk"],
        6: ["Best", "Loyal", "Regular", "Occasional", "Fading", "At risk"],
    }[k]
    out["Segment"] = pd.Series(labels, index=out.index).map(dict(zip(scores.index, names)))
    return out, names


def segment_summary(segmented, order):
    summary = segmented.groupby("Segment").agg(
        Customers=("Monetary", "size"),
        Median_recency_days=("Recency", "median"),
        Median_orders=("Frequency", "median"),
        Median_spend=("Monetary", "median"),
        Total_revenue=("Monetary", "sum"),
    ).reindex(order)
    summary["Revenue_share"] = summary["Total_revenue"] / summary["Total_revenue"].sum()
    return summary
