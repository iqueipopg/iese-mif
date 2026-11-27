"""Customer segments and diversification: K-Means and hierarchical clustering.

Answers Q1-Q8 of the case in order. Part 1 segments 2,216 retail customers (Kaggle Customer Personality
Analysis); part 2 clusters 28 ETFs on the correlation of their daily returns. The case supplies the ETF prices as
data/etf-prices.csv; if the file is missing, the script downloads the same series from Yahoo Finance.
Every figure goes to output/figures and every table to output/tables.

    uv run python src/segments.py
"""

from pathlib import Path

import kagglehub
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "output" / "figures"
TAB = ROOT / "output" / "tables"
DATA = ROOT / "data"
for d in (FIG, TAB, DATA):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREY = "#2b5c8a", "#c0392b", "#9a9a9a"

FEATURES = ["Age", "Income", "Total_Spending", "Total_Purchases", "Total_Campaigns_Accepted", "Recency",
            "NumWebVisitsMonth", "Kidhome", "Teenhome"]
MNT = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
CHANNELS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumDealsPurchases"]
CAMPAIGNS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]
TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "XLK", "XLF", "XLE", "XLV", "XLP", "XLU", "XLY", "XLI", "XLB",
           "XLRE", "XLC", "VNQ", "TLT", "IEF", "SHY", "TIP", "LQD", "HYG", "GLD", "SLV", "USO", "DBC", "UUP"]


def profile(df, labels):
    """Mean of each feature by cluster, with the cluster size."""
    out = df.assign(cluster=labels).groupby("cluster")[FEATURES].mean()
    out.insert(0, "n", pd.Series(labels).value_counts().sort_index())
    return out.round(1)


def nested_share(coarse, fine):
    """How many fine clusters sit entirely inside one coarse cluster."""
    t = pd.crosstab(fine, coarse)
    return int(((t > 0).sum(axis=1) == 1).sum()), len(t)


# ---------------------------------------------------------------- Part 1: customers
path = Path(kagglehub.dataset_download("imakash3011/customer-personality-analysis"))
raw = pd.read_csv(path / "marketing_campaign.csv", sep="\t")
cust = raw.dropna(subset=["Income"]).copy()
cust["Age"] = 2022 - cust["Year_Birth"]
cust["Total_Spending"] = cust[MNT].sum(axis=1)
cust["Total_Purchases"] = cust[CHANNELS].sum(axis=1)
cust["Total_Campaigns_Accepted"] = cust[CAMPAIGNS].sum(axis=1)
cust["Has_Children"] = ((cust["Kidhome"] + cust["Teenhome"]) > 0).astype(int)
X = cust[FEATURES]
print(f"Q1 customers: {len(raw)} rows, {len(cust)} after dropping missing income")

# Q1: ranges
ranges = X.describe().T[["min", "max", "std"]]
ranges["range"] = ranges["max"] - ranges["min"]
ranges.round(1).to_csv(TAB / "q1_feature_ranges.csv")
print(ranges.round(1))
Z = StandardScaler().fit_transform(X)

# Q2: K-Means, k = 4
km4 = KMeans(n_clusters=4, n_init=10, random_state=SEED).fit(Z)
prof4 = profile(X, km4.labels_)
prof4.to_csv(TAB / "q2_kmeans4_profiles.csv")
print("Q2 K-Means k=4 profiles\n", prof4)

# Q3: elbow and silhouette
rows = []
for k in range(2, 11):
    m = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
    rows.append({"k": k, "inertia": m.inertia_, "silhouette": silhouette_score(Z, m.labels_)})
elbow = pd.DataFrame(rows)
elbow.round(3).to_csv(TAB / "q3_elbow_silhouette.csv", index=False)
print("Q3\n", elbow.round(3))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
ax[0].plot(elbow.k, elbow.inertia, "o-", color=BLUE)
ax[0].set(title="Inertia (elbow)", xlabel="k")
ax[1].plot(elbow.k, elbow.silhouette, "o-", color=RED)
ax[1].set(title="Silhouette", xlabel="k")
fig.tight_layout()
fig.savefig(FIG / "q3_elbow_silhouette.png")
plt.close(fig)

# Q4: four linkages, cut at four clusters
links = {m: linkage(Z, method=m) for m in ["single", "complete", "average", "ward"]}
sizes, tiny = {}, []
for m, L in links.items():
    lab = fcluster(L, t=4, criterion="maxclust")
    counts = pd.Series(lab).value_counts()
    sizes[m] = sorted(counts.values, reverse=True)
    small = counts[counts < 20].index
    if len(small):
        tiny.append(cust.loc[np.isin(lab, small), ["ID", "Year_Birth", "Age", "Income", "Total_Spending"]]
                    .assign(linkage=m))
pd.DataFrame(sizes).T.to_csv(TAB / "q4_linkage_sizes.csv")
tiny = pd.concat(tiny).drop_duplicates(subset=["ID"]) if tiny else pd.DataFrame()
tiny.to_csv(TAB / "q4_tiny_cluster_members.csv", index=False)
print("Q4 sizes", sizes)
print(tiny.sort_values("Age", ascending=False).head(15))

# Q5: Ward dendrogram and jumps
ward = links["ward"]
heights = ward[::-1, 2][:12]
pd.DataFrame({"clusters_after_merge": range(1, 13), "merge_height": heights.round(1)}).to_csv(
    TAB / "q5_ward_merge_heights.csv", index=False)
print("Q5 last Ward merge heights", heights.round(1))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [2, 1]})
dendrogram(ward, ax=ax[0], truncate_mode="lastp", p=30, color_threshold=heights[3], no_labels=True)
ax[0].axhline((heights[3] + heights[4]) / 2, color=RED, ls="--", lw=0.8)
ax[0].set(title="Ward dendrogram (last 30 merges)", ylabel="merge distance")
ax[1].bar(range(1, 11), heights[:10], color=BLUE)
ax[1].set(title="Last merge heights", xlabel="clusters after merge")
fig.tight_layout()
fig.savefig(FIG / "q5_ward_dendrogram.png")
plt.close(fig)
ward4 = fcluster(ward, t=4, criterion="maxclust")
profile(X, ward4).to_csv(TAB / "q5_ward4_profiles.csv")
print("Q5 Ward k=4 profiles\n", profile(X, ward4))

# Q6: Ward vs K-Means
ari = adjusted_rand_score(km4.labels_, ward4)
pd.crosstab(pd.Series(km4.labels_, name="kmeans"), pd.Series(ward4, name="ward")).to_csv(TAB / "q6_crosstab.csv")
ward6 = fcluster(ward, t=6, criterion="maxclust")
km6 = KMeans(n_clusters=6, n_init=10, random_state=SEED).fit(Z).labels_
q6 = pd.DataFrame([
    {"method": "K-Means", "silhouette_k4": silhouette_score(Z, km4.labels_),
     "nested_6_in_4": "%d of %d" % nested_share(km4.labels_, km6)},
    {"method": "Ward", "silhouette_k4": silhouette_score(Z, ward4),
     "nested_6_in_4": "%d of %d" % nested_share(ward4, ward6)},
])
q6["ARI_k4"] = ari
q6.round(3).to_csv(TAB / "q6_ward_vs_kmeans.csv", index=False)
print("Q6\n", q6.round(3))
n = 1_000_000
print(f"Q6 pairwise distances for a million customers: {n * (n - 1) / 2:.2e}, "
      f"{n * (n - 1) / 2 * 8 / 1e12:.1f} TB in float64")

# ---------------------------------------------------------------- Part 2: ETFs
csv = DATA / "etf-prices.csv"
if not csv.exists():
    import yfinance as yf

    px = yf.download(TICKERS, start="2019-01-01", end="2026-10-01", auto_adjust=True, progress=False)["Close"]
    px[TICKERS].to_csv(csv)
prices = pd.read_csv(csv, index_col=0, parse_dates=True)[TICKERS]
rets = prices.pct_change().dropna()
print(f"Q7 ETFs: {rets.shape[1]} tickers, {len(rets)} days, {rets.index[0].date()} to {rets.index[-1].date()}")


def etf_tree(r):
    corr = r.corr()
    dist = np.sqrt(0.5 * (1 - corr))
    L = linkage(squareform(dist.values, checks=False), method="average")
    return corr, L


def plot_tree_and_matrix(corr, L, title, fname):
    order = corr.index[leaves_list(L)]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1, 1.1]})
    dendrogram(L, labels=corr.index.tolist(), ax=ax[0], color_threshold=0.5, leaf_font_size=6, orientation="left")
    ax[0].axvline(0.5, color=RED, ls="--", lw=0.8)
    ax[0].set(title=f"{title}: average linkage", xlabel="distance √(½(1−ρ))")
    im = ax[1].imshow(corr.loc[order, order], cmap="RdBu_r", vmin=-1, vmax=1)
    ax[1].set_xticks(range(len(order)), order, rotation=90, fontsize=5)
    ax[1].set_yticks(range(len(order)), order, fontsize=5)
    ax[1].set_title("Correlation, reordered")
    fig.colorbar(im, ax=ax[1], shrink=0.7)
    fig.tight_layout()
    fig.savefig(FIG / fname)
    plt.close(fig)


corr, L = etf_tree(rets)
groups = fcluster(L, t=0.5, criterion="distance")
bets = pd.Series(groups, index=corr.index, name="bet").sort_values()
bets.to_csv(TAB / "q7_etf_bets.csv")
print(f"Q7 independent bets at d=0.5: {bets.nunique()}")
for g, members in bets.groupby(bets):
    print("  ", g, list(members.index))
print(f"Q7 rho SPY-HYG {corr.loc['SPY', 'HYG']:.2f}, LQD-IEF {corr.loc['LQD', 'IEF']:.2f}, "
      f"LQD-SPY {corr.loc['LQD', 'SPY']:.2f}, HYG-IEF {corr.loc['HYG', 'IEF']:.2f}")
plot_tree_and_matrix(corr, L, "2019-2026", "q7_etf_dendrogram_corr.png")

# Q8: two windows
rows = []
for name, r in {"2019-2021": rets.loc[:"2021"], "2022-2026": rets.loc["2022":]}.items():
    c, Lw = etf_tree(r)
    g = pd.Series(fcluster(Lw, t=0.5, criterion="distance"), index=c.index)
    rows.append({"window": name, "days": len(r), "bets": g.nunique(),
                 "rho_SPY_TLT": c.loc["SPY", "TLT"], "rho_SPY_IEF": c.loc["SPY", "IEF"],
                 "TLT_cluster": ", ".join(g[g == g["TLT"]].index)})
    plot_tree_and_matrix(c, Lw, name, f"q8_etf_{name}.png")
q8 = pd.DataFrame(rows)
q8.round(2).to_csv(TAB / "q8_windows.csv", index=False)
print("Q8\n", q8.round(2).to_string())
