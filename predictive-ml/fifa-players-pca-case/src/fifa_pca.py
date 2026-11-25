"""FIFA 22 players: how many real dimensions hide in 35 skill ratings? Principal Component Analysis.

Answers Q1-Q12 of the case in order. Unsupervised: there is no target. Every figure goes to output/figures and
every table to output/tables.

    uv run python src/fifa_pca.py
"""

from pathlib import Path

import kagglehub
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "output" / "figures"
TAB = ROOT / "output" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREY = "#2b5c8a", "#c0392b", "#9a9a9a"

MAIN = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
PREFIXES = ("attacking_", "skill_", "movement_", "power_", "mentality_", "defending_")

# ---------------------------------------------------------------- Q1-Q2: load, pick skills, drop goalkeepers
path = Path(kagglehub.dataset_download("stefanoleone992/fifa-22-complete-player-dataset"))
df = pd.read_csv(path / "players_22.csv", low_memory=False)
print(f"Q1. {len(df)} players, {df.shape[1]} columns")
SKILLS = MAIN + [c for c in df.columns if c.startswith(PREFIXES) and c != "skill_moves"]  # skill_moves is a 1-5 star rating, not a skill
print(f"Q2. {len(SKILLS)} skill attributes")
df["position"] = df["player_positions"].str.split(",").str[0].str.strip()
out = df[~df["position"].str.startswith("GK")].reset_index(drop=True)
print(f"   Goalkeepers removed: {len(df) - len(out)}; outfield players: {len(out)}; "
      f"missing skill values: {int(out[SKILLS].isna().sum().sum())}")

# ---------------------------------------------------------------- Q3: raw scales
desc = out[SKILLS].agg(["mean", "std"]).T
desc.to_csv(TAB / "q3_skill_mean_std.csv")
print("Q3.", ", ".join(f"{c}: mean {desc.loc[c, 'mean']:.1f}, sd {desc.loc[c, 'std']:.1f}"
                       for c in ["pace", "shooting", "defending"]))
print(f"   Std ranges from {desc['std'].min():.1f} ({desc['std'].idxmin()}) to {desc['std'].max():.1f} "
      f"({desc['std'].idxmax()})")

# ---------------------------------------------------------------- Q4: standardise
Z = StandardScaler().fit_transform(out[SKILLS])
print(f"Q4. After scaling: max |mean| {np.abs(Z.mean(0)).max():.1e}, std in [{Z.std(0).min():.3f}, {Z.std(0).max():.3f}]")

# ---------------------------------------------------------------- Q5-Q6: PCA and scree plot
pca = PCA().fit(Z)
evr = pca.explained_variance_ratio_
cum = evr.cumsum()
pd.DataFrame({"component": np.arange(1, len(evr) + 1), "explained": evr, "cumulative": cum}).to_csv(
    TAB / "q5_q6_explained_variance.csv", index=False)
print(f"Q5. {pca.n_components_} components; PC1 {evr[0]:.1%}, PC2 {evr[1]:.1%}, PC3 {evr[2]:.1%}")
need = {t: int(np.argmax(cum >= t) + 1) for t in (0.7, 0.8, 0.9)}
print("Q6. Components needed:", ", ".join(f"{int(t * 100)}% -> {n}" for t, n in need.items()))

fig, ax = plt.subplots(figsize=(4.4, 2.7))
k = np.arange(1, 16)
ax.bar(k, evr[:15], color=BLUE, label="individual")
ax.plot(k, cum[:15], "o-", ms=3, color=RED, label="cumulative")
for t in (0.7, 0.8, 0.9):
    ax.axhline(t, color=GREY, ls=":", lw=0.8)
    ax.text(15.4, t, f"{int(t * 100)}%", va="center", fontsize=7, color=GREY)
ax.set_xlabel("Principal component")
ax.set_ylabel("Share of variance")
ax.set_title(f"Scree plot: {need[0.8]} components hold 80% of the variance", fontsize=8)
ax.legend(frameon=False, fontsize=7, loc="center right")
fig.tight_layout()
fig.savefig(FIG / "q6_scree.png")
plt.close(fig)

# ---------------------------------------------------------------- Q7-Q8: loadings
# Sign convention: PC1 positive = higher overall skill, so that the reading is stable across runs.
comps = pca.components_.copy()
if comps[0] @ np.ones(len(SKILLS)) < 0:
    comps[0] *= -1
if comps[1][SKILLS.index("defending")] < 0:  # PC2 positive = defensive
    comps[1] *= -1
if comps[2][SKILLS.index("power_strength")] > 0:  # PC3 positive = light and quick
    comps[2] *= -1
load = pd.DataFrame(comps[:4].T, index=SKILLS, columns=["PC1", "PC2", "PC3", "PC4"])
load.to_csv(TAB / "q7_q8_loadings.csv")
for pc in ["PC1", "PC2", "PC3"]:
    s = load[pc].sort_values()
    print(f"Q7/8. {pc} top +:", ", ".join(f"{i} {v:.2f}" for i, v in s[::-1].head(5).items()))
    print(f"       {pc} top -:", ", ".join(f"{i} {v:.2f}" for i, v in s.head(5).items()))

fig, axes = plt.subplots(1, 3, figsize=(7.8, 4.6), sharey=True)
order = load["PC1"].sort_values().index
for ax, pc in zip(axes, ["PC1", "PC2", "PC3"]):
    v = load.loc[order, pc]
    ax.barh(range(len(v)), v, color=[BLUE if x > 0 else RED for x in v])
    ax.axvline(0, color="black", lw=0.5)
    ax.set_title(f"{pc} ({evr[int(pc[-1]) - 1]:.0%})", fontsize=8)
axes[0].set_yticks(range(len(order)), [c.split("_", 1)[-1] for c in order], fontsize=6)
fig.tight_layout()
fig.savefig(FIG / "q7_loadings.png")
plt.close(fig)

# ---------------------------------------------------------------- Q9-Q10: players in PC space
scores = Z @ comps[:10].T
pcs = pd.DataFrame(scores, columns=[f"PC{i}" for i in range(1, 11)])
pcs["name"], pcs["position"] = out["short_name"], out["position"]
POS = ["ST", "RW", "LW", "CAM", "CM", "CDM", "CB", "LB", "RB"]
centroids = pcs[pcs.position.isin(POS)].groupby("position")[["PC1", "PC2", "PC3"]].mean().loc[POS]
centroids.to_csv(TAB / "q9_position_centroids.csv")
print("Q9. Position centroids (PC1, PC2):", ", ".join(f"{p} ({r.PC1:.1f}, {r.PC2:.1f})" for p, r in centroids.iterrows()))

FAMOUS = ["L. Messi", "Cristiano Ronaldo", "R. Lewandowski", "K. Mbappé", "Neymar Jr", "K. De Bruyne",
          "V. van Dijk", "N. Kanté", "Sergio Ramos", "M. Salah", "E. Haaland"]
famous = pcs[pcs.name.isin(FAMOUS)].drop_duplicates("name")
famous[["name", "position", "PC1", "PC2", "PC3"]].to_csv(TAB / "q10_famous_players.csv", index=False)
print("Q10.", ", ".join(f"{r['name']} ({r.PC1:.1f}, {r.PC2:.1f})" for _, r in famous.iterrows()))

groups = {"Forwards (ST, CF)": (["ST", "CF"], RED), "Wingers (RW, LW, RM, LM)": (["RW", "LW", "RM", "LM"], "#e08e0b"),
          "Midfield (CAM, CM, CDM)": (["CAM", "CM", "CDM"], "#3a7d44"),
          "Full-backs (LB, RB, LWB, RWB)": (["LB", "RB", "LWB", "RWB"], "#8e44ad"), "Centre-backs (CB)": (["CB"], BLUE)}
fig, ax = plt.subplots(figsize=(6.0, 4.2))
for lab, (ps, c) in groups.items():
    s = pcs[pcs.position.isin(ps)]
    ax.scatter(s.PC1, s.PC2, s=2, alpha=0.25, color=c, label=lab, rasterized=True)
# The top attackers sit on top of each other: spread their labels out to the right, ordered by PC2.
crowd = famous[famous.PC1 > 9].sort_values("PC2", ascending=False)
label_y = dict(zip(crowd["name"], np.linspace(crowd.PC2.max() + 0.6, crowd.PC2.min() - 2.2, len(crowd))))
for _, r in famous.iterrows():
    ax.scatter(r.PC1, r.PC2, s=14, color="black")
    if r["name"] in label_y:
        ax.annotate(r["name"], (r.PC1, r.PC2), xytext=(15.0, label_y[r["name"]]), textcoords="data", fontsize=6,
                    va="center", arrowprops={"arrowstyle": "-", "lw": 0.4, "color": GREY})
    else:
        ax.annotate(r["name"], (r.PC1, r.PC2), xytext=(-44 if r["name"] == "V. van Dijk" else 3, 3),
                    textcoords="offset points", fontsize=6)
ax.set_xlim(right=19)
ax.set_xlabel(f"PC1: overall technical and attacking quality ({evr[0]:.0%})")
ax.set_ylabel(f"PC2: defender (+) vs finisher (-) ({evr[1]:.0%})")
ax.legend(frameon=False, fontsize=6, markerscale=5, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "q9_q10_players_pc1_pc2.png")
plt.close(fig)

# ---------------------------------------------------------------- Q11: predict position, 35 features vs few PCs
lab = out["position"].map({p: g for g, (ps, _) in groups.items() for p in ps})
keep = lab.notna().to_numpy()
Xs, yl = out.loc[keep, SKILLS], lab[keep]
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
q11 = []
for n in [2, 3, 4, 10, None]:
    steps = [StandardScaler()] + ([PCA(n_components=n)] if n else []) + [LogisticRegression(max_iter=2000)]
    acc = cross_val_score(make_pipeline(*steps), Xs, yl, cv=cv).mean()
    q11.append(("all 35 skills" if n is None else f"first {n} PCs", acc))
q11 = pd.DataFrame(q11, columns=["features", "cv_accuracy"])
q11.to_csv(TAB / "q11_position_from_pcs.csv", index=False)
print("Q11. Position group (5 classes), logistic regression, 5-fold CV:",
      ", ".join(f"{f} {a:.1%}" for f, a in q11.itertuples(index=False)))

# ---------------------------------------------------------------- Q12: who is most like Messi?
messi = pcs[pcs.name == "L. Messi"].iloc[0]
sim = {}
for n in (3, 4):
    cols = [f"PC{i}" for i in range(1, n + 1)]
    d = np.sqrt(((pcs[cols] - messi[cols].to_numpy(dtype=float)) ** 2).sum(1))
    sim[n] = pcs.assign(distance=d).query("name != 'L. Messi'").nsmallest(8, "distance")[["name", "position", "distance"]]
    sim[n].to_csv(TAB / f"q12_closest_to_messi_{n}pcs.csv", index=False)
    print(f"Q12. Closest to Messi with {n} PCs:", ", ".join(f"{r['name']} {r.distance:.2f}" for _, r in sim[n].iterrows()))
