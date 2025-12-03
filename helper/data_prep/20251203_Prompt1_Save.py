#Let's Filter
files = glob.glob("../../Surface_Testing/processed/claude-haiku-4-5-20251001/prompt1/*.csv")
dfs = [pd.read_csv(f, index_col=0) for f in files]
combined = pd.concat(dfs, axis=0)
tallied = combined.groupby("gene_name").sum()

# Score based on YES proportion
p = tallied["YES"] / np.sum(tallied, axis=1)
score = pd.Series(0, index=tallied.index, dtype=int)
score[p == 0] = 0
score[(p > 0) & (p < 0.25)] = 1
score[(p >= 0.25) & (p < 0.5)] = 2
score[(p >= 0.5) & (p < 0.75)] = 3
score[(p >= 0.75) & (p < 1)] = 4
score[p == 1] = 5

tallied["claude_summary"] = score.astype(int)
tallied.to_csv("Surface-Prompt1-claude-haiku-4-5-20251001.csv")








