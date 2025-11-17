import py_target_id as tid
from rpy2.robjects import r
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
import sys
from importlib.resources import files
from py_target_id import utils
from scipy.stats import pearsonr
import subprocess
import random
from collections import defaultdict

#Run them all
import glob
import anthropic
import re
client = anthropic.Anthropic()

os.makedirs("download", exist_ok=True)
os.makedirs("processed", exist_ok=True)
valid_genes = tid.utils.valid_genes()
surface = tid.utils.surface_genes()

############
# Prompt 1
############

all_genes = surface
batch_size = 10
num_passes = 8

# Track votes: gene -> [pass1_result, pass2_result, ...]
os.makedirs("processed/claude-haiku-4-5-20251001/prompt1", exist_ok=True)

for pass_num in range(num_passes):
	votes = defaultdict(list)
	seed = 1 + pass_num
    random.seed(seed)
    print(f"\n{'='*60}")
    print(f"Pass {pass_num + 1}/{num_passes}")
    print(f"{'='*60}")
    
    shuffled = all_genes.copy()
    random.shuffle(shuffled)
    
    for i in range(0, len(shuffled), batch_size):
        
        batch = shuffled[i:i+batch_size]
        print(i)

        m = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            temperature=0,
            messages=[{"role": "user", "content": f"Return only Yes or No for each gene. Are these Genes present on the Cell Surface in ANY context (including GPI-anchored)? No matter how minimal.\n\n{', '.join(batch)}"}]
        )
        
        for line in m.content[0].text.split('\n'):
            if ':' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    gene = parts[0].strip()
                    response = parts[1].strip().upper()
                    
                    if 'YES' in response:
                        votes[gene].append('YES')
                    elif 'NO' in response:
                        votes[gene].append('NO')
    
    print(f"Completed pass {pass_num + 1}")

	df = pd.DataFrame({
	    'gene_name': votes.keys(),
	    'YES': [1 if votes[gene] == ['YES'] else 0 for gene in votes.keys()],
	    'NO': [1 if votes[gene] == ['NO'] else 0 for gene in votes.keys()]
	})
	df.sort_values("gene_name").to_csv("processed/claude-haiku-4-5-20251001/prompt1/results_seed" + str(seed) + ".csv")    

























total_genes = len(votes)
yes_count = sum(1 for v in votes.values() if v == ['YES'])
no_count = sum(1 for v in votes.values() if v == ['NO'])

m = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=200,
    temperature=0,
    messages=[{"role": "user", "content": f"Return only Yes or No. Is ABCB11 Present on the Cell Surface in ANY context? No matter how minimal."}]
)
m.content[0].text.split('\n')


# Tally votes (majority wins)
print(f"\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}")

surface_confirmed = set()
non_surface_confirmed = set()
uncertain = set()

for gene in all_genes:
    if gene not in votes:
        print(f"⚠️ {gene}: NO DATA")
        continue
    
    yes_votes = votes[gene].count('YES')
    no_votes = votes[gene].count('NO')
    total = yes_votes + no_votes
    
    if total == 0:
        continue
    
    yes_pct = yes_votes / total * 100
    
    if yes_votes >= 3:  # Majority surface (3+ out of 5)
        surface_confirmed.add(gene)
        status = "✅ SURFACE"
    elif no_votes >= 3:  # Majority non-surface (3+ out of 5)
        non_surface_confirmed.add(gene)
        status = "❌ NON-SURFACE"
    else:  # Tie/uncertain (2-3 split)
        uncertain.add(gene)
        status = "❓ UNCERTAIN"
    
    print(f"{gene}: {yes_votes}/{total} YES ({yes_pct:.0f}%) - {status}")

print(f"\n{'='*60}")
print(f"✅ Surface confirmed: {len(surface_confirmed)}")
print(f"❌ Non-surface confirmed: {len(non_surface_confirmed)}")
print(f"❓ Uncertain (2-3 split): {len(uncertain)}")
print(f"{'='*60}")

# Save results
with open('surface_targets_final.txt', 'w') as f:
    f.write('\n'.join(sorted(surface_confirmed)))

with open('non_surface_targets_final.txt', 'w') as f:
    f.write('\n'.join(sorted(non_surface_confirmed)))

with open('uncertain_targets.txt', 'w') as f:
    f.write('\n'.join(sorted(uncertain)))

print("\nResults saved!")
print(f"Surface: surface_targets_final.txt")
print(f"Non-surface: non_surface_targets_final.txt")
print(f"Uncertain: uncertain_targets.txt")