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

os.makedirs("download", exist_ok=True)
os.makedirs("processed", exist_ok=True)
valid_genes = tid.utils.valid_genes()
surface = tid.utils.surface_genes()

#############
# SCLC
#############
#https://cellxgene.cziscience.com/collections/62e8f058-9c37-48bc-9200-e767f318a8ec
subprocess.run(
    "wget https://datasets.cellxgene.cziscience.com/5b7cbce2-ed53-47a4-99f8-10af741814e6.h5ad -O download/msk_htan_lung_sclc.h5ad",
    shell=True
)
adata = sc.read_h5ad('download/msk_htan_lung_sclc.h5ad')
mask = adata.obs["cell_type_fine"].str.contains("^SCLC")
adata = adata[mask, :].copy()

#CP10k
cp10k = tid.utils.cp10k(adata.raw.X)

#Summarize
mask = adata.obs["cell_type_fine"].str.contains("^SCLC-A")
sclc_a = ad.AnnData(
    X=tid.utils.summarize_matrix(cp10k[mask,:], groups = adata[mask,:].obs["donor_id"].values, metric="mean", axis=0)
)
sclc_a = sclc_a[sclc_a.obs_names != "PleuralEffusion", :]
sclc_a.var_names = adata.var["feature_name"].values.astype(str)
sclc_a.var_names_make_unique()
sclc_a = sclc_a[:, sclc_a.var_names.isin(valid_genes)].copy()
sclc_a = tid.run.compute_positivity_matrix(sclc_a, fallback_threshold = 0.5)
sclc_a.write_h5ad("processed/MSK_HTAN_SCLC_A.h5ad")

#Summarize
mask = adata.obs["cell_type_fine"].str.contains("^SCLC-N")
sclc_n = ad.AnnData(
    X=tid.utils.summarize_matrix(cp10k[mask,:], groups = adata[mask,:].obs["donor_id"].values, metric="mean", axis=0)
)
sclc_n = sclc_n[sclc_n.obs_names != "PleuralEffusion", :]
sclc_n.var_names = adata.var["feature_name"].values.astype(str)
sclc_n.var_names_make_unique()
sclc_n = sclc_n[:, sclc_n.var_names.isin(valid_genes)].copy()
sclc_n = tid.run.compute_positivity_matrix(sclc_n, fallback_threshold = 0.5)
sclc_n.write_h5ad("processed/MSK_HTAN_SCLC_N.h5ad")

#Summarize
mask = adata.obs["cell_type_fine"].str.contains("^SCLC-P")
sclc_p = ad.AnnData(
    X=tid.utils.summarize_matrix(cp10k[mask,:], groups = adata[mask,:].obs["donor_id"].values, metric="mean", axis=0)
)
sclc_p = sclc_p[sclc_p.obs_names != "PleuralEffusion", :]
sclc_p.var_names = adata.var["feature_name"].values.astype(str)
sclc_p.var_names_make_unique()
sclc_p = sclc_p[:, sclc_p.var_names.isin(valid_genes)].copy()
sclc_p = tid.run.compute_positivity_matrix(sclc_p, fallback_threshold = 0.5)
sclc_p.write_h5ad("processed/MSK_HTAN_SCLC_P.h5ad")

#############
# Breast Cancer
#############
#https://cellxgene.cziscience.com/collections/9432ae97-4803-4b9f-8f64-2b41e42ad3cb
subprocess.run(
    "wget https://datasets.cellxgene.cziscience.com/303bc6a5-0811-4ee6-97b8-399f883ce0a2.h5ad -O download/human_breast_cancer_atlas.h5ad",
    shell=True
)
adata = sc.read_h5ad('download/human_breast_cancer_atlas.h5ad', backed='r')

disease_map = {
    'HER2 positive breast carcinoma': 'BRCA-HER2pos',
    'breast apocrine carcinoma': 'BRCA-Apocrine',
    'breast cancer': 'BRCA-BC',
    'breast carcinoma': 'BRCA-Carcinoma',
    'breast mucinous carcinoma': 'BRCA-Mucinous',
    'estrogen-receptor positive breast cancer': 'BRCA-ERpos',
    'invasive ductal breast carcinoma': 'BRCA-IDC',
    'invasive lobular breast carcinoma': 'BRCA-ILC',
    'invasive tubular breast carcinoma || invasive lobular breast carcinoma': 'BRCA-Tubular-ILC',
    'metaplastic breast carcinoma': 'BRCA-Metaplastic',
    'triple-negative breast carcinoma': 'BRCA-TNBC'
}

for subtype in disease_map.keys():
    print(subtype)
    mask = (adata.obs["cell_type"].str.contains("^malignant")) & (adata.obs["disease"]==subtype)
    sub_adata = adata[mask, :].to_memory()
    malig_mean_adata = ad.AnnData(
        X=tid.utils.summarize_matrix(tid.utils.cp10k(sub_adata.raw.X), groups = sub_adata.obs["donor_id"].values, metric="mean", axis=0)
    )
    malig_mean_adata.var_names = sub_adata.var["feature_name"].values.astype(str)
    malig_mean_adata.var_names_make_unique()
    malig_mean_adata = malig_mean_adata[:, malig_mean_adata.var_names.isin(valid_genes)].copy()
    malig_mean_adata = tid.run.compute_positivity_matrix(malig_mean_adata, fallback_threshold = 0.5)
    malig_mean_adata.write_h5ad("processed/" + disease_map[subtype] + ".h5ad")

#############
# Breast Cancer
#############
#https://cellxgene.cziscience.com/collections/9432ae97-4803-4b9f-8f64-2b41e42ad3cb
subprocess.run(
    "wget https://datasets.cellxgene.cziscience.com/303bc6a5-0811-4ee6-97b8-399f883ce0a2.h5ad -O download/human_breast_cancer_atlas.h5ad",
    shell=True
)
adata = sc.read_h5ad('download/human_breast_cancer_atlas.h5ad', backed='r')

disease_map = {
    'HER2 positive breast carcinoma': 'BRCA-HER2pos',
    'breast apocrine carcinoma': 'BRCA-Apocrine',
    'breast cancer': 'BRCA-BC',
    'breast carcinoma': 'BRCA-Carcinoma',
    'breast mucinous carcinoma': 'BRCA-Mucinous',
    'estrogen-receptor positive breast cancer': 'BRCA-ERpos',
    'invasive ductal breast carcinoma': 'BRCA-IDC',
    'invasive lobular breast carcinoma': 'BRCA-ILC',
    'invasive tubular breast carcinoma || invasive lobular breast carcinoma': 'BRCA-Tubular-ILC',
    'metaplastic breast carcinoma': 'BRCA-Metaplastic',
    'triple-negative breast carcinoma': 'BRCA-TNBC'
}

for subtype in disease_map.keys():
    print(subtype)
    mask = (adata.obs["cell_type"].str.contains("^malignant")) & (adata.obs["disease"]==subtype)
    sub_adata = adata[mask, :].to_memory()
    malig_mean_adata = ad.AnnData(
        X=tid.utils.summarize_matrix(tid.utils.cp10k(sub_adata.raw.X), groups = sub_adata.obs["donor_id"].values, metric="mean", axis=0)
    )
    malig_mean_adata.var_names = sub_adata.var["feature_name"].values.astype(str)
    malig_mean_adata.var_names_make_unique()
    malig_mean_adata = malig_mean_adata[:, malig_mean_adata.var_names.isin(valid_genes)].copy()
    malig_mean_adata = tid.run.compute_positivity_matrix(malig_mean_adata, fallback_threshold = 0.5)
    malig_mean_adata.write_h5ad("processed/" + disease_map[subtype] + ".h5ad")

#############
# Breast Cancer
#############
#https://cellxgene.cziscience.com/collections/9432ae97-4803-4b9f-8f64-2b41e42ad3cb
subprocess.run(
    "wget https://datasets.cellxgene.cziscience.com/09595871-5cde-4351-88f9-b3c37b3ed466.h5ad -O download/icb_datasets.h5ad",
    shell=True
)
adata = sc.read_h5ad('download/icb_datasets.h5ad', backed='r')
obs = adata.obs.copy()
obs = obs[obs["author_cell_type_update"]=="Malignant"]
obs = obs[obs.Primary_or_met=="Primary"]
obs["Cancer_type_update"].value_counts()

sub_adata = adata[obs[obs["Cancer_type_update"]=="HCC"].index, :].to_memory()
malig_mean_adata = ad.AnnData(
    X=tid.utils.summarize_matrix(tid.utils.cp10k(sub_adata.raw.X), groups = sub_adata.obs["donor_id"].values, metric="mean", axis=0)
)
malig_mean_adata.var = sub_adata.var
malig_mean_adata = malig_mean_adata[:, malig_mean_adata.var["feature_name"].isin(valid_genes)].copy()
malig_mean_adata.var_names = malig_mean_adata.var["feature_name"].values.astype(str)
malig_mean_adata.var_names_make_unique()
malig_mean_adata = malig_mean_adata[:, malig_mean_adata.var_names.isin(valid_genes)].copy()
malig_mean_adata = tid.run.compute_positivity_matrix(malig_mean_adata, fallback_threshold = 0.1)
malig_mean_adata.write_h5ad("processed/HCC_ICB_Dataset.h5ad")

#############
# Breast Cancer
#############
#https://cellxgene.cziscience.com/collections/9432ae97-4803-4b9f-8f64-2b41e42ad3cb
subprocess.run(
    "wget https://datasets.cellxgene.cziscience.com/dbb5ad81-1713-4aee-8257-396fbabe7c6e.h5ad -O download/hca_lung_atlas.h5ad",
    shell=True
)
adata = sc.read_h5ad('download/hca_lung_atlas.h5ad', backed='r')
obs = adata.obs.copy()

disease_map = {
    'lung adenocarcinoma': 'Lung_LUAD',           # ⭐⭐⭐ TARGET: Malignant epithelial cells
    'lung large cell carcinoma': 'Lung_LCC',      # ⭐⭐⭐ TARGET: Malignant large cells
    'squamous cell lung carcinoma': 'Lung_SCC',   # ⭐⭐⭐ TARGET: Malignant squamous cells
    'pleomorphic carcinoma': 'Lung_PleoCa',       # ⭐⭐⭐ TARGET: Mixed malignant cells
}

for subtype in disease_map.keys():
    print(subtype)
    mask = (adata.obs["cell_type"].str.contains("^unknown")) & (adata.obs["disease"]==subtype)
    sub_adata = adata[mask, :].to_memory()
    malig_mean_adata = ad.AnnData(
        X=tid.utils.summarize_matrix(tid.utils.cp10k(sub_adata.raw.X), groups = sub_adata.obs["donor_id"].values, metric="mean", axis=0)
    )
    malig_mean_adata.var_names = sub_adata.var["feature_name"].values.astype(str)
    malig_mean_adata.var_names_make_unique()
    malig_mean_adata = malig_mean_adata[:, malig_mean_adata.var_names.isin(valid_genes)].copy()
    malig_mean_adata = tid.run.compute_positivity_matrix(malig_mean_adata, fallback_threshold = 0.5)
    malig_mean_adata.write_h5ad("processed/" + disease_map[subtype] + ".h5ad")



obs = obs[obs["disease"]=="Malignant"]
obs = obs[obs.Primary_or_met=="Primary"]
obs["Cancer_type_update"].value_counts()

sub_adata = adata[obs[obs["Cancer_type_update"]=="HCC"].index, :].to_memory()
malig_mean_adata = ad.AnnData(
    X=tid.utils.summarize_matrix(tid.utils.cp10k(sub_adata.raw.X), groups = sub_adata.obs["donor_id"].values, metric="mean", axis=0)
)
malig_mean_adata.var = sub_adata.var
malig_mean_adata = malig_mean_adata[:, malig_mean_adata.var["feature_name"].isin(valid_genes)].copy()
malig_mean_adata.var_names = malig_mean_adata.var["feature_name"].values.astype(str)
malig_mean_adata.var_names_make_unique()
malig_mean_adata = malig_mean_adata[:, malig_mean_adata.var_names.isin(valid_genes)].copy()
malig_mean_adata = tid.run.compute_positivity_matrix(malig_mean_adata, fallback_threshold = 0.1)
malig_mean_adata.write_h5ad("processed/HCC_ICB_Dataset.h5ad")

#Run them all
import glob
import anthropic
import re
client = anthropic.Anthropic()

# Get all .h5ad files
files = glob.glob('processed/*.h5ad')

sc_df_off = utils.get_ref_sc_off_target()
sc_ref_med_adata = utils.get_ref_lv4_sc_med_adata()
sc_ref_med_adata = utils.add_ref_weights(sc_ref_med_adata, sc_df_off, "Off_Target.V0")

ff_df_off = utils.get_ref_ffpe_off_target()
ff_ref_med_adata = utils.get_ref_lv4_ffpe_med_adata()
ff_ref_med_adata = utils.add_ref_weights(ff_ref_med_adata, ff_df_off, "Off_Target.V0")

risk = tid.utils.get_single_risk_scores()

for f in files:
    
    m = sc.read_h5ad(f)
    
    df_sc = tid.run.target_id_v1(malig_adata=m, ref_adata=sc_ref_med_adata)
    df_ff = tid.run.target_id_v1(malig_adata=m, ref_adata=ff_ref_med_adata)
    
    df_sc = df_sc.loc[df_sc["gene_name"].isin(surface),:]
    df_ff = df_ff.loc[df_ff["gene_name"].isin(surface),:]

    df_sc = pd.merge(df_sc, risk, how = "left")
    df_ff = pd.merge(df_ff, risk, how = "left")

    df_sc = tid.run.target_quality_v2_01(df_sc)
    df_ff = tid.run.target_quality_v2_01(df_ff)

    sc_top = df_sc[df_sc["Hazard_GTEX_v1"] < 25].head(100)["gene_name"].to_list()
    ff_top = df_ff[df_ff["Hazard_GTEX_v1"] < 25].head(100)["gene_name"].to_list()

    genes = np.unique(sc_top + ff_top)
   
    # Pass 1: Find non-surface
    m1 = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=300, temperature=0,
        messages=[{"role": "user", "content": f"List ONLY non-surface genes (intracellular/secreted/ER/mito). One per line:\n\n{', '.join(genes)}"}])
    non_surf = {g.strip() for g in m1.content[0].text.strip().split('\n') if g.strip() and not g.startswith('#')}

    # Pass 2: Verify YES/NO
    m2 = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=400, temperature=0,
        messages=[{"role": "user", "content": f"YES if NOT surface, NO if IS surface. Format: GENE: YES/NO\n\n{', '.join(sorted(non_surf))}"}])
    confirmed = {line.split(':')[0].strip().replace('**','') for line in m2.content[0].text.split('\n') if 'YES' in line.upper() and ':' in line}

    # Pass 3: Challenge borderline
    m3 = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=300, temperature=0,
        messages=[{"role": "user", "content": f"KEEP (definitely non-surface) or REMOVE (might be surface):\n\n{', '.join(sorted(confirmed))}"}])
    remove = {line.split(':')[0].strip().replace('**','') for line in m3.content[0].text.split('\n') if 'REMOVE' in line.upper()}

    final_surface = sorted([g for g in genes if g not in (confirmed - remove)])
    print('\n'.join(final_surface))

len(surface)

batch_size = 20  # Process in chunks

non_surf_all = set()

for i in range(0, len(genes), batch_size):
    batch = genes[i:i+batch_size]
    m1 = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=150, temperature=0,
        messages=[{"role": "user", "content": f"Non-surface genes only:\n\n{', '.join(batch)}"}])
    non_surf = {g.strip() for g in m1.content[0].text.strip().split('\n') if g.strip() and len(g.strip()) > 2}
    non_surf_all.update(non_surf)

# Pass 2: Verify in smaller batches
confirmed_all = set()
for i in range(0, len(non_surf_all), batch_size):
    batch = sorted(list(non_surf_all))[i:i+batch_size]
    m2 = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=150, temperature=0,
        messages=[{"role": "user", "content": f"YES if not surface:\n\n{', '.join(batch)}"}])
    confirmed = set(re.findall(r'(\w+):\s*YES', m2.content[0].text))
    confirmed_all.update(confirmed)

final = sorted(set(genes) - confirmed_all)
print('\n'.join(final))


In [597]: m1.content[0].text
Out[597]: 'ABCA13, ABCA4, ABCB11, ABCB4, ABCC11, ABCG5, ALG10, B4GALNT2, BIK, BSND, C4BPB, CATSPER1, CATSPERG, CCDC144A, CCDC168'



import anthropic

client = anthropic.Anthropic()

# First pass: identify non-surface
message1 = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=200,
    temperature=0,
    messages=[
        {"role": "user", "content": f"List only genes that are NOT cell surface (intracellular/secreted). One per line, gene name only:\n\n{', '.join(top)}"}
    ]
)

non_surface = set(message1.content[0].text.strip().split('\n'))
non_surface = {g.strip() for g in non_surface if g.strip()}

print("First pass non-surface genes:")
print('\n'.join(sorted(non_surface)))

# Second pass: verify each one
print("\n" + "="*50)
print("VERIFICATION PASS")
print("="*50)

verification_list = ', '.join(sorted(non_surface))
message2 = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=300,
    temperature=0,
    messages=[
        {"role": "user", "content": f"Confirm these are NOT cell surface. Format: GENE: YES/NO (reason)\n\n{verification_list}"}
    ]
)

print(message2.content[0].text)






df_off = utils.get_ref_sc_off_target()
ref_med_adata = utils.get_ref_lv4_sc_med_adata()
ref_med_adata = utils.add_ref_weights(ref_med_adata, df_off, "Off_Target.V0")

df = tid.run.target_id_v1(malig_adata=malig_mean_adata, ref_adata=ref_med_adata)
df2=df.loc[df["gene_name"].isin(surface),:]
df2[df2["Positive_Final_v2"] > 15].head(50)
df2 = pd.merge(df2, tid.utils.get_single_risk_scores(), how = "left")
df2 = tid.run.target_quality_v2_01(df2)
df2[df2["Hazard_GTEX_v1"] < 25].head(30)

tid.utils.ref_med_gene_sorted(ref_med_adata, "IGSF23")
tid.utils.ref_med_gene_sorted(ref_med_adata, "IGSF23")

malig_mean_adata[:,sub_adata.var["gene"]=="LY"].X


np.min(np.sum(sub_adata.raw.X, axis= 0))

df2=df.loc[df["gene_name"].isin(["LY6G6D"]),:]


df2.loc[df2["gene_name"].isin(["KCNA7"]),:].iloc[0]

malig_mean_adata[:,"NOX1"].X


disease_map = {
    'HER2 positive breast carcinoma': 'BRCA-HER2pos',
    'breast apocrine carcinoma': 'BRCA-Apocrine',
    'breast cancer': 'BRCA-BC',
    'breast carcinoma': 'BRCA-Carcinoma',
    'breast mucinous carcinoma': 'BRCA-Mucinous',
    'estrogen-receptor positive breast cancer': 'BRCA-ERpos',
    'invasive ductal breast carcinoma': 'BRCA-IDC',
    'invasive lobular breast carcinoma': 'BRCA-ILC',
    'invasive tubular breast carcinoma || invasive lobular breast carcinoma': 'BRCA-Tubular-ILC',
    'metaplastic breast carcinoma': 'BRCA-Metaplastic',
    'triple-negative breast carcinoma': 'BRCA-TNBC'
}

for subtype in disease_map.keys():
    print(subtype)
    mask = (adata.obs["cell_type"].str.contains("^malignant")) & (adata.obs["disease"]==subtype)
    sub_adata = adata[mask, :].to_memory()
    malig_mean_adata = ad.AnnData(
        X=tid.utils.summarize_matrix(tid.utils.cp10k(sub_adata.raw.X), groups = sub_adata.obs["donor_id"].values, metric="mean", axis=0)
    )
    malig_mean_adata.var_names = sub_adata.var["feature_name"].values.astype(str)
    malig_mean_adata.var_names_make_unique()
    malig_mean_adata = malig_mean_adata[:, malig_mean_adata.var_names.isin(valid_genes)].copy()
    malig_mean_adata = tid.run.compute_positivity_matrix(malig_mean_adata, fallback_threshold = 0.5)
    malig_mean_adata.write_h5ad("processed/" + disease_map[subtype] + ".h5ad")






https://datasets.cellxgene.cziscience.com/62ebf73a-704b-432f-8c39-f05072209c27.h5ad













ref_med_adata = tid.utils.get_ref_lv4_ffpe_med_adata()
df = tid.run.target_id_v1(malig_adata=malig_mean_adata, ref_adata=ref_med_adata)
df2=df.loc[df["gene_name"].isin(surface),:]
df2[df2["Positive_Final_v2"] > 10].head(50)
df2 = pd.merge(df2, tid.utils.get_single_risk_scores(), how = "left")
df2 = tid.run.target_quality_v2_01(df2)
df2[(df2["Hazard_SC_v1"] < 20)].head(25)
df2[df2.gene_name.isin(["ABCC11"])].iloc[0]


df2.head(25)
df2.iloc[0]

# Apply the mapping
sub_adata.obs['disease_short'] = sub_adata.obs['disease'].map(disease_map)

disease_map[sub_adata.obs['disease'].values[0]]


sub_adata.obs['disease'].values[0].map(disease_map)

















df = tid.run.target_id_v1(malig_adata=sclc_a, ref_adata=ref_med_adata)
surface = tid.utils.surface_genes()
df2=df.loc[df["gene_name"].isin(surface),:]
df2[df2["Positive_Final_v2"] > 20].head(50)



#Summarize Expression
adata.var_names = adata.var["feature_name"].values.astype(str)



adata = sc.read_h5ad('5b7cbce2-ed53-47a4-99f8-10af741814e6.h5ad')
mask = adata.obs["cell_type_fine"].str.contains("^SCLC-A")
sclc = adata[mask, :].copy()

#CP10k
sclc.X = tid.utils.cp10k(sclc.X)

#Summarize Expression
sclc.var_names = sclc.var["feature_name"].values.astype(str)

#Summarize
malig_mean_adata = ad.AnnData(
    X=tid.utils.summarize_matrix(sclc.X, groups = sclc.obs["donor_id"].values, metric="mean", axis=0)
)
malig_mean_adata.var_names = sclc.var["feature_name"].values.astype(str)
malig_mean_adata.var_names_make_unique()

#Get Ref
ref_med_adata = tid.utils.get_ref_lv4_sc_med_adata()

#Join
malig_mean_adata = malig_mean_adata[:, malig_mean_adata.var_names.isin(ref_med_adata.var_names)]
malig_mean_adata = malig_mean_adata[malig_mean_adata.obs_names != "PleuralEffusion", :].copy()
ref_med_adata = ref_med_adata[:, malig_mean_adata.var_names]

#Positivity (CP10k > 0.5)
malig_mean_adata=tid.run.compute_positivity_matrix(malig_mean_adata, fallback_threshold = 0.5)

#Run
df = tid.run.target_id_v1(malig_adata=malig_mean_adata, ref_adata=ref_med_adata)

#Find
surface = tid.utils.surface_genes()
df2=df.loc[df["gene_name"].isin(surface),:]
df2[df2["Positive_Final_v2"] > 25].head(50)
df2[df2.gene_name=="DLL3"].iloc[0]






















df[df.gene_name=="DLL3"].iloc[0]
df[df["Positive_Final_v2"] > 50].head(50)

# Get DLL3 expression

dll3 = sclc[:, "ENSG00000090932"].X.toarray().flatten()
donor = sclc.obs["donor_id"].values
result = {d: np.mean(dll3[donor == d]) for d in np.unique(donor)}





# Group by donor and get mean
split(dll3, donor) %>% lapply(median) %>% unlist


result = {d: np.mean(dll3[donor == d]) for d in np.unique(donor)}




cp10k = sclc.raw.X
row_sums = np.array(cp10k.sum(axis=1)).flatten()
cp10k_normalized = cp10k.multiply(1 / row_sums.reshape(-1, 1))

def cp10k(m):
    """Normalize matrix rows to sum to 1 (CPM per 10k normalization)
    
    Preserves input matrix format (sparse or dense).
    """
    row_sums = np.array(m.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    
    scale = 10000 / row_sums.reshape(-1, 1)
    
    if hasattr(m, 'multiply'):  # Sparse matrix
        result = m.multiply(scale)
        # Preserve original format
        if hasattr(m, 'format'):
            result = result.asformat(m.format)
    else:  # Dense array
        result = m * scale
    
    return result



pearsonr(np.sum(cp10k, axis=1).flatten(), 10**sclc.obs["libsize"].values)

correlation, pvalue = pearsonr(
    np.array(np.sum(cp10k, axis=1)).flatten(),
    10**sclc.obs["libsize"].values
)

cp10k = sclc.X.copy()
cp10k.data = np.exp(cp10k.data)-1
sclc.X = cp10k

np.sum(np.exp(sclc.X[0,:].toarray().flatten())-1)


libsize_raw = 10 ** sclc.obs["libsize"]
raw_counts = np.expm1(sclc.X)
np.sum(2**sclc.X[0,:].toarray().flatten()-1)
np.sum(np.exp(sclc.X[1,:].toarray().flatten()-1))

dll3 = (sclc[:,"ENSG00000090932"].X.toarray())
sclc.obs["donor_id"]

# Get DLL3 expression
dll3 = sclc[:, "ENSG00000090932"].X.toarray().flatten()
np.sum(sclc.X, axis = 1)

# Add to obs
sclc.obs["DLL3"] = dll3

# Group by donor and get mean
dll3_per_donor = sclc.obs.groupby("donor_id")["DLL3"].mean()


Gene: DLL3 (ENSG00000090932) - Summary
