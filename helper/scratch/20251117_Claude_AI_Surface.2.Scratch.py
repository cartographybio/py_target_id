import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Dict, List
from collections import defaultdict
import json

# ============================================================================
# Configuration
# ============================================================================

# Biomedical LLMs on Hugging Face (local inference)
BIO_MODELS = {
    "BioGPT": "microsoft/biogpt",
    "SciBERT": "allenai/scibert_scivocab_uncased",
    "PubMedBERT": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    "BioBERT": "dmis-lab/biobert-base-cased-v1.2",
}

# Check GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("Running on CPU")

print(f"Using device: {DEVICE}\n")

# ============================================================================
# BioGPT Querier - Generative Model
# ============================================================================

class BioGPTQuerier:
    """Query BioGPT for cell surface protein generation"""
    
    def __init__(self, model_id="microsoft/biogpt", device=DEVICE):
        self.model_id = model_id
        self.device = device
        print(f"Loading {model_id}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.model.eval()
        print(f"✓ {model_id} loaded\n")
    
    def query(self, prompt: str, max_length: int = 512) -> str:
        """Generate text from prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def extract_proteins_from_response(self, text: str) -> List[str]:
        """Extract protein names/genes from generated text"""
        # Simple heuristic: look for capitalized words and common gene patterns
        proteins = []
        words = text.split()
        
        for word in words:
            # CD markers (CD4, CD8, etc.)
            if word.startswith("CD") and word[2:].isdigit():
                proteins.append(word)
            # Common surface proteins in all caps or title case
            elif word.isupper() and len(word) >= 3 and word not in ["THE", "AND", "FOR", "ARE"]:
                proteins.append(word)
        
        return list(set(proteins))

# ============================================================================
# SciBERT / PubMedBERT - Token Classification
# ============================================================================

class BiomedicalNERQuerier:
    """Use biomedical models for named entity recognition (proteins/genes)"""
    
    def __init__(self, model_id="allenai/scibert_scivocab_uncased", device=DEVICE):
        self.model_id = model_id
        self.device = device
        print(f"Loading {model_id}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.model.eval()
        print(f"✓ {model_id} loaded\n")
    
    def query(self, prompt: str, max_length: int = 256) -> str:
        """Generate based on biomedical text model"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=1,
                temperature=0.7,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

# ============================================================================
# Curator - Aggregates Results
# ============================================================================

class LocalBioCurator:
    def __init__(self):
        self.results = defaultdict(lambda: {
            "mentioned_by": [],
            "count": 0
        })
    
    def add_proteins(self, model_name: str, proteins: List[str]):
        """Track which models identified each protein"""
        for protein in proteins:
            protein_clean = protein.strip().upper()
            if len(protein_clean) >= 2:
                self.results[protein_clean]["mentioned_by"].append(model_name)
                self.results[protein_clean]["count"] += 1
    
    def get_curated_list(self, min_models: int = 1) -> pd.DataFrame:
        """Generate consensus list"""
        rows = []
        for protein, data in self.results.items():
            num_models = len(set(data["mentioned_by"]))
            if num_models >= min_models:
                rows.append({
                    "protein": protein,
                    "num_models": num_models,
                    "models": ", ".join(set(data["mentioned_by"])),
                    "mentions": data["count"]
                })
        
        df = pd.DataFrame(rows)
        df = df.sort_values("num_models", ascending=False)
        return df
    
    def save(self, filename: str = "local_bio_surface_proteins.csv"):
        df = self.get_curated_list()
        df.to_csv(filename, index=False)
        print(f"\n✓ Saved to {filename}")
        return df

# ============================================================================
# Prompts for Surface Protein Discovery
# ============================================================================

PROMPTS = {

    "general": """List human cell surface proteins (Including GPI-Anchored) and their functions. Include: CD4, CD8, CD19, CD20, EGFR, HER2, PD-1""",
    
    "carcell": """List surface proteins  (Including GPI-Anchored)  expressed on cancer cells and tumor-associated cells. Include markers for CAR-T targeting.""",
    
    "tissue": """Describe tissue-specific cell surface proteins  (Including GPI-Anchored). Include: lung epithelial markers, cardiac markers, neuronal markers, liver markers.""",
}

# ============================================================================
# Main Execution
# ============================================================================
biogpt = BioGPTQuerier()

batch = ["PRSS21"]
query = f"Return only Yes or No for each gene. Are these Genes present on the Cell Surface in ANY context (including GPI-anchored)? No matter how minimal.\n\n{', '.join(batch)}"
response = biogpt.query(query, max_length=10000)






curator = LocalBioCurator()

print("=" * 70)
print("LOCAL BIOMEDICAL LLM SURFACE PROTEIN DISCOVERY")
print("=" * 70 + "\n")

# Start with BioGPT (best for generation)
print("[1] Querying BioGPT for surface proteins...\n")
try:
    biogpt = BioGPTQuerier()
    
    for prompt_name, prompt_text in PROMPTS.items():
        print(f"  Prompt: {prompt_name}")
        response = biogpt.query(prompt_text, max_length=300)
        proteins = biogpt.extract_proteins_from_response(response)
        
        curator.add_proteins("BioGPT", proteins)
        print(f"    Found: {', '.join(proteins[:5])}..." if len(proteins) > 5 else f"    Found: {', '.join(proteins)}")
        print()

except Exception as e:
    print(f"  ✗ Error with BioGPT: {e}\n")

# Query additional models
print("[2] Querying SciBERT...\n")
try:
    scibert = BiomedicalNERQuerier(BIO_MODELS["SciBERT"])
    
    for prompt_name, prompt_text in PROMPTS.items():
        print(f"  Prompt: {prompt_name}")
        response = scibert.query(prompt_text, max_length=200)
        
        # Extract capitalized terms as potential proteins
        words = response.split()
        proteins = [w for w in words if w[0].isupper() and len(w) >= 2]
        
        curator.add_proteins("SciBERT", proteins)
        print(f"    Found: {', '.join(proteins[:5])}..." if len(proteins) > 5 else f"    Found: {', '.join(proteins)}")
        print()

except Exception as e:
    print(f"  ✗ Error with SciBERT: {e}\n")

# Generate and save results
print("[3] Generating curated list...\n")
df = curator.save()

print(f"Total unique proteins: {len(df)}")
print(f"Proteins mentioned by 2+ models: {len(df[df['num_models'] >= 2])}")
print(f"\nTop 15 consensus proteins:")
print(df.head(15).to_string(index=False))






def main():
    curator = LocalBioCurator()
    
    print("=" * 70)
    print("LOCAL BIOMEDICAL LLM SURFACE PROTEIN DISCOVERY")
    print("=" * 70 + "\n")
    
    # Start with BioGPT (best for generation)
    print("[1] Querying BioGPT for surface proteins...\n")
    try:
        biogpt = BioGPTQuerier()
        
        for prompt_name, prompt_text in PROMPTS.items():
            print(f"  Prompt: {prompt_name}")
            response = biogpt.query(prompt_text, max_length=300)
            proteins = biogpt.extract_proteins_from_response(response)
            
            curator.add_proteins("BioGPT", proteins)
            print(f"    Found: {', '.join(proteins[:5])}..." if len(proteins) > 5 else f"    Found: {', '.join(proteins)}")
            print()
    
    except Exception as e:
        print(f"  ✗ Error with BioGPT: {e}\n")
    
    # Query additional models
    print("[2] Querying SciBERT...\n")
    try:
        scibert = BiomedicalNERQuerier(BIO_MODELS["SciBERT"])
        
        for prompt_name, prompt_text in PROMPTS.items():
            print(f"  Prompt: {prompt_name}")
            response = scibert.query(prompt_text, max_length=200)
            
            # Extract capitalized terms as potential proteins
            words = response.split()
            proteins = [w for w in words if w[0].isupper() and len(w) >= 2]
            
            curator.add_proteins("SciBERT", proteins)
            print(f"    Found: {', '.join(proteins[:5])}..." if len(proteins) > 5 else f"    Found: {', '.join(proteins)}")
            print()
    
    except Exception as e:
        print(f"  ✗ Error with SciBERT: {e}\n")
    
    # Generate and save results
    print("[3] Generating curated list...\n")
    df = curator.save()
    
    print(f"Total unique proteins: {len(df)}")
    print(f"Proteins mentioned by 2+ models: {len(df[df['num_models'] >= 2])}")
    print(f"\nTop 15 consensus proteins:")
    print(df.head(15).to_string(index=False))
    
    return curator, df

if __name__ == "__main__":
    curator, results = main()