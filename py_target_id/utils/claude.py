"""
Gene pair generation utilities.
"""
__all__ = ['claude_target_eval_v1']

import anthropic
from pathlib import Path

def claude_target_eval_v1(gene: str, api_key: str, cache_dir: str | Path = "/home/jgranja_cartography_bio/data/cache/claude_target_eval_v1/") -> str:
    """Evaluate a gene as TCE/ADC target. Checks cache first; queries API if not cached."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache_file = cache_dir / f"{gene.upper()}.txt"
    
    if cache_file.exists():
        return cache_file.read_text()

    print("Running Claude for " + gene)
    client = anthropic.Anthropic(api_key = api_key)
    m = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        temperature=0,
        messages=[{"role": "user", "content": f"Evaluate {gene} as a TCE (bispecific T-cell engager) and ADC (antibody-drug conjugate) target: (1) Surface accessibility and epitope presentation; (2) Internalization kinetics - is it rapidly internalized (favorable for ADC, less for TCE)? (3) Antigen shedding or circulating forms that could sequester drug; (4) Overall suitability for TCE vs ADC modalities. Be concise."}]
    )
    
    result = m.content[0].text
    cache_file.write_text(result)
    return result
