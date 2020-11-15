import matplotlib.pyplot as plt
import seaborn as sns
import gensim
from gensim.models import KeyedVectors as KV
import pandas as pd
import numpy as np
import json

VECS_DB = '../PubMed-and-PMC-w2v.bin'

def main():

    # Load the model
    print('Loading model...')
    model = KV.load_word2vec_format(VECS_DB, binary=True)
    print('end')

    with open('drug_to_gene_symbols.json', 'r') as f:
        drug_to_symbols = json.load(f)
   
    model_words = set(model.vocab.keys())

    all_genes = set()
    drug_toks = set()
    for drug, genes in drug_to_symbols.items():
        toks = drug.split()
        if set(toks) <= model_words and len(set(genes) & model_words) > 0:
            drug_toks.update(toks)
            genes_in_model = set(genes) & model_words
            all_genes.update(genes_in_model)
    all_genes = sorted(all_genes)
    drug_toks = sorted(drug_toks)        


    gene_vecs = model[all_genes]
    df = pd.DataFrame(
        data=gene_vecs,
        index=all_genes
    )
    df.to_csv('drug_genes_word_vectors.tsv', sep='\t')
  
    drug_tok_vecs = model[drug_toks]
    df = pd.DataFrame(
        data=drug_tok_vecs,
        index=drug_toks
    )
    df.to_csv('drug_token_word_vectors.tsv', sep='\t') 

if __name__ == '__main__':
    main()
