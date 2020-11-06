import matplotlib.pyplot as plt
import seaborn as sns
import gensim
from gensim.models import KeyedVectors as KV
import umap
import pandas as pd
import numpy as np

VECS_DB = '../PubMed-and-PMC-w2v.bin'

def main():

    # Load the model
    print('Loading model...')
    model = KV.load_word2vec_format(VECS_DB, binary=True)
    print('end')

    # Load genes in gene set   
    gene_set_to_genes = _parse_gene_sets('./gene_sets/h.all.v7.1.symbols.gmt')
    gene_set_to_genes = {k: set(v) for k,v in gene_set_to_genes.items()}
    all_genes = set()
    for genes in gene_set_to_genes.values():
        all_genes.update(genes)

    # Filter genes by those only in WV model
    genes_in_model = sorted(all_genes & set(model.vocab.keys()))
    print("{} total genes".format(len(all_genes)))
    print("{} genes in model".format(len(all_genes))) 
    vecs = model[genes_in_model]
    print(np.sum(vecs, axis=1))
    print(vecs.shape)

    df = pd.DataFrame(
        data=vecs,
        index=genes_in_model
    )
    df.to_csv('hallmark_genes_word_vectors.tsv', sep='\t')
  


def _parse_gene_sets(gene_sets_f):
    gene_set_to_genes = {}
    with open(gene_sets_f, 'r') as f:
        for l in f:
            toks = l.split('\t')
            gene_set = toks[0]
            genes = [x.strip() for x in toks[2:]]
            gene_set_to_genes[gene_set] = genes
    return gene_set_to_genes

if __name__ == '__main__':
    main()
