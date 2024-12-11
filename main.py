from architectures import BaseModel, GRUseq2seq, TransformerSeq2Seq

# Example usage
model = GRUseq2seq(n_features=1, hid_dim=64, n_layers=3, dropout=0.3, learning_rate=0.001, results_directory='')

if __name__ == '__main__':
    print(model)
