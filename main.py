from NeuralLib import GRUseq2seq

# Example usage
model = GRUseq2seq(model_name='model1', n_features=1, hid_dim=64, n_layers=3, dropout=0.3, learning_rate=0.001)

if __name__ == '__main__':
    print(model)
