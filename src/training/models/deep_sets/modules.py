import torch.nn as nn
import torch
import torch.nn.functional as F


from src.training.div.utils import count_parameters

# def __init__(self, input_dim, output_dim, hidden_dim=64, num_heads=8, num_inds=10,
#              num_seeds=1, num_enc_layers=2, num_dec_layers=2, use_ISAB=True):

class DeepSet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, pooling_op, num_enc_layers=2, num_dec_layers=1, hidden_activation=nn.ReLU(), output_activation=nn.ReLU()):
        super(DeepSet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim # The latent dimension
        self.pooling_op = pooling_op
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers

        # Build Encoder that maps vectors from the input to the latent space.
        enc_layers = []
        for i in range(num_enc_layers):
            enc_layers.append(nn.Linear(input_dim, hidden_dim))
            enc_layers.append(hidden_activation)
            input_dim = hidden_dim
        self.enc = nn.Sequential(*enc_layers)

        # Build Decoder that maps vectors from the latent space to the output space.
        dec_layers = []
        for i in range(num_dec_layers - 1):
            dec_layers.append(nn.Linear(hidden_dim, hidden_dim))
            dec_layers.append(hidden_activation)
        dec_layers.append(nn.Linear(hidden_dim, output_dim))
        if output_activation is not None:
            dec_layers.append(output_activation)

        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        """Assumes input is [B, L, d] """
        B, L, _ = x.shape

        x = self.enc(x)
        x = self.pooling_op(x, dim=1)
        x = self.dec(x)
        return x



def main():
    model = DeepSet(141, 128, 256, pooling_op=torch.sum, num_enc_layers=2, num_dec_layers=2)
    X = torch.rand(4, 73, 141)

    output = model(X)
    print(output.shape)
    #177920
    print(count_parameters(model))

    res = torch.mean(X, dim=1)
    print(res)
    print(res.shape)

if __name__ == "__main__":
    main()

