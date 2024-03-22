
import torch.nn as nn
import torch.nn.functional as F

from src.training.models.set_transformer.modules import ISAB, SAB, PMA
from src.evaluation.GMM_likelihood_evaluation import neg_ll_GMM

class SetTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_heads=8, num_inds=10,
                 num_seeds=1, num_enc_layers=2, num_dec_layers=2, use_ISAB=True):
        super().__init__()
        """ PMA not counted in num_dec_layers. """

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_inds = num_inds
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_seeds = num_seeds
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.use_ISAB = use_ISAB

        # Build Encoder
        enc_layers = []
        if self.use_ISAB == True:
            first_enc_layer = ISAB(dim_in=self.input_dim, dim_out=self.hidden_dim, num_heads=self.num_heads, num_inds=self.num_inds)
        else:
            first_enc_layer = SAB(dim_in=self.input_dim, dim_out=self.hidden_dim, num_heads=self.num_heads)
        enc_layers.append(first_enc_layer)

        for i in range(num_enc_layers - 1):
            if self.use_ISAB == True:
                layer = ISAB(dim_in=self.hidden_dim, dim_out=self.hidden_dim, num_heads=self.num_heads, num_inds=self.num_inds)
            else:
                layer = SAB(dim_in=self.hidden_dim, dim_out=self.hidden_dim, num_heads=self.num_heads)
            enc_layers.append(layer)
        self.enc = nn.Sequential(*enc_layers)

        # Build Decoder
        dec_layers = []
        first_dec_layer = PMA(dim=self.hidden_dim, num_heads=self.num_heads, num_seeds=self.num_seeds)
        dec_layers.append(first_dec_layer)

        for i in range(num_dec_layers):
            if self.use_ISAB == True:
                layer = ISAB(dim_in=self.hidden_dim, dim_out=self.hidden_dim, num_heads=self.num_heads, num_inds=self.num_inds)
            else:
                layer = SAB(dim_in=self.hidden_dim, dim_out=self.hidden_dim, num_heads=self.num_heads)
            dec_layers.append(layer)

        final_layer = nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim, bias=False)
        dec_layers.append(final_layer)

        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        """ Never squeeze? More clean and then squeeze outside?"""
        # if self.num_seeds == 1:
        #     return x.squeeze(1)
        # else:
        #     return x
        return x.squeeze(1)

class ParamEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_components):
        super().__init__()
        self.embedder = SetTransformer(input_dim=input_dim, output_dim=embedding_dim, hidden_dim=64, num_seeds=1)
        """ Very unclean, but just for testing out."""
        _tmp = SetTransformer(input_dim=embedding_dim, output_dim=1, num_seeds=num_components)
        _tmp.dec = nn.Sequential(*(list(_tmp.dec.children())[:-1]))
        self.estimator = _tmp

        self.mu_linear = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=input_dim),
        )
        self.sigma_linear = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=input_dim)
        )
        self.pi_linear = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=1)
        )

    def forward(self, x):
        x_embed = self.embedder(x)
        output = self.estimator(x_embed.unsqueeze(1))

        mu = self.mu_linear(output)
        sigma = F.softplus(self.sigma_linear(output))
        pi = F.softmax(self.pi_linear(output).squeeze(2), dim=1)

        return x_embed, pi, mu, sigma


import torch
import math

def main():
    # model = SetTransformer(input_dim=3, output_dim=3)
    model = ParamEstimator(input_dim=64, hidden_dim=64, embedding_dim=5, num_components=2)#, num_inds_estimator=10)
    x = torch.randn(size=(13, 14, 64))
    x_embed, mu, sigma, pi = model(x)
    print(mu.shape)
    print(sigma.shape)
    print(pi.shape)
    print(pi)

    # FROM MVN diag: def log_prob(self, X, params):
    # mu, sigma = params

    loss = neg_ll_GMM(x, (pi, mu, sigma))

    print(loss)

if __name__ == "__main__":
    main()