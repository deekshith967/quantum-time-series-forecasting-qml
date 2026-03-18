import torch
import torch.nn as nn
import numpy as np

from classification_circuits import construct_classification_circuit


class Classifier(nn.Module):

    def __init__(self, encoder, args, model_weights=None):

        super(Classifier, self).__init__()

        self.encoder = encoder
        self.args = args

        n_qubits = args.num_latent + args.num_trash

        n_weights = ((4 * n_qubits - 2) * args.depth) * 4

        if model_weights is None:

            self.model_weights = nn.Parameter(
                0.01 * torch.randn(n_weights * args.n_cells),
                requires_grad=True
            )

        else:

            self.model_weights = nn.Parameter(model_weights)

    def forward(self, features):
        circuit_output = construct_classification_circuit(
            self.args,
            self.model_weights,
            features.detach().numpy(),
            self.encoder
        )
        # circuit_output has shape (batch_size, 2)
        return circuit_output[:, 0].to(torch.float32)
