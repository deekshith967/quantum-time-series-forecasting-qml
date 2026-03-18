import os
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from pennylane import numpy as np
import copy

BASE_DIR = "./encoder_details_latent_4_trash_6/"
os.makedirs(BASE_DIR, exist_ok=True)

print("encoder.py loaded")


def construct_autoencoder_circuit(args, weights, features=None):

    dev = qml.device("default.qubit", wires=args.num_latent + 2 * args.num_trash + 1)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def autoencoder_circuit(weights, features=None):

        weights = weights.reshape(-1, args.num_latent + args.num_trash)

        if features is not None:
            qml.AngleEmbedding(
                features[:, : args.num_latent + args.num_trash],
                wires=range(args.num_latent + args.num_trash),
                rotation="X",
            )

        qml.BasicEntanglerLayers(
            weights,
            wires=range(args.num_latent + args.num_trash),
        )

        aux_qubit = args.num_latent + 2 * args.num_trash

        qml.Hadamard(wires=aux_qubit)

        for i in range(args.num_trash):
            qml.CSWAP(
                wires=[
                    aux_qubit,
                    args.num_latent + i,
                    args.num_latent + args.num_trash + i,
                ]
            )

        qml.Hadamard(wires=aux_qubit)

        return qml.probs(wires=[aux_qubit])

    return autoencoder_circuit(weights, features)


class AutoEncoder(nn.Module):

    def __init__(self, args, weights=None):
        super().__init__()

        self.n_qubits = args.num_latent + args.num_trash
        self.args = args

        self.weights = (
            weights
            if weights is not None
            else nn.Parameter(
                0.01 * torch.randn(args.depth * self.n_qubits),
                requires_grad=True,
            )
        )

    def forward(self, features):

        probs = construct_autoencoder_circuit(
            self.args,
            self.weights,
            features,
        )

        return probs[:, 1]


def autoencoder_circuit_trained(weights, args):

    weights = weights.reshape(-1, args.num_latent + args.num_trash)

    qml.BasicEntanglerLayers(
        weights,
        wires=range(args.num_latent + args.num_trash),
    )


def train_encoder(flattened, args):

    print("\nStarting Encoder Training\n")

    dataset = args.dataset

    best_model_path = os.path.join(
        BASE_DIR, f"{dataset}_best_encoder_weights.pth"
    )

    if os.path.exists(best_model_path):

        print("Loading existing encoder")

        enc = AutoEncoder(args)
        enc.load_state_dict(torch.load(best_model_path))
        enc.eval()

        return enc

    enc = AutoEncoder(args)

    optimizer = optim.SGD(enc.parameters(), lr=args.lr)

    best_loss = float("inf")

    losses = []

    for i in range(1, 301):

        train_indices = np.random.randint(
            0,
            len(flattened),
            (args.batch_size,),
        )

        features = torch.tensor(
            np.array([flattened[x] for x in train_indices]),
            dtype=torch.float32,
        )

        enc.zero_grad()

        output = enc(features)

        loss = torch.sum(output)

        loss.backward()

        optimizer.step()

        current_loss = float(loss / args.batch_size)

        losses.append(current_loss)

        print(f"Encoder Loss: {current_loss:.4f} | Iteration {i}")

        if current_loss < best_loss:

            best_loss = current_loss

            torch.save(enc.state_dict(), best_model_path)

            print("New best encoder saved")

    np.save(
        os.path.join(BASE_DIR, f"{dataset}_encoder_losses.npy"),
        losses,
    )

    return enc
