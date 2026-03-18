import pennylane as qml
import torch
import numpy as np
from encoder import autoencoder_circuit_trained


def circuit_7(weights, args):

    n_qubits = args.num_latent + args.num_trash

    w = 0

    for _ in range(args.depth):

        for i in range(n_qubits):
            qml.RX(weights[w], wires=i)
            w += 1

        for i in range(n_qubits):
            qml.RZ(weights[w], wires=i)
            w += 1

        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])


dev = None
cached_qnode = None


def construct_classification_circuit(args, weights, features, trained_encoder=None):
    global dev, cached_qnode

    n_qubits = args.num_latent + 2 * args.num_trash + 1

    if dev is None:
        dev = qml.device("default.qubit", wires=n_qubits)

    if cached_qnode is None:

        @qml.qnode(dev, interface="torch")
        def circuit(w, f, enc_w):

            # f shape = (5 features, 10 timesteps)
            time_steps = f.shape[-1]

            for t in range(time_steps):


                feature = f[:, t]

                if t == 0:
                    qml.AngleEmbedding(
                        feature[:args.num_latent],
                        wires=range(args.num_latent)
                    )
                    circuit_7(w, args)

                else:
                    autoencoder_circuit_trained(enc_w, args)

                    qml.AngleEmbedding(
                        feature[:args.num_latent],
                        wires=range(args.num_latent)
                    )
                    circuit_7(w, args)

            circuit_7(w, args)

            return qml.probs(wires=0)

        cached_qnode = circuit

    return cached_qnode(
        weights,
        torch.tensor(features, dtype=torch.float32),
        trained_encoder.weights
    )
