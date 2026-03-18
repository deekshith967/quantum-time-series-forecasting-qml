import os
import argparse

from train_loop import train
from encoder import train_encoder
from classification_model import Classifier
from dataset import load_dataset, split_features_labels
from test import test


parser = argparse.ArgumentParser()

parser.add_argument("--loss", type=str, default="MSE")
parser.add_argument("--eval_every", type=int, default=1)
parser.add_argument("--dataset", type=str, default="sp500")
parser.add_argument("--train_iter", type=int, default=200)

parser.add_argument("--n_cells", type=int, default=5)
parser.add_argument("--depth", type=int, default=2)

parser.add_argument("--mode", type=str, default="train")

parser.add_argument("--num_latent", type=int, default=4)
parser.add_argument("--num_trash", type=int, default=6)

parser.add_argument("--lr", type=float, default=0.003)
parser.add_argument("--batch_size", type=int, default=128)

# 🔹 ADD THIS (fix for your error)
parser.add_argument("--test_percentage", type=float, default=0.2)

args = parser.parse_args()

print("Arguments:", args)


# Load dataset
X, Y, tX, tY, flattened = load_dataset(args)

print("Dataset loaded:", X.shape)


# Train encoder
trained_encoder = train_encoder(flattened, args)


# Initialize classifier
model = Classifier(trained_encoder, args)


# Split training/validation
train_set, validation_set, labels_train, labels_val = split_features_labels(
    X,
    Y,
    0.2,
)

test_set = tX
labels_test = tY


# Run training or testing
if args.mode == "train":

    train(
        model,
        train_set,
        labels_train,
        validation_set,
        labels_val,
        args,
    )

elif args.mode == "test":

    BASE_DIR = "./QEncoder_SP500_prediction/"

    test_dir = os.path.join(
        BASE_DIR,
        "evaluation_results/weights/",
    )

    import numpy as np
    preds, actuals = test(
        model,
        args,
        test_dir,
        test_set,
        labels_test,
    )
    import os
    os.makedirs("results", exist_ok=True)
    np.save("results/test_predictions.npy", preds)
    np.save("results/test_actuals.npy", actuals)