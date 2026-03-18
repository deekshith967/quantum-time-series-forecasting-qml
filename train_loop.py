from pennylane import numpy as np
import torch
import torch.optim as optim
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

BASE_DIR = "./QEncoder_SP500_prediction/"


def accuracy(y, y_hat):

    r2 = r2_score(y, y_hat)
    mse = mean_squared_error(y, y_hat)
    mae = mean_absolute_error(y, y_hat)
    mape = mean_absolute_percentage_error(y, y_hat)

    return r2, mse, mae, mape


def train(
    model,
    train_set,
    labels_train,
    validation_set,
    labels_val,
    args,
):

    print("Training Started...", flush=True)

    losses, r2_scores, mses, maes, mapes = [], [], [], [], []

    import csv
    log_file = os.path.join(".", "results", "training_log.csv")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss", "R2", "MSE", "MAE"])

    opt = optim.RMSprop(model.parameters(), lr=args.lr)

    experiment_dir = BASE_DIR
    os.makedirs(experiment_dir, exist_ok=True)

    evaluation_results_dir = os.path.join(experiment_dir, "evaluation_results")
    os.makedirs(evaluation_results_dir, exist_ok=True)

    os.makedirs(os.path.join(evaluation_results_dir, "accs"), exist_ok=True)
    os.makedirs(os.path.join(evaluation_results_dir, "losses"), exist_ok=True)
    os.makedirs(os.path.join(evaluation_results_dir, "weights"), exist_ok=True)

    experiment = f"{args.dataset}_{args.loss}_{args.depth}_{args.n_cells}_{args.num_latent}_{args.num_trash}"

    if args.loss == "BCE":
        loss_fun = torch.nn.BCELoss()
    elif args.loss == "MSE":
        loss_fun = torch.nn.MSELoss()
    else:
        raise ValueError("Unsupported loss")

    for i in range(args.train_iter):

        model.train()
        opt.zero_grad()

        # 🔥 RANDOM BATCH SAMPLING
        train_indices = np.random.randint(
            0,
            len(train_set),
            (args.batch_size,),
        )

        train_labels = torch.tensor(
            [labels_train[x] for x in train_indices],
            dtype=torch.float32,
        )

        features = torch.tensor(
            np.array([train_set[x] for x in train_indices]),
            dtype=torch.float32,
        )

        out = model(features)

        loss = loss_fun(out, train_labels)

        loss.backward()
        opt.step()

        r2_train, mse_train, mae_train, mape_train = accuracy(
            np.array(train_labels.detach()),
            np.array(out.detach()),
        )

        print(
            f"Train {i}: Loss={loss.item():.5f} R2={r2_train:.4f}",
            flush=True,
        )

        losses.append(loss.item())

        # Save log
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([i, loss.item(), r2_train, mse_train, mae_train])

        # 🔹 VALIDATION
        if i % args.eval_every == 0:

            model.eval()

            val_indices = np.random.randint(
                0,
                len(validation_set),
                (args.batch_size,),
            )

            val_labels = torch.tensor(
                [labels_val[x] for x in val_indices],
                dtype=torch.float32,
            )

            val_features = torch.tensor(
                np.array([validation_set[x] for x in val_indices]),
                dtype=torch.float32,
            )

            with torch.no_grad():
                out_val = model(val_features)

            r2_val, mse_val, mae_val, mape_val = accuracy(
                np.array(val_labels),
                np.array(out_val),
            )

            print(
                f"Validation {i}: R2={r2_val:.4f} MSE={mse_val:.4f}",
                flush=True,
            )

            r2_scores.append(r2_val)
            mses.append(mse_val)
            maes.append(mae_val)
            mapes.append(mape_val)

            # 🔹 SAVE BEST MODEL
            if len(mses) == 1 or mse_val <= min(mses[:-1]):

                torch.save(
                    {
                        "epoch": i,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "losses": losses,
                        "r2_scores": r2_scores,
                        "mses": mses,
                        "maes": maes,
                        "mapes": mapes,
                    },
                    os.path.join(
                        evaluation_results_dir,
                        "weights",
                        f"{experiment}_weights",
                    ),
                )

            # 🔹 SAVE METRICS
            np.save(
                os.path.join(evaluation_results_dir, "accs", f"r2_{experiment}.npy"),
                r2_scores,
            )

            np.save(
                os.path.join(evaluation_results_dir, "accs", f"mse_{experiment}.npy"),
                mses,
            )

            np.save(
                os.path.join(evaluation_results_dir, "accs", f"mae_{experiment}.npy"),
                maes,
            )

            np.save(
                os.path.join(evaluation_results_dir, "accs", f"mape_{experiment}.npy"),
                mapes,
            )

    return model