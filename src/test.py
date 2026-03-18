import os
import torch
import numpy as np
from metrics import metrics


def test(
    model,
    args,
    test_dir,
    test_set,
    labels_test,
):

    experiment = f"{args.dataset}_{args.loss}_{args.depth}_{args.n_cells}_{args.num_latent}_{args.num_trash}"

    checkpoint_path = os.path.join(
        test_dir,
        f"{experiment}_weights",
    )

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    with torch.no_grad():
        test_tensor = torch.tensor(test_set, dtype=torch.float32)
        batch_size = args.batch_size
        all_preds = []
        for i in range(0, len(test_tensor), batch_size):
            batch_data = test_tensor[i:i+batch_size]
            preds = model(batch_data)
            all_preds.append(preds)
        
        predictions = torch.cat(all_preds)

    metrics(predictions, labels_test)

    # Return for plotting
    return predictions.numpy(), np.array(labels_test)
