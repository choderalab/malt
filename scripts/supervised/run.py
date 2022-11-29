import torch
import malt

def run(args):
    rmses_valid = []
    rmses_test = []

    for seed in [1001, 1984, 2666]:
        torch.manual_seed(seed)
        data = getattr(malt.data.collections, args.data)()
        data = data.shuffle(seed=seed)
        data_train, data_valid, data_test = data.split([8, 1, 1])
        g, y = next(iter(data_train.view(batch_size=len(data_train))))
        regressor = getattr(
            malt.models.regressor,
            {
                "nn": "NeuralNetworkRegressor", "gp": "ExactGaussianProcessRegressor"
            }[args.regressor],
        )

        model = malt.models.SupervisedModel(
            representation=malt.models.representation.DGLRepresentation(
                depth=args.depth,
                out_features=args.width,
                layer=getattr(malt.models.zoo, args.layer),
            ),
            regressor=regressor(
                num_points=len(data_train),
                in_features=args.width,
            ),
        )

        if args.regressor == "nn":
            batch_size = args.batch_size
        else:
            batch_size = -1

        trainer = malt.trainer.get_default_trainer(
            without_player=True,
            n_epochs=5000,
            learning_rate=args.learning_rate,
            reduce_factor=args.reduce_factor,
            batch_size=batch_size,
        )
        model = trainer(model, data_train, data_valid)
        model.eval()

        g, y = next(iter(data_test.view(batch_size=len(data_test))))
        y_hat = model(g).loc.squeeze()
        rmse_test = (y_hat - y).pow(2).mean().pow(0.5).item()

        g, y = next(iter(data_valid.view(batch_size=len(data_test))))
        y_hat = model(g).loc.squeeze()
        rmse_valid = (y_hat - y).pow(2).mean().pow(0.5).item()

        rmses_valid.append(rmse_valid)
        rmses_test.append(rmse_test)
    
    import numpy as np
    rmses_valid = np.array(rmses_valid)
    rmses_test = np.array(rmses_test)

    import json
    import pandas as pd
    key = dict(vars(args))
    key.pop("out")
    key = json.dumps(key)
    df = pd.DataFrame.from_dict(
            {key: [rmses_valid.mean(), rmses_valid.std(), rmses_test.mean(), rmses_test.std()]},
        orient="index",
        columns=["rmse_valid", "rmse_valid_std", "rmse_test", "rmse_test_std"],
    )
    df.to_csv(args.out, mode="a")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="esol")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--regressor", type=str, default="gp")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--reduce_factor", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--layer", type=str, default="GCN")
    parser.add_argument("--out", type=str, default="out.csv")
    args = parser.parse_args()
    run(args)
