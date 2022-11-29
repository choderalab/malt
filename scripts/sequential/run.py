import numpy as np
import torch
import malt

def run(args):
    # data = malt.data.collections.linear_alkanes(10)
    data = getattr(malt.data.collections, args.data)()
    # data = data.shuffle(seed=2666)
    # data_train, data_valid, data_test = data.split([8, 1, 1])
    # data_train = data_valid = data_test = data
    # g, y = next(iter(data.view(batch_size=len(data))))

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
        ),
        regressor=regressor(
            in_features=args.width,
        ),
    )

    
    utility_function = getattr(malt.utility_functions, args.utility_function)
    from malt.agents.player import SequentialModelBasedPlayer
    player = SequentialModelBasedPlayer(
       model = model,
       policy=malt.policy.UtilityFunction(utility_function),
       trainer=malt.trainer.get_default_trainer(),
       merchant=malt.agents.merchant.DatasetMerchant(data),
       assayer=malt.agents.assayer.DatasetAssayer(data),
    )


    while True:
        if player.step() is None:
            break

    import json
    history = [molecule.y for molecule in player.portfolio]
    # history = np.array(history)

    import json
    import pandas as pd
    key = dict(vars(args))
    key.pop("out")
    key = json.dumps(key)
    df = pd.DataFrame.from_dict(
        {key: [json.dumps(history)]},
         orient="index",
    )
    df.to_csv(args.out, header=False, mode="a")

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
    parser.add_argument("--out", type=str, default="out.csv")
    parser.add_argument("--acquisition_size", type=int, default=1)
    parser.add_argument("--policy", type=str, default="Greedy")
    parser.add_argument("--utility_function", type=str, default="expectation")
    args = parser.parse_args()
    run(args)
