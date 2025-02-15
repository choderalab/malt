import pytest

def test_training_on_linear_alkane_without_player():
    import malt
    data = malt.data.collections.linear_alkanes(10)
    representation = malt.models.representation.DGLRepresentation(out_features=32)
    regressor=malt.models.regressor.NeuralNetworkRegressor(
        in_features=32,
    )

    model = malt.models.supervised_model.SupervisedModel(
        representation=representation,
        regressor=regressor,
    )
    trainer = malt.trainer.get_default_trainer(
        without_player=True,
    )
    model = trainer(model, data, data)

def test_training_on_linear_alkane_with_player():
    import malt
    import torch
    data = malt.data.collections.linear_alkanes(10)
    merchant = malt.agents.merchant.DatasetMerchant(data)
    assayer = malt.agents.assayer.DatasetAssayer(data)

    representation = malt.models.representation.DGLRepresentation(out_features=32)
    regressor=malt.models.regressor.NeuralNetworkRegressor(
        in_features=32,
    )
    model = malt.models.supervised_model.SupervisedModel(
        representation=representation,
        regressor=regressor,
    )

    if torch.cuda.is_available():
        model.cuda()

    player = malt.agents.player.SequentialModelBasedPlayer(
        model=model,
        merchant=merchant,
        assayer=assayer,
        policy=malt.policy.UtilityFunction(),
        trainer=malt.trainer.get_default_trainer(),
    )

    player.step()
