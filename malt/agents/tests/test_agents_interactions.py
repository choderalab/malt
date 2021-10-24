import pytest


def test_import():
    import malt
    import malt.agents


def test_player_with_linear_alkane():
    import malt
    from malt.agents.player import SequentialModelBasedPlayer
    player = SequentialModelBasedPlayer(
       model = malt.models.supervised_model.SimpleSupervisedModel(
           representation=malt.models.representation.DGLRepresentation(
               out_features=128
           ),
           regressor=malt.models.regressor.NeuralNetworkRegressor(
               in_features=128, out_features=1
           ),
           likelihood=malt.models.likelihood.HomoschedasticGaussianLikelihood(),
       ),
       policy=malt.policy.Greedy(),
       trainer=malt.trainer.get_default_trainer(),
       merchant=malt.agents.merchant.DatasetMerchant(
           malt.data.collections.linear_alkanes(50),
       ),
       assayer=malt.agents.assayer.DatasetAssayer(
           malt.data.collections.linear_alkanes(50),
       )
    )

    while True:
        if player.step() is None:
            break

    print([point for point in player.portfolio.points])
