def test_representation_convert():
    import torch
    import malt
    from malt.models.utils import to_pyro
    representation=malt.models.representation.DGLRepresentation(
        out_features=128
    )
    dataset = malt.data.collections.linear_alkanes(5)
    g, y = next(iter(dataset.view(batch_size=5)))
    y_hat = representation.model(g)
    assert list(y_hat.shape) == [5, 128]

def test_regressor_convert():
    import torch
    import malt
    regressor=malt.models.regressor.NeuralNetworkRegressor(
        in_features=128,
    )
    model = regressor.get_model()
    guide = regressor.get_guide()

    h = torch.zeros(2, 128)
    y_hat = model(h)

def test_convert():
    import torch
    import malt

    net = malt.models.supervised_model.SupervisedModel(
        representation=malt.models.representation.DGLRepresentation(
            out_features=128
        ),
        regressor=malt.models.regressor.NeuralNetworkRegressor(
            in_features=128,
        ),
    )

    dataset = malt.data.collections.linear_alkanes(5)
    g, y = next(iter(dataset.view(batch_size=5)))
    y_hat = net.model(g, y)

def test_train():
    import torch
    import malt
    dataset = malt.data.collections.linear_alkanes(5)
    model = malt.models.supervised_model.SupervisedModel(
        representation=malt.models.representation.DGLRepresentation(
            out_features=4
        ),
        regressor=malt.models.regressor.NeuralNetworkRegressor(
            in_features=4,
        ),
    )

    trainer = malt.trainer.get_default_trainer_pyro(
        n_epochs=10, without_player=True,
    )

    model, guide = trainer(model, dataset, dataset)

# def test_predictive():
#     import torch
#     import pyro
#     import malt
#     dataset = malt.data.collections.linear_alkanes(5)
#     model = malt.models.supervised_model.SupervisedModel(
#         representation=malt.models.representation.DGLRepresentation(
#             out_features=4
#         ),
#         regressor=malt.models.regressor.NeuralNetworkRegressor(
#             in_features=4,
#         ),
#     )

#     trainer = malt.trainer.get_default_trainer_pyro(
#         n_epochs=10, without_player=True,
#         guide="AutoDiagonalNormal",
#     )

#     g, y = next(iter(dataset.view(batch_size=5)))
#     model, guide = trainer(model, dataset, dataset)

# def test_gp():
#     import torch
#     import pyro
#     import malt
#     dataset = malt.data.collections.linear_alkanes(5)
#     model = malt.models.supervised_model.SupervisedModel(
#         representation=malt.models.representation.DGLRepresentation(
#             out_features=4
#         ),
#         regressor=malt.models.regressor.ExactGaussianProcessRegressor(
#             in_features=4,
#             num_points=1,
#         ),
#     )

#     trainer = malt.trainer.get_default_trainer_pyro(
#         n_epochs=10, without_player=True,
#         guide="AutoDiagonalNormal",
#     )

#     g, y = next(iter(dataset.view(batch_size=5)))
#     model, guide = trainer(model, dataset, dataset)

#     predictive = pyro.infer.Predictive(model.parametrize, guide=guide, num_samples=100, return_sites=["_RETURN"])
