def test_representation_convert():
    import torch
    import malt
    from malt.models.utils import to_pyro
    representation=malt.models.representation.DGLRepresentation(
        out_features=128
    )
    model, guide = to_pyro(representation)

def test_regressor_convert():
    import torch
    import malt
    from malt.models.utils import to_pyro
    regressor=malt.models.regressor.NeuralNetworkRegressor(
        in_features=128,
    )
    model, guide = to_pyro(regressor)

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

    model, guide = net.to_pyro()
