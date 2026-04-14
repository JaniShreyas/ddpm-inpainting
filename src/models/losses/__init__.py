from .epsilon_loss import EpsilonLoss
from .v_loss import VLoss
from .x_loss import XLoss
from .prediction_or_loss_type import PredictionOrLossType

LOSS_REGISTRY = {
    PredictionOrLossType.EPSILON: EpsilonLoss,
    PredictionOrLossType.V: VLoss,
    PredictionOrLossType.X: XLoss
}

def get_loss_function(loss_type: PredictionOrLossType, alphas_cumprod, prediction_type: PredictionOrLossType):
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    return LOSS_REGISTRY[loss_type](alphas_cumprod, prediction_type)