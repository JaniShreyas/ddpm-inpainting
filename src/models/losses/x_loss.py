import torch.nn.functional as F
from .prediction_type import PredictionOrLossType

class XLoss:
    def __init__(self, alphas_cumprod, prediction_type: PredictionOrLossType):
        self.alphas_cumprod = alphas_cumprod
        self.prediction_type = prediction_type
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()
        self.prediction_conversion_function = {
            PredictionOrLossType.EPSILON: self._epsilon_to_x,
            PredictionOrLossType.V: self.v_to_x,
            prediction_type.X: lambda pred, noisy_image, t: pred  # Identity function for x prediction
        }

    def _epsilon_to_x(self, pred, noisy_image, t):
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        pred = (noisy_image - sqrt_one_minus_alphas_cumprod_t * pred) / sqrt_alphas_cumprod_t
        return pred
    
    def v_to_x(self, pred, noisy_image, t):
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        pred = sqrt_alphas_cumprod_t * noisy_image - sqrt_one_minus_alphas_cumprod_t * pred
        return pred

    def __call__(self, pred, target, noisy_image, t):
        # Convert prediction
        pred = self.prediction_conversion_function[self.prediction_type](pred, noisy_image, t)
        # Calculate loss
        loss = F.mse_loss(pred, target)
        return loss