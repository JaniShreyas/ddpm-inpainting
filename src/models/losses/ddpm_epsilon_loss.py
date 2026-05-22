# The idea is that there are 3 types
# Each type is based on the prediction type defined in the config
# For this epsilon_loss.py, the prediction type will correspondingly require updating the backbone's output (the input to the function) to epsilon

import torch
import torch.nn as nn
import torch.nn.functional as F
from .prediction_or_loss_type import PredictionOrLossType

# The corresponding changes from x and v to epsilon require the cumulative noise alpha
# These can be defined in the EpsilonLoss class

class EpsilonLoss(nn.Module):
    def __init__(self, alphas_cumprod, prediction_type: PredictionOrLossType):
        super().__init__()
        self.prediction_type = prediction_type
        
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        self.prediction_conversion_function = {
            PredictionOrLossType.EPSILON: lambda pred, noisy_image, t: pred,  # Identity function for epsilon prediction
            PredictionOrLossType.V: self._v_to_epsilon,
            PredictionOrLossType.X: self._x_to_epsilon
        }

    def _v_to_epsilon(self, pred, noisy_image, t):
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        pred = sqrt_one_minus_alphas_cumprod_t * noisy_image + sqrt_alphas_cumprod_t * pred
        return pred

    def _x_to_epsilon(self, pred, noisy_image, t):
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        pred = (noisy_image - sqrt_alphas_cumprod_t * pred) / sqrt_one_minus_alphas_cumprod_t
        return pred

    def forward(self, pred, target, noisy_image, t):
        # Convert prediction
        pred = self.prediction_conversion_function[self.prediction_type](pred, noisy_image, t)
        # Calculate loss
        loss = F.mse_loss(pred, target)
        return loss