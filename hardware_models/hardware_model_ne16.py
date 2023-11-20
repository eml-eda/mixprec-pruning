import torch
from plinio.methods.mixprec.nn.mixprec_conv2d import MixPrec_Conv2d
from plinio.methods.mixprec.nn.mixprec_identity import MixPrec_Identity
from plinio.methods.mixprec.nn.mixprec_linear import MixPrec_Linear
from plinio.methods.mixprec.nn.mixprec_qtz import MixPrecType

from Ne16PerfModelGeneralized import Ne16PerfModel_generalized


def compute_cycles_ne16(self) -> torch.Tensor:
    """Compute the number of cycles for a forward pass of the input model by considering the NE16
    accelerator hardware model. The activation precision is assumed to be 8-bit.

    Output
    ------
    - `torch.Tensor`: sum of the number of cycles required by each NAS-able layer of the model"""

    total_latency = torch.tensor(
        0,
        dtype=torch.float32,
        device=next(self.named_nas_parameters())[1].device)

    # Iterate over all the NAS-able layers of the network and over all the possibile weight
    # precisions. The activations precisions are fixed to 8-bit.
    for layer in self._target_layers:
        th = 0.

        if isinstance(layer, MixPrec_Identity):
            continue

        for w_prec in layer.mixprec_w_quantizer.precisions:
            # If we are considering the pruned channels, there is not any contribution to the
            # total latency
            if w_prec == 0:
                continue

            # Compute the "effective" number of channels at the selected weight precision
            if layer.w_mixprec_type == MixPrecType.PER_CHANNEL:
                theta_alpha_sum = layer.mixprec_w_quantizer.theta_alpha.sum(dim=-1)[
                    layer.mixprec_w_quantizer.precisions.index(w_prec)]
            else:
                theta_alpha_sum = layer.mixprec_w_quantizer.theta_alpha[
                    layer.mixprec_w_quantizer.precisions.index(w_prec)] * layer.out_features_eff

            # Extract the layer's parameters in order to initialize the NE16 model
            is_depthwise = False
            if isinstance(layer, MixPrec_Conv2d):
                layer_params = (
                    layer.out_height,
                    layer.out_width,
                    theta_alpha_sum,
                    layer.input_features_calculator.features)
                is_depthwise = (
                    layer.groups == layer.in_channels and layer.groups == layer.out_channels)
                kernel_size = layer.kernel_size
            elif isinstance(layer, MixPrec_Linear):
                layer_params = (
                    1,
                    1,
                    theta_alpha_sum,
                    layer.input_features_calculator.features)
                kernel_size = (1, 1)
            else:
                print("Layer of type '{}' not supported".format(type(layer)))
                continue

            # Instantiate the NE16 model and extract the latency value
            latency, total_ops, total_MACs_cycle = Ne16PerfModel_generalized(
                name='conv',
                ks=kernel_size,
                depthwise=is_depthwise,
                WEIGHTS_BITWIDTH=w_prec,
                layer=layer_params)

            th += latency
        total_latency += th
    return total_latency
