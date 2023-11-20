import torch
from plinio.methods.mixprec.nn.mixprec_qtz import MixPrecType
from plinio.methods.mixprec.nn.mixprec_identity import MixPrec_Identity

MPIC = {
    2: {2: 6.5, 4: 4., 8: 2.2},
    4: {2: 3.9, 4: 3.5, 8: 2.1},
    8: {2: 2.5, 4: 2.3, 8: 2.1}}


def mpic_lut(a_bit: int, w_bit: int) -> float:
    """Retrieve the number of MACs/cycle given the activation and weight precision
    according to the MPIC LUT values.
    Reference: "A Mixed-Precision RISC-V Processor for Extreme-Edge DNN Inference", Ottavi et al.
    (https://arxiv.org/pdf/2010.04073.pdf)

    Parameters
    ----------
    - a_bit [`int`]: activation precision
    - w_bit [`int`]: weight precision

    Output
    ------
    - `float`: number of MACs/cycle"""
    return MPIC[a_bit][w_bit]


def compute_cycles_mpic(self) -> torch.Tensor:
    """Compute the number of cycles for a forward pass of the input model.

    Output
    ------
    - `torch.Tensor`: sum of the number of cycles required by each NAS-able layer of the model"""

    total_latency = torch.tensor(
        0,
        dtype=torch.float32,
        device=next(self.named_nas_parameters())[1].device)

    # Iterate over all the NAS-able layers of the network and over all the possibile activations
    # and weight precisions
    for layer in self._target_layers:
        # Skip the identity layer, which corresponds to the first layer
        if isinstance(layer, MixPrec_Identity):
            continue
        macs = layer.get_macs_layer()
        th = 0
        for act_prec in layer.input_quantizer.precisions:
            th_w = 0.
            for w_prec in layer.mixprec_w_quantizer.precisions:
                # If we are considering the pruned channels, there is not any contribution to the
                # total latency
                if w_prec == 0:
                    continue

                if layer.w_mixprec_type == MixPrecType.PER_CHANNEL:
                    theta_alpha_sum = layer.mixprec_w_quantizer.theta_alpha.sum(dim=-1)[
                        layer.mixprec_w_quantizer.precisions.index(w_prec)]
                else:
                    theta_alpha_sum = layer.mixprec_w_quantizer.theta_alpha[
                        layer.mixprec_w_quantizer.precisions.index(w_prec)] * layer.out_features_eff
                theta_alpha_sum_norm = theta_alpha_sum / (layer.out_features_eff + 1e-6)

                th_w += theta_alpha_sum_norm / mpic_lut(act_prec, w_prec)

            th += layer.input_quantizer.theta_alpha[
                layer.input_quantizer.precisions.index(act_prec)] * macs * th_w

        total_latency += th
    return total_latency


def compute_energy_mpic(cycles: torch.Tensor) -> float:
    """Compute the energy consumption according to the MPIC model.

    Parameters
    ----------
    - cycles [`torch.Tensor`]: number of cycles

    Output
    ------
    - `float`: energy consumption in J"""

    frequency = 250 * 1e+6
    mean_power = torch.tensor([5.30, 5.39, 5.46, 5.38]).mean() * 1e-3
    energy = (cycles / frequency) * mean_power
    return energy.item()
