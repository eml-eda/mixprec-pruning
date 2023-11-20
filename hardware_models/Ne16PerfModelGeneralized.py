import numpy as np
import torch
from numpy import prod


def div_and_ceil(a, b):
    return ((a - 1) // b) + 1


class FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ch, N):
        return torch.floor_divide(ch, N)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class DivAndCeilSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        return ((a - 1) // b) + 1

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class ModuloSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        return a % b

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def Ne16PerfModel_generalized(name, ks, depthwise, WEIGHTS_BITWIDTH, layer):
    n_3x3 = FloorSTE.apply(ks[0], 3) * FloorSTE.apply(ks[1], 3)
    n_1x1 = ModuloSTE.apply(ks[0], 3) * ks[1] + ModuloSTE.apply(ks[1], 3) * ks[0] - ModuloSTE.apply(ks[0], 3) * ModuloSTE.apply(ks[1], 3)
    # print(f"Kernel {ks[0]}x{ks[1]}, n_3x3 {n_3x3}, n_1x1 {n_1x1}")
    total_latency = 0
    total_ops = 0
    if n_3x3 > 0:
        for i in np.arange(n_3x3):
            ne16 = Ne16PerfModel(name, (3,3), depthwise=depthwise, WEIGHTS_BITWIDTH = WEIGHTS_BITWIDTH)
            ne16.set_layer(layer)
            total_ops += ne16.ops
            total_latency += ne16.latency
    if n_1x1 > 0:
        for i in np.arange(n_1x1):
            ne16 = Ne16PerfModel(name, (1,1), depthwise=depthwise, WEIGHTS_BITWIDTH = WEIGHTS_BITWIDTH)
            ne16.set_layer(layer)
            total_ops += ne16.ops
            total_latency += ne16.latency
    return total_latency, total_ops, total_ops/total_latency


class Ne16PerfModel:
    INPUT_BUFFER_SHAPE = (5, 5, 16)
    OUTPUT_BUFFER_SHAPE = (3, 3, 32)
    FIFO_LATENCY = 6
    SHIFTER_COUNT = 4
    ADDER_COUNT = 8
    MULTIPLIER_COUNT = 4
    MEMORY_THROUGHPUT = 256  # bits per cycle

    def __init__(self, operation, kernel_shape, depthwise=False, nq_shift=False, nq_bias=False, nq_bits=32, WEIGHTS_BITWIDTH = 8):
        self.operation = operation
        self.kernel_shape = kernel_shape
        self.depthwise = depthwise
        self.nq_shift = nq_shift
        self.nq_bias = nq_bias
        self.nq_bits = nq_bits
        self.WEIGHTS_BITWIDTH = WEIGHTS_BITWIDTH
        self.INPUT_BITWIDTH = 8
        self.OUTPUT_BITWIDTH = 8
        self.layer = (
                self.OUTPUT_BUFFER_SHAPE[0],
                self.OUTPUT_BUFFER_SHAPE[1],
                self.OUTPUT_BUFFER_SHAPE[2] if not depthwise else self.INPUT_BUFFER_SHAPE[2],
                self.INPUT_BUFFER_SHAPE[2])

    def set_layer(self, layer):
        self.layer = layer
        return self

    def set_subtile(self, h_out=None, w_out=None, k_out=None, k_in=None):
        h_out = h_out if h_out is not None else self.OUTPUT_BUFFER_SHAPE[0]
        w_out = w_out if w_out is not None else self.OUTPUT_BUFFER_SHAPE[1]
        k_out = k_out if k_out is not None else self.OUTPUT_BUFFER_SHAPE[2]
        k_in  = k_in  if k_in  is not None else self.INPUT_BUFFER_SHAPE[2]
        self.INPUT_BUFFER_SHAPE = (h_out + 2, w_out + 2, k_in)
        self.OUTPUT_BUFFER_SHAPE = (h_out, w_out, k_out)

    @property
    def is_3x3(self):
        return self.operation == 'conv' and self.kernel_shape == (3, 3) and not self.depthwise

    @property
    def is_1x1(self):
        return self.operation == 'conv' and self.kernel_shape == (1, 1) and not self.depthwise

    @property
    def is_dw(self):
        return self.operation == 'conv' and self.kernel_shape == (3, 3) and self.depthwise

    @property
    def load_latency(self):
        return 10 + self.OUTPUT_BUFFER_SHAPE[0] * self.OUTPUT_BUFFER_SHAPE[1] * DivAndCeilSTE.apply(self.INPUT_BUFFER_SHAPE[2] * self.INPUT_BITWIDTH, self.MEMORY_THROUGHPUT) if self.is_1x1 \
            else self.FIFO_LATENCY + self.INPUT_BUFFER_SHAPE[0] * self.INPUT_BUFFER_SHAPE[1] * DivAndCeilSTE.apply(self.INPUT_BUFFER_SHAPE[2] * self.INPUT_BITWIDTH, self.MEMORY_THROUGHPUT)

    def weight_offset_latency(self, k):
        return (self.FIFO_LATENCY + k) if self.is_dw else self.FIFO_LATENCY

    def matrixvec_latency(self, k):
        return (self.FIFO_LATENCY + k) if self.is_1x1 else (self.FIFO_LATENCY + k * self.WEIGHTS_BITWIDTH)

    @property
    def update_idx_latency(self):
        return 2

    @property
    def nq_shift_latency(self):
        return 0 if not self.nq_shift else DivAndCeilSTE.apply(self.OUTPUT_BUFFER_SHAPE[2], self.SHIFTER_COUNT)

    def nq_bias_latency(self, k):
        return 0 if not self.nq_bias else 8 + DivAndCeilSTE.apply(k, self.ADDER_COUNT)

    def nq_scale_latency(self, k):
        return 9 + DivAndCeilSTE.apply(k * (self.nq_bits // 8), self.MULTIPLIER_COUNT)

    def normquant_latency(self, k):
        return self.nq_shift_latency + self.nq_scale_latency(k) + self.nq_bias_latency(k)

    @property
    def streamout_latency(self):
        return 3 + self.OUTPUT_BUFFER_SHAPE[0] * self.OUTPUT_BUFFER_SHAPE[1] * DivAndCeilSTE.apply(self.OUTPUT_BUFFER_SHAPE[2] * self.OUTPUT_BITWIDTH, self.MEMORY_THROUGHPUT) + 1  # + end

    @property
    def latency(self):
        k_out_body = self.INPUT_BUFFER_SHAPE[2] if self.is_dw else self.OUTPUT_BUFFER_SHAPE[2]
        n_out_body = FloorSTE.apply(self.layer[2], k_out_body)
        k_out_rem = ModuloSTE.apply(self.layer[2], k_out_body)

        # nothing depends on k_in so no need for remainder
        n_in = DivAndCeilSTE.apply(self.layer[3], self.INPUT_BUFFER_SHAPE[2])

        # depthwise doesn't care about spatial remainder, it just fetches the same
        n_spatial = DivAndCeilSTE.apply(self.layer[0], self.OUTPUT_BUFFER_SHAPE[0]) * DivAndCeilSTE.apply(self.layer[1], self.OUTPUT_BUFFER_SHAPE[1])

        if self.is_dw:
            def iteration_latency(k):
                return self.load_latency + self.weight_offset_latency(k) + self.matrixvec_latency(k) + self.update_idx_latency +\
                       self.normquant_latency(k) + self.streamout_latency
        else:
            def iteration_latency(k):
                return n_in * (self.load_latency + self.weight_offset_latency(None) + self.matrixvec_latency(k) + self.update_idx_latency) +\
                       self.normquant_latency(k) + self.streamout_latency

        total_latency = n_spatial * (n_out_body * iteration_latency(k_out_body) + (iteration_latency(k_out_rem) if k_out_rem != 0 else 0))

        if self.is_dw:
            total_weight_offset_latency = n_spatial * (n_out_body * self.weight_offset_latency(k_out_body) + (self.weight_offset_latency(k_out_rem) if k_out_rem != 0 else 0))
            total_matrixvec_latency     = n_spatial * (n_out_body * self.matrixvec_latency(k_out_body)     + (self.matrixvec_latency(k_out_rem) if k_out_rem != 0 else 0))
            total_load_latency          = n_spatial * (n_out_body + (1 if k_out_rem != 0 else 0)) * self.load_latency
            total_update_idx_latency    = n_spatial * (n_out_body + (1 if k_out_rem != 0 else 0)) * self.update_idx_latency

            total_normquant_latency = n_spatial * (n_out_body * self.normquant_latency(k_out_body) + (self.normquant_latency(k_out_rem) if k_out_rem != 0 else 0))
            total_streamout_latency = n_spatial * (n_out_body + (1 if k_out_rem != 0 else 0)) * self.streamout_latency
        else:
            total_weight_offset_latency = n_spatial * (n_out_body * n_in * self.weight_offset_latency(k_out_body) + ((n_in * self.weight_offset_latency(k_out_rem)) if k_out_rem != 0 else 0))
            total_matrixvec_latency     = n_spatial * (n_out_body * n_in * self.matrixvec_latency(k_out_body)     + ((n_in * self.matrixvec_latency(k_out_rem)) if k_out_rem != 0 else 0))
            total_load_latency          = n_spatial * (n_out_body + (1 if k_out_rem != 0 else 0)) * n_in * self.load_latency
            total_update_idx_latency    = n_spatial * (n_out_body + (1 if k_out_rem != 0 else 0)) * n_in * self.update_idx_latency

            total_normquant_latency = n_spatial * (n_out_body * self.normquant_latency(k_out_body) + (self.normquant_latency(k_out_rem) if k_out_rem != 0 else 0))
            total_streamout_latency = n_spatial * (n_out_body + (1 if k_out_rem != 0 else 0)) * self.streamout_latency

        total_component_wise_latency = total_weight_offset_latency + total_matrixvec_latency + total_load_latency + total_update_idx_latency + total_normquant_latency + total_streamout_latency

        # assert total_latency == total_component_wise_latency, f"total latencies don't match: {total_latency} vs. {total_component_wise_latency}"

        """
        print(f'Latency breakdown:\n'
              f'  - load: {total_load_latency} ({total_load_latency / total_latency:.2%})\n'
              f'  - weight offset: {total_weight_offset_latency} ({total_weight_offset_latency / total_latency:.2%})\n'
              f'  - matrixvec: {total_matrixvec_latency} ({total_matrixvec_latency / total_latency:.2%})\n'
              f'  - update idx: {total_update_idx_latency} ({total_update_idx_latency / total_latency:.2%})\n'
              f'  - normquant: {total_normquant_latency} ({total_normquant_latency / total_latency:.2%})\n'
              f'  - streamout: {total_streamout_latency} ({total_streamout_latency / total_latency:.2%})\n'
              f'TOTAL: {total_latency}')
        """

        return total_latency

    @property
    def max_ops(self):
        Ho, Wo, Ko, Ki = self.layer
        Ho_max_util = DivAndCeilSTE.apply(Ho, self.OUTPUT_BUFFER_SHAPE[0]) * self.OUTPUT_BUFFER_SHAPE[0]
        Wo_max_util = DivAndCeilSTE.apply(Wo, self.OUTPUT_BUFFER_SHAPE[1]) * self.OUTPUT_BUFFER_SHAPE[1]
        Ki_max_util = DivAndCeilSTE.apply(Ki, self.INPUT_BUFFER_SHAPE[2]) * self.INPUT_BUFFER_SHAPE[2]

        return prod(self.kernel_shape) * Ki_max_util * Ho_max_util * Wo_max_util * Ko if self.is_3x3 or self.is_1x1 \
            else prod(self.kernel_shape) * Ki * Ho * Wo

    @property
    def utilization(self):
        return self.ops / self.max_ops

    @property
    def ops(self):
        Ho, Wo, Ko, Ki = self.layer
        if self.is_3x3 or self.is_1x1:
            return prod(self.kernel_shape) * Ki * Ho * Wo * Ko
        else:
            return prod(self.kernel_shape) * Ki * Ho * Wo

    @property
    def perf(self):
        return self.ops / self.latency

    @property
    def max_perf(self):
        ops =  prod(self.kernel_shape) * self.INPUT_BUFFER_SHAPE[2] * self.OUTPUT_BUFFER_SHAPE[0] *\
               self.OUTPUT_BUFFER_SHAPE[1] * (1 if self.is_dw else self.OUTPUT_BUFFER_SHAPE[2])
        k = self.INPUT_BUFFER_SHAPE[2] if self.is_dw else self.OUTPUT_BUFFER_SHAPE[2]
        latency = self.load_latency + self.weight_offset_latency(k) + self.matrixvec_latency(k) +\
                  self.update_idx_latency + (self.normquant_latency(k) + self.streamout_latency if self.is_dw else 0)
        return ops/latency

    def tiled_layer_latency(self, layer_shape_in, layer_shape_out, tile_shape_out):
        body_h_count = FloorSTE.apply(layer_shape_out[0], tile_shape_out[0])
        body_w_count = FloorSTE.apply(layer_shape_out[1], tile_shape_out[1])
        body_k_count = FloorSTE.apply(layer_shape_out[2], tile_shape_out[2])
        rem_h = ModuloSTE.apply(layer_shape_out[0], tile_shape_out[0])
        rem_w = ModuloSTE.apply(layer_shape_out[1], tile_shape_out[1])
        rem_k = ModuloSTE.apply(layer_shape_out[2], tile_shape_out[2])
        layers = [
            (tile_shape_out[0], tile_shape_out[1], tile_shape_out[2], layer_shape_in[2]),
            (tile_shape_out[0], tile_shape_out[1], rem_k, layer_shape_in[2]),
            (tile_shape_out[0], rem_w, tile_shape_out[2], layer_shape_in[2]),
            (tile_shape_out[0], rem_w, rem_k, layer_shape_in[2]),
            (rem_h, tile_shape_out[1], tile_shape_out[2], layer_shape_in[2]),
            (rem_h, tile_shape_out[1], rem_k, layer_shape_in[2]),
            (rem_h, rem_w, tile_shape_out[2], layer_shape_in[2]),
            (rem_h, rem_w, rem_k, layer_shape_in[2])
        ]
        n_tiles = [
            body_h_count * body_w_count * body_k_count,
            body_h_count * body_w_count * (1 if rem_k > 0 else 0),
            body_h_count * (1 if rem_w > 0 else 0) * body_k_count,
            body_h_count * (1 if rem_w > 0 else 0) * (1 if rem_k > 0 else 0),
            (1 if rem_h > 0 else 0) * body_w_count * body_k_count,
            (1 if rem_h > 0 else 0) * body_w_count * (1 if rem_k > 0 else 0),
            (1 if rem_h > 0 else 0) * (1 if rem_w > 0 else 0) * body_k_count,
            (1 if rem_h > 0 else 0) * (1 if rem_w > 0 else 0) * (1 if rem_k > 0 else 0)
        ]

        latency = 0
        ops = 0
        max_ops = 0
        for layer, n in zip(layers, n_tiles):
            self.set_layer(layer)
            latency += n * self.latency
            ops += n * self.ops
            max_ops += n * self.max_ops

        return latency, ops, max_ops

    @property
    def layer_shape_in(self):
        return (self.layer[0] + self.kernel_shape[0] - 1, self.layer[1] + self.kernel_shape[1] - 1, self.layer[3])

    def dma_latency(self, dma_stall=8, bandwidth=4):
        h_out, w_out, k_out, _ = self.layer
        h_in, w_in, k_in = self.layer_shape_in
        mem = h_in * w_in * k_in + h_out * w_out * k_out + self.kernel_shape[0] * self.kernel_shape[1] * k_out * k_in
        return (mem / bandwidth) * dma_stall


ne16Model3x3 = Ne16PerfModel('conv', (3, 3), depthwise=False)
ne16Model1x1 = Ne16PerfModel('conv', (1, 1), depthwise=False)
ne16Model3x3dw = Ne16PerfModel('conv', (3, 3), depthwise=True)


if __name__ == "__main__":
    # layer = (Ho, Wo, Ko, Ki)
    layer = (25, 5, 64, 64)
    WEIGHTS_BITWIDTH = 8
    ks = (1, 1)
    depth = False

    total_latency, total_ops, total_MACs_cycle = Ne16PerfModel_generalized('conv', (ks[0], ks[1]), depthwise=depth, WEIGHTS_BITWIDTH = WEIGHTS_BITWIDTH, layer = layer)
    print(f'Model performance for {ks[0]}x{ks[1]} {"depthwise" if depth else ""} convolution '
          f'with layer {layer}:')
    print(f'  - Latency: {total_latency} cycles\n'
          f'  - Operations: {total_ops} MACs\n'
          f'  - Performance: {total_MACs_cycle:.2f} MAC/cycle\n')

    # ne16 = Ne16PerfModel('conv', (ks[0], ks[1]), depthwise=depth, WEIGHTS_BITWIDTH = WEIGHTS_BITWIDTH)
    # ne16.set_layer(layer)
    # print(f'Model performance for {ks[0]}x{ks[1]} {"depthwise" if depth else ""} convolution '
    #       f'with layer {layer}:')
    # print(f'  - Latency: {ne16.latency} cycles\n'
    #       f'  - Operations: {ne16.ops} MACs\n'
    #       f'  - Performance: {ne16.perf:.2f} MAC/cycle out of {ne16.max_perf:.2f} MAC/cycle\n'
    #       f'  - Peak-Performance-Percentage: {ne16.perf / ne16.max_perf:.2%}\n'
    #       f'  - Utilization: {ne16.utilization:.2%}')
