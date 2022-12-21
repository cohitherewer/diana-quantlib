import torch
from torch import nn
import time
import numpy as np

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def measure_time(func, *args, **kwargs):
    torch.cuda.synchronize(device)
    t0 = time.time()
    _ = func(*args, **kwargs)
    torch.cuda.synchronize(device)
    return time.time() - t0


def measure_performance(
    func, benchmark_loop_count, *args, **kwargs
):  # returns
    # dry runs
    times = []
    for i in range(5):
        measure_time(func, *args, **kwargs)
    for i in range(benchmark_loop_count):
        times.append(measure_time(func, *args, **kwargs))
    return np.mean(np.asarray(times) * 1e3), np.std(np.asarray(times) * 1e3)


def measure_memory_usage(func, *args):
    torch.cuda.empty_cache()
    a = torch.cuda.memory_allocated(device)
    valid_out = func(*args)
    b = torch.cuda.memory_allocated(device)
    print("memory usage: ", b - a)
    return b - a


def conv_method(
    feature_map: torch.Tensor, kernel: torch.Tensor, stride, padding
):

    negative_x = torch.minimum(
        feature_map, torch.Tensor([0]).to(feature_map.device)
    )
    positive_x = torch.maximum(
        feature_map, torch.Tensor([0]).to(feature_map.device)
    )
    negative_kernel = torch.minimum(
        kernel, torch.Tensor([0]).to(feature_map.device)
    )
    positive_kernel = torch.maximum(
        kernel, torch.Tensor([0]).to(feature_map.device)
    )
    clipping = [-200, 200]
    # positive products are summed together, negative products are summed together

    conv_out_neg = torch.clamp(
        torch.nn.functional.conv2d(
            negative_x, positive_kernel, stride=stride, padding=padding
        )
        + torch.nn.functional.conv2d(
            positive_x, negative_kernel, stride=stride, padding=padding
        ),
        min=clipping[0],
        max=clipping[1],
    )  # positive fmap * neg kernal + pos kernel * neg featuremap
    conv_out_pos = torch.clamp(
        torch.nn.functional.conv2d(
            positive_x, positive_kernel, stride=stride, padding=padding
        )
        + torch.nn.functional.conv2d(
            negative_x, negative_kernel, stride=stride, padding=padding
        ),
        min=clipping[0],
        max=clipping[1],
    )
    return conv_out_neg + conv_out_pos


def unfold_mat_mult(
    feature_map: torch.Tensor, kernel: torch.Tensor, stride, padding
):
    out_H = int(
        ((feature_map.size(2) - kernel_size + 2 * padding) / stride) + 1
    )
    out_W = int(
        ((feature_map.size(3) - kernel_size + 2 * padding) / stride) + 1
    )
    clipping = [-200, 200]
    inp_unf = nn.functional.unfold(
        feature_map, kernel_size, stride=stride, padding=padding
    ).transpose(1, 2)
    inp_unf_pos = torch.maximum(
        inp_unf, torch.Tensor([0]).to(feature_map.device)
    )
    inp_unf_neg = torch.minimum(
        inp_unf, torch.Tensor([0]).to(feature_map.device)
    )
    negative_kernel = (
        torch.minimum(kernel, torch.Tensor([0]).to(feature_map.device))
        .view(kernel.size(0), -1)
        .t()
    )
    positive_kernel = (
        torch.maximum(kernel, torch.Tensor([0]).to(feature_map.device))
        .view(kernel.size(0), -1)
        .t()
    )
    out_unf_neg = torch.clamp(
        inp_unf_pos.matmul(negative_kernel)
        + inp_unf_neg.matmul(positive_kernel),
        min=clipping[0],
        max=clipping[1],
    )
    out_unf_pos = torch.clamp(
        inp_unf_pos.matmul(positive_kernel)
        + inp_unf_neg.matmul(negative_kernel),
        min=clipping[0],
        max=clipping[1],
    )
    out_unf = (out_unf_pos + out_unf_neg).transpose(
        1, 2
    )  # matmul(kernel.view(kernel.size(0), -1).t()).transpose(1, 2)
    return torch.nn.functional.fold(out_unf, (out_H, out_W), (1, 1))


def unfold_sparse_mat_mult(
    feature_map: torch.Tensor, kernel: torch.Tensor, stride, padding
):
    out_H = int(
        ((feature_map.size(2) - kernel_size + 2 * padding) / stride) + 1
    )
    out_W = int(
        ((feature_map.size(3) - kernel_size + 2 * padding) / stride) + 1
    )
    clipping = [-200, 200]
    inp_unf = nn.functional.unfold(
        feature_map, kernel_size, stride=stride, padding=padding
    ).transpose(1, 2)
    inp_unf_pos = torch.maximum(
        inp_unf, torch.Tensor([0]).to(feature_map.device)
    ).to_sparse()
    print(inp_unf_pos.size())
    inp_unf_neg = torch.minimum(
        inp_unf, torch.Tensor([0]).to(feature_map.device)
    ).to_sparse()
    negative_kernel = (
        torch.minimum(kernel, torch.Tensor([0]).to(feature_map.device))
        .view(kernel.size(0), -1)
        .t()
        .to_sparse()
    )
    print(negative_kernel.size())
    positive_kernel = (
        torch.maximum(kernel, torch.Tensor([0]).to(feature_map.device))
        .view(kernel.size(0), -1)
        .t()
        .to_sparse()
    )
    for i in range(inp_unf.size(0)):
        out_unf_neg = torch.clamp(
            (
                torch.sparse.mm(inp_unf_pos[i], negative_kernel)
                + torch.sparse.mm(inp_unf_neg[i], positive_kernel)
            ).to_dense(),
            min=clipping[0],
            max=clipping[1],
        )
        out_unf_pos = torch.clamp(
            (
                torch.sparse.mm(inp_unf_pos[i], positive_kernel)
                + torch.sparse.mm(inp_unf_neg[i], negative_kernel)
            ).to_dense(),
            min=clipping[0],
            max=clipping[1],
        )
    out_unf = (out_unf_pos + out_unf_neg).transpose(
        1, 2
    )  # matmul(kernel.view(kernel.size(0), -1).t()).transpose(1, 2)
    return torch.nn.functional.fold(out_unf, (out_H, out_W), (1, 1))


def greater_than(feature_map):
    return torch.maximum(feature_map, torch.Tensor([0]).to(feature_map.device))


feature_map_batch_size = 256
feature_map_H = 4
feature_map_W = 4
input_channels = 3

feature_map = torch.floor(
    torch.rand(
        feature_map_batch_size, input_channels, feature_map_H, feature_map_W
    )
    * 10
).to(device)


kernel_size = 2
kernel_count = 3
kernel = torch.floor(
    (torch.rand(kernel_count, input_channels, kernel_size, kernel_size) - 0.5)
    * 4
).to(device)
kernel.requires_grad = True
padding = 0
stride = 1

print("#################### Using standard conv2d method ####################")

# Measuring memory
torch.cuda.empty_cache()
a = torch.cuda.memory_allocated(device)
valid_out = nn.functional.conv2d(
    feature_map, kernel, None, stride, padding=padding
)
b = torch.cuda.memory_allocated(device)
# print(valid_out)
valid_out = None
torch.cuda.empty_cache()

print("memory usage: ", b - a)
mean, std = measure_performance(
    nn.functional.conv2d, 100, feature_map, kernel, None, stride, padding
)
print(f" Using the regular conv2d the mean time is {mean} with std {std} ")
# Conv way


##### TWO LINE CONVOLUTION #####
## Convolution way ###


print("#################### Using conv2d method ####################")
a = torch.cuda.memory_allocated(device)
padded_out = conv_method(feature_map, kernel, stride, padding)
b = torch.cuda.memory_allocated(device)
print("memory usage: ", b - a)
padded_out = None
torch.cuda.empty_cache()
# print(padded_out)
# assert( torch.all (valid_out.eq(padded_out)))
mean, std = measure_performance(
    conv_method,
    100,
    feature_map=feature_map,
    kernel=kernel,
    stride=stride,
    padding=padding,
)
print(f" Using the split- conv2d the mean time is {mean} with std {std} ")
# Measuring memory

## Convolution way end ##
print("#################### Using unfold method ####################")
a = torch.cuda.memory_allocated(device)
unfolded_out = unfold_mat_mult(feature_map, kernel, stride, padding)
b = torch.cuda.memory_allocated(device)
print("memory usage: ", b - a)
unfolder_out = None
torch.cuda.empty_cache()

# print(unfolded_out)
mean, std = measure_performance(
    conv_method,
    100,
    feature_map=feature_map,
    kernel=kernel,
    stride=stride,
    padding=padding,
)
print(f" Using the unfold conv2d the mean time is {mean} with std {std} ")
# assert( torch.all (valid_out.eq(unfolded_out)))
# Measuring memory
mean, std = measure_performance(greater_than, 100, feature_map=feature_map)
print(f" Using th greater than method the mean time is {mean} with std {std} ")

# print("#################### Using unfold sparse matrices method ####################")
# padded_out = unfold_sparse_mat_mult(feature_map , kernel, stride , padding)
# print(padded_out)
# assert( torch.all (valid_out.eq(padded_out)))
# Measuring memory
