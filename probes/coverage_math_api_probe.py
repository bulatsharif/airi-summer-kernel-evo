import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


ELEMENTS = 128
FUNCTIONS = 12


@cute.kernel
def math_api_kernel(input_tensor: cute.Tensor, output_tensor: cute.Tensor):
    thread_idx, _, _ = cute.arch.thread_idx()
    value = input_tensor[thread_idx].to(cutlass.Float32)
    positive = cutlass.Float32(value + 2.0)
    output_tensor[thread_idx, 0] = cute.exp(value)
    output_tensor[thread_idx, 1] = cute.log(positive)
    output_tensor[thread_idx, 2] = cute.tanh(value)
    output_tensor[thread_idx, 3] = cute.erf(value)
    output_tensor[thread_idx, 4] = cute.sqrt(positive)
    output_tensor[thread_idx, 5] = cute.rsqrt(positive)
    output_tensor[thread_idx, 6] = cute.sin(value)
    output_tensor[thread_idx, 7] = cute.cos(value)
    output_tensor[thread_idx, 8] = cute.exp2(value)
    output_tensor[thread_idx, 9] = cute.log2(positive)
    output_tensor[thread_idx, 10] = cute.floor(value)
    output_tensor[thread_idx, 11] = -cute.floor(-value)


@cute.jit
def run_math_api(input_tensor: cute.Tensor, output_tensor: cute.Tensor):
    math_api_kernel(input_tensor, output_tensor).launch(
        grid=(1, 1, 1),
        block=(ELEMENTS, 1, 1),
    )


def main():
    source = torch.linspace(-1.0, 1.0, ELEMENTS, device="cuda")
    output = torch.empty(
        (ELEMENTS, FUNCTIONS), device="cuda", dtype=torch.float32
    )
    source_tensor = from_dlpack(source)
    output_tensor = from_dlpack(output)
    compiled = cute.compile(run_math_api, source_tensor, output_tensor)
    compiled(source_tensor, output_tensor)
    torch.cuda.synchronize()
    reference = torch.stack(
        (
            torch.exp(source),
            torch.log(source + 2.0),
            torch.tanh(source),
            torch.erf(source),
            torch.sqrt(source + 2.0),
            torch.rsqrt(source + 2.0),
            torch.sin(source),
            torch.cos(source),
            torch.exp2(source),
            torch.log2(source + 2.0),
            torch.floor(source),
            torch.ceil(source),
        ),
        dim=1,
    )
    error = (output - reference).abs().max().item()
    if error > 1.0e-5:
        raise RuntimeError(f"math api validation failed: max_abs={error}")
    print(f"math_api_max_abs={error:.8f} PASS")


main()
