import torch
import triton
import triton.language as tl

"""Triton kernel for sparse nearest neighbor search in a batch sparse long input tensor.
This kernel performs a nearest neighbor search by implicitly broadcasting the input tensor and candidate item embedding tensor,
and computes the dot product similarity between the broadcasted tensors and candidate embedding tensor.
For more details refer to: https://arxiv.org/abs/2506.02267"""

@triton.jit
def sparse_nn(
    inputs_ptr,
    offsets_ptr,
    cand_ptr,
    output_ptr,
    num_elems_per_slice,
    S,
    D: tl.constexpr,  # noqa
    BLOCK_SIZE: tl.constexpr,  # noqa
):
    """Triton kernel that takes a batch sparse long input tensor and a candidate item embedding tensor, performs nn search by implicitly broadcasting bcast_value and bcast_offset to the shape of (B, S, D) and computes the cosine similarity between the broadcasted tensors and candidate embedding tensor.
    Since we don't have to explicitly broadcast request level sequence features and perform redundant nn searches this kernel is more efficient.

    The kernel is equivalent to the following torch implementation:

    def torch_nn_broadcast(inputs, offsets, cand_tensor):
        bcast_tensor =  torch.repeat_interleave(inputs, torch.diff(offsets), dim=0)
        torch_output = torch.bmm(bcast_tensor, cand_tensor.unsqueeze(-1)).squeeze(2)
        return torch_output

    Requirements:
        - The inputs tensor should be of shape (U, S, D), dtype torch.int8 and contiguous
        - The offsets tensor should be of shape (B+1, ), dtype torch.int32 and end with B
        - S*D == num_elems_per_slice, S and D should be divisible by 16

    Args:
        inputs_ptr (torch.Tensor): Pointer to the input tensor: (U, S, D)
        offsets_ptr (torch.Tensor): Pointer to the offsets tensor: (B+1, ), the last element of this should be B
        cand_ptr (torch.Tensor): Pointer to the cand_tensor tensor: (B, D)
        output_ptr (torch.Tensor): Pointer to the output tensor: (B, S)
        num_elems_per_slice (int): Number of elements in each slice of the input tensor. (N * D), dtype torch.int32
        S (int): Sequence length. dtype torch.int32
        D (tl.constexpr): Dimension of the input tensor
        BLOCK_SIZE (tl.constexpr): Block size for kernel

    """
    pid = tl.program_id(0)
    block_id = tl.program_id(1)
    positions = tl.arange(0, BLOCK_SIZE)

    # Each pid processes 1 input tensor, and each block processes 1 block of the input tensor
    read_positions = pid * num_elems_per_slice + block_id * BLOCK_SIZE + positions
    last_read_position = (pid + 1) * num_elems_per_slice
    input = tl.load(inputs_ptr + read_positions, mask=read_positions < last_read_position)  # noqa: A001
    input = tl.reshape(input, (BLOCK_SIZE // D, D)).to(tl.float32)  # noqa: A001

    prev = tl.load(offsets_ptr + pid)
    curr = tl.load(offsets_ptr + pid + 1)

    write_positions = prev * S + block_id * BLOCK_SIZE // D + tl.arange(0, BLOCK_SIZE // D)
    last_write_position = (prev + 1) * S
    for i in range(prev, curr):
        gsv5 = tl.load(cand_ptr + i * D + tl.arange(0, D))[None, :]
        scores = input * gsv5
        scores = tl.sum(scores, axis=1).to(tl.float16)
        tl.store(output_ptr + write_positions, scores, mask=write_positions < last_write_position)
        write_positions += S
        last_write_position += S

def main():
    # This is a placeholder for testing the kernel.
    # You would typically call this kernel from a higher-level function
    # that prepares the inputs and launches the kernel.
    U, S, D = 64, 16000, 32
    B = 256
    num_elems_per_slice = S * D
    BLOCK_SIZE = 4096
    inputs = torch.randint(-128, 127, (U, S, D), dtype=torch.int8, device="cuda")
    offsets = torch.arange(0, B + 1, B // U, device="cuda").to(torch.int32)
    # pad with value B to size B+1
    gsv5_cand = torch.randn(B, D, device="cuda").to(torch.float16)

    # Find torch_output
    def torch_nn_broadcast(inputs, offsets, gsv5_cand):
        bcast_tensor = torch.repeat_interleave(inputs, torch.diff(offsets), dim=0).to(torch.float16)
        torch_output = torch.bmm(bcast_tensor, gsv5_cand.unsqueeze(-1).to(torch.float16)).squeeze(2)
        return torch_output.to(torch.float16)
    
    def triton_sparse_nn(inputs, offsets, gsv5_cand):
        output = torch.empty(B, S, device="cuda", dtype=torch.float16)
        grid = inputs.shape[0], triton.cdiv(num_elems_per_slice, BLOCK_SIZE)
        sparse_nn[grid](
            inputs_ptr=inputs,
            offsets_ptr=offsets,
            cand_ptr=gsv5_cand,
            output_ptr=output,
            num_elems_per_slice=num_elems_per_slice,
            S=S,
            D=D,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output

    torch_output = torch_nn_broadcast(inputs, offsets, gsv5_cand)
    offsets_triton = torch.cat([offsets, torch.tensor([B] * (B + 1 - offsets.size(0)), device="cuda").to(torch.int32)])
    triton_output = triton_sparse_nn(inputs, offsets_triton, gsv5_cand)
    print("Device: ", torch.cuda.get_device_name())
    print("Outputs match:", torch.allclose(torch_output, triton_output, atol=1e-2, rtol=1e-2))

    print("Triton benchmark (ms): ", triton.testing.do_bench(lambda: triton_sparse_nn(inputs, offsets_triton, gsv5_cand), warmup=5, rep=10))
    print("Torch benchmark (ms): ", triton.testing.do_bench(lambda: torch_nn_broadcast(inputs, offsets, gsv5_cand), warmup=5, rep=10))
    
    """Expected output:
    Device:  NVIDIA L40S
    Outputs match Slow v/s Fast ?: True
    Triton benchmark (ms):  0.08302029967308044
    Torch benchmark (ms):  1.6435072422027588
    """
if __name__ == "__main__":
    main()