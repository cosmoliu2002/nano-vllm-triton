from functools import reduce

import torch
import triton
import triton.language as tl


@triton.jit
def cat_kernel_v1(
    A_ptr, B_ptr, C_ptr,
    unit_offset_A, unit_offset_B, unit_offset_C,
    BLOCK_SIZE: tl.constexpr
):
    block_idx = tl.program_id(0)
    
    for offset in range(0, unit_offset_A, BLOCK_SIZE):
        A_offset_idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = A_offset_idx < unit_offset_A
        val_from_A = tl.load(A_ptr + unit_offset_A*block_idx + A_offset_idx , mask=mask)
        val_from_A = tl.cos(val_from_A.to(tl.float32))
        tl.store(C_ptr + block_idx*unit_offset_C + A_offset_idx, val_from_A, mask=mask)

    for offset in range(0, unit_offset_B, BLOCK_SIZE):
        B_offset_idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = B_offset_idx < unit_offset_B
        val_from_B = tl.load(B_ptr + unit_offset_B*block_idx + B_offset_idx , mask=mask)
        val_from_B = tl.sin(val_from_B.to(tl.float32))
        tl.store(C_ptr + block_idx*unit_offset_C + unit_offset_A + B_offset_idx, val_from_B, mask=mask)

@triton.jit
def cat_kernel_v2_blocks(
    A_ptr, B_ptr, C_ptr,
    A_section_numel, B_section_numel, C_section_numel,
    block_num_process_every_C_section,
    BLOCK_SIZE: tl.constexpr
):
    block_idx = tl.program_id(0)
    block_process_numel = tl.cdiv(C_section_numel, block_num_process_every_C_section) 
    section_offset = (block_idx//block_num_process_every_C_section)
    start_offset = section_offset * C_section_numel
    c_section_offset = (block_idx % block_num_process_every_C_section) * block_process_numel

    if c_section_offset < A_section_numel:
        for offset in range(0, block_process_numel, BLOCK_SIZE):
            offset_idx = c_section_offset + offset + tl.arange(0, BLOCK_SIZE)
            mask = (offset_idx < A_section_numel)
            val_from_A = tl.load(A_ptr + section_offset*A_section_numel + offset_idx, mask=mask)
            val_from_A = tl.cos(val_from_A.to(tl.float32))
            tl.store(C_ptr + start_offset + offset_idx, val_from_A, mask=mask)

    if c_section_offset >= A_section_numel or (c_section_offset + block_process_numel - 1) >= A_section_numel:
        for offset in range(0, block_process_numel, BLOCK_SIZE):
            offset_idx = c_section_offset + offset + tl.arange(0, BLOCK_SIZE)
            mask = (offset_idx >= A_section_numel) & (offset_idx < C_section_numel)
            val_from_B = tl.load(B_ptr + section_offset*B_section_numel + (offset_idx - A_section_numel), mask=mask)
            val_from_B = tl.sin(val_from_B.to(tl.float32))
            tl.store(C_ptr + start_offset+offset_idx, val_from_B, mask=mask)

@triton.jit
def cat_kernel_v2_sections(
    A_ptr, B_ptr, C_ptr,
    A_section_numel, B_section_numel, C_section_numel,
    C_section_num_processed_by_every_block,
    section_num,
    BLOCK_SIZE: tl.constexpr
):
    block_idx = tl.program_id(0)
    
    for sub_section_index in range(C_section_num_processed_by_every_block):
        sub_section_offset = block_idx * C_section_num_processed_by_every_block + sub_section_index
        if sub_section_offset < section_num:
            C_section_start = C_ptr + sub_section_offset * C_section_numel
            A_section_start = A_ptr + sub_section_offset * A_section_numel
            B_section_start = B_ptr + sub_section_offset * B_section_numel
            
            for offset in range(0, A_section_numel, BLOCK_SIZE):
                offset_idx = offset + tl.arange(0, BLOCK_SIZE)
                mask = offset_idx < A_section_numel
                val_from_A = tl.load(A_section_start + offset_idx, mask=mask)
                val_from_A = tl.cos(val_from_A.to(tl.float32))
                tl.store(C_section_start + offset_idx, val_from_A, mask=mask)

            for offset in range(0, B_section_numel, BLOCK_SIZE):
                offset_idx = offset + tl.arange(0, BLOCK_SIZE)
                mask = offset_idx < B_section_numel
                val_from_B = tl.load(B_section_start + offset_idx, mask=mask)
                val_from_B = tl.sin(val_from_B.to(tl.float32))
                tl.store(C_section_start + A_section_numel + offset_idx, val_from_B, mask=mask)

@triton.jit
def cat_kernel_dim_0(
    A_ptr, B_ptr, C_ptr,
    A_numel, B_numel, block_process_numel,
    BLOCK_SIZE: tl.constexpr
):
    block_idx = tl.program_id(0)
    start_offset = block_idx * block_process_numel
    
    if start_offset < A_numel:
        for offset in range(0, block_process_numel, BLOCK_SIZE):
            offset_idx = start_offset + offset + tl.arange(0, BLOCK_SIZE)
            mask = (offset_idx < A_numel)
            val_from_A = tl.load(A_ptr + offset_idx, mask=mask)
            val_from_A = tl.cos(val_from_A.to(tl.float32))
            tl.store(C_ptr + offset_idx, val_from_A, mask=mask)
    
    if start_offset + block_process_numel > A_numel:
        for offset in range(0, block_process_numel, BLOCK_SIZE):
            offset_idx = start_offset + offset + tl.arange(0, BLOCK_SIZE)
            mask = (offset_idx >= A_numel) & (offset_idx < A_numel + B_numel)
            val_from_B = tl.load(B_ptr + offset_idx - A_numel, mask=mask)
            val_from_B = tl.sin(val_from_B.to(tl.float32))
            tl.store(C_ptr + offset_idx, val_from_B, mask=mask)

def triton_cat_cos_sin(A: torch.Tensor, B: torch.Tensor, dim: int):
    A = A.contiguous()
    B = B.contiguous()
    output_shape = list(A.shape)
    output_shape[dim] = A.shape[dim] + B.shape[dim]
    C = torch.empty(output_shape, device=A.device, dtype=A.dtype)
    
    if dim == 0:
        block_num = 1024
        elements_per_block = (C.numel() + block_num - 1) // block_num
        cat_kernel_dim_0[(block_num,)](  
            A, B, C,
            A.numel(), B.numel(), elements_per_block,
            BLOCK_SIZE=512
        )
        return C
    
    section_num = reduce(lambda x, y: x * y, output_shape[:dim])
    unit_offset_A = reduce(lambda x, y: x * y, A.shape[dim:])
    unit_offset_B = reduce(lambda x, y: x * y, B.shape[dim:])
    unit_offset_C = unit_offset_A + unit_offset_B
    
    if section_num < 128:
        block_num_process_every_C_section = min((unit_offset_C + 1023) // 1024, 1024)
        num_blocks = section_num * block_num_process_every_C_section
        cat_kernel_v2_blocks[(num_blocks,)](  
            A, B, C,
            unit_offset_A, unit_offset_B, unit_offset_C,
            block_num_process_every_C_section,
            BLOCK_SIZE=512
        )
    elif section_num <= 2048:
        cat_kernel_v1[(section_num,)](  
            A, B, C,
            unit_offset_A, unit_offset_B, unit_offset_C,
            BLOCK_SIZE=1024
        )
    else: 
        if unit_offset_C >= 4096:
            C_section_num_processed_by_every_block = min((section_num + 1023) // 1024, 2048)
        else:
            C_section_num_processed_by_every_block = 128
        num_blocks = (section_num + C_section_num_processed_by_every_block - 1) // C_section_num_processed_by_every_block
        cat_kernel_v2_sections[(num_blocks,)](
            A, B, C,
            unit_offset_A, unit_offset_B, unit_offset_C,
            C_section_num_processed_by_every_block, 
            section_num, 
            BLOCK_SIZE=512
        )
    
    return C