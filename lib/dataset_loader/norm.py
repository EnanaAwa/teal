import torch
import torch_scatter

def mlu_primal_norm(
    p2e: torch.Tensor,
    tms: torch.Tensor,
    caps: torch.Tensor,
):
    p2e_scaled = tms * (p2e * (1/caps[None,:]))

    p2e_norm = induced_1_2_norm(p2e_scaled)
    e2p_norm = induced_1_2_norm(p2e_scaled.t())

    l_norm = p2e_norm * e2p_norm
    return l_norm

def mlu_dual_norm(
    f2e: torch.Tensor,
    p2e: torch.Tensor,
    tms: torch.Tensor,
    caps: torch.Tensor,
    k: int
):
    f2e_sqrt = torch.sqrt(f2e)
    flows = tms.reshape(-1, k, 1)[:,0,:]

    #f2e_scaled = (f2e_sqrt * (1/caps[None,:]))
    f2e_scaled = f2e_sqrt
    f2e_max = scatter_max_coo_tensor(f2e_scaled)
    f2e_max = torch.sqrt(flows)

    #p2e_scaled =  torch.sqrt(tms) * (p2e * (1/caps[None,:]))
    p2e_scaled =  torch.sqrt(tms) * p2e
    l_1_2_norm = induced_1_2_norm(p2e_scaled.t())
    #print(f"induced 1_2_norm for etp: {torch.sum(f2e_max) * l_1_2_norm}")

    return l_1_2_norm * torch.sum(f2e_max)

def induced_1_2_norm(
    coo_tensor: torch.Tensor,
    dim: int = 1
):
    coo_tensor = coo_tensor.coalesce()
    values = coo_tensor.values()
    values = (values ** 2)

    sum_values = torch_scatter.scatter_sum(
        values,
        index=coo_tensor.indices()[dim,:],
        dim=0,
        dim_size=coo_tensor.size(dim)
    )
    l2_norms = torch.sqrt(sum_values)
    return torch.max(l2_norms)


def scatter_max_coo_tensor(
    coo_tensor: torch.Tensor
):
    coo_tensor = coo_tensor.coalesce()
    indices = coo_tensor.indices()
    values = coo_tensor.values()
    values = values * (values < 15)

    row_idx = indices[0,:]
    mx_values, _ = torch_scatter.scatter_max(
        values, 
        row_idx,
        dim=0,
        dim_size=coo_tensor.size(0)
    )
    return mx_values