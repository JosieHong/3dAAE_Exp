'''
Date: 2022-04-27 16:59:41
LastEditors: yuhhong
LastEditTime: 2022-04-27 22:33:36
'''
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment



# Borrow from https://github.com/ThibaultGROUEIX/AtlasNet
def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]

# Import CUDA version of approximate EMD, from https://github.com/zekunhao1995/pcgan-pytorch/
try:
    from .StructuralLosses.nn_distance import nn_distance
    def distChamferCUDA(x, y):
        return nn_distance(x, y)
except:
    print("distChamferCUDA not available; fall back to slower version.")
    def distChamferCUDA(x, y):
        return distChamfer(x, y)


def emd_approx(x, y):
    bs, npts, mpts, dim = x.size(0), x.size(1), y.size(1), x.size(2)
    assert npts == mpts, "EMD only works if two point clouds are equal size"
    dim = x.shape[-1]
    x = x.reshape(bs, npts, 1, dim)
    y = y.reshape(bs, 1, mpts, dim)
    dist = (x - y).norm(dim=-1, keepdim=False)  # (bs, npts, mpts)

    emd_lst = []
    dist_np = dist.cpu().detach().numpy()
    for i in range(bs):
        d_i = dist_np[i]
        r_idx, c_idx = linear_sum_assignment(d_i)
        emd_i = d_i[r_idx, c_idx].mean()
        emd_lst.append(emd_i)
    emd = np.stack(emd_lst).reshape(-1)
    emd_torch = torch.from_numpy(emd).to(x)
    return emd_torch


try:
    from .StructuralLosses.match_cost import match_cost
    def emd_approx_cuda(sample, ref):
        B, N, N_ref = sample.size(0), sample.size(1), ref.size(1)
        assert N == N_ref, "Not sure what would EMD do in this case"
        emd = match_cost(sample, ref)  # (B,)
        emd_norm = emd / float(N)  # (B,)
        return emd_norm
except:
    print("emd_approx_cuda not available. Fall back to slower version.")
    def emd_approx_cuda(sample, ref):
        return emd_approx(sample, ref)


def mmd_between_point_cloud_sets(sample_pcs, ref_pcs, batch_size, 
                                accelerated_cd=False, reduced=True,
                                accelerated_emd=False): 
    """Computes the MMD-CD and MMD-EMD between two sets of point-clouds
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
    """
    
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    cd_lst = []
    emd_lst = []
    iterator = range(0, N_sample, batch_size)

    for b_start in iterator:
        b_end = min(N_sample, b_start + batch_size)
        sample_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]

        if accelerated_cd:
            dl, dr = distChamferCUDA(sample_batch, ref_batch)
        else:
            dl, dr = distChamfer(sample_batch, ref_batch)
        cd_lst.append(dl.mean(dim=1) + dr.mean(dim=1))

        if accelerated_emd:
            emd_batch = emd_approx_cuda(sample_batch, ref_batch)
        else:
            emd_batch = emd_approx(sample_batch, ref_batch)
        emd_lst.append(emd_batch)

    if reduced:
        cd = torch.cat(cd_lst).mean()
        emd = torch.cat(emd_lst).mean()
    else:
        cd = torch.cat(cd_lst)
        emd = torch.cat(emd_lst)

    # results = {
    #     'MMD-CD': cd,
    #     'MMD-EMD': emd,
    # }
    return cd, emd
