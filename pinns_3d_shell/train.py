from typing import Dict, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

import pinnstorch


# -----------------------------
# 1) Utils géométrie : sampling
# -----------------------------
def sample_sphere_surface(n: int, radius: float, seed: int = 0):
    """
    Uniform-ish sampling on sphere surface (radius fixed).
    Returns x,y,z shape (n,1)
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    v = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)

    theta = 2.0 * np.pi * u
    phi = np.arccos(2.0 * v - 1.0)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)


def sample_shell_interior(n: int, r_in: float, r_out: float, seed: int = 0):
    """
    Sample points uniformly-ish in spherical shell volume by:
    - sampling direction on sphere
    - sampling radius with r^3 uniformity for volume
    Returns x,y,z shape (n,1)
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    v = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)

    theta = 2.0 * np.pi * u
    phi = np.arccos(2.0 * v - 1.0)

    # volume-uniform radius: r = (r_in^3 + w*(r_out^3-r_in^3))^(1/3)
    w = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    r = (r_in**3 + w * (r_out**3 - r_in**3)) ** (1.0 / 3.0)

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)


# ---------------------------------------
# 2) Solution analytique (manufactured)
# ---------------------------------------
def u_true_np(x, y, z, t, r_in: float, r_out: float):
    """
    u_true(x,y,z,t) = exp(-t) * sin(k*(r - r_in))
    with k = pi/(r_out-r_in)
    This ensures u=0 on r=r_in and r=r_out (Dirichlet).
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    k = np.pi / (r_out - r_in)
    return np.exp(-t) * np.sin(k * (r - r_in))


def forcing_np(x, y, z, t, r_in: float, r_out: float, alpha: float):
    """
    We solve manufactured PDE:
      u_t - alpha * Laplacian(u) - f(x,y,z,t) = 0
    with u = exp(-t) sin(k(r-r_in)).
    For radial u(r), Laplacian in 3D:
      Δu = u_rr + (2/r) u_r
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    k = np.pi / (r_out - r_in)

    # u(r,t)
    u = np.exp(-t) * np.sin(k * (r - r_in))

    # u_t
    u_t = -u

    # u_r, u_rr
    u_r = np.exp(-t) * k * np.cos(k * (r - r_in))
    u_rr = -np.exp(-t) * (k**2) * np.sin(k * (r - r_in))

    # Laplacian radial (avoid r=0; here r>=r_in>0)
    lap = u_rr + (2.0 / r) * u_r

    f = u_t - alpha * lap
    return f


# ---------------------------------------
# 3) read_data_fn : PointCloudData
# ---------------------------------------
def read_data_fn(root_path):
    """
    Create a PointCloudData for a 3D spherical shell:
    - interior points (for PDE)
    - boundary points on inner and outer spheres (for BC)
    time points: 1D grid

    PointCloudData in pinnstorch expects:
      spatial=[x,y,z] each shape (N_space,1)
      time=[t] shape (N_time,1)
      solution: dict var -> array (N_space, N_time)
    """

    # Domain parameters
    r_in, r_out = 0.35, 0.9
    t0, t1 = 0.0, 1.0

    # Points counts
    n_interior = 60_000
    n_b_inner = 10_000
    n_b_outer = 10_000

    # Time discretization
    n_time = 200
    t = np.linspace(t0, t1, n_time).reshape(-1, 1).astype(np.float32)

    # Interior
    x_i, y_i, z_i = sample_shell_interior(n_interior, r_in, r_out, seed=123)

    # Boundary (exact radii)
    x_bi, y_bi, z_bi = sample_sphere_surface(n_b_inner, r_in, seed=456)
    x_bo, y_bo, z_bo = sample_sphere_surface(n_b_outer, r_out, seed=789)

    # Combine spatial points
    x = np.concatenate([x_i, x_bi, x_bo], axis=0).astype(np.float32)
    y = np.concatenate([y_i, y_bi, y_bo], axis=0).astype(np.float32)
    z = np.concatenate([z_i, z_bi, z_bo], axis=0).astype(np.float32)

    # Build solution grid u_true(x_space, t_time)
    # u_true expects broadcasting: (N_space,1) with (1,N_time)
    X = x  # (N,1)
    Y = y
    Z = z
    T = t.T  # (1, Nt)

    u = u_true_np(X, Y, Z, T, r_in=r_in, r_out=r_out).astype(np.float32)  # (N,Nt)

    # Return in PointCloudData
    return pinnstorch.data.PointCloudData(
        spatial=[x, y, z],
        time=[t],
        solution={"u": u},
    )


# ---------------------------------------
# 4) pde_fn : résidu PDE + résidu BC
# ---------------------------------------
def pde_fn(
    outputs: Dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
    extra_variables: Dict[str, torch.Tensor],
):
    """
    Manufactured 3D heat equation residual:
      f_u = u_t - alpha*(u_xx+u_yy+u_zz) - forcing(x,y,z,t)

    Boundary condition (Dirichlet u=0 on r=r_in and r=r_out) enforced via residual:
      f_bc = u * mask_boundary

    This avoids relying on automatic boundary detection (not possible with PointCloud).
    """

    alpha = extra_variables["alpha"]

    # Gradients first order
    u_x, u_y, u_z, u_t = pinnstorch.utils.gradient(outputs["u"], [x, y, z, t])

    # Second derivatives
    u_xx = pinnstorch.utils.gradient(u_x, x)[0]
    u_yy = pinnstorch.utils.gradient(u_y, y)[0]
    u_zz = pinnstorch.utils.gradient(u_z, z)[0]

    lap_u = u_xx + u_yy + u_zz

    # Compute forcing analytically (torch)
    # Parameters must match read_data_fn
    r_in = torch.tensor(0.35, device=x.device, dtype=x.dtype)
    r_out = torch.tensor(0.9, device=x.device, dtype=x.dtype)
    k = torch.tensor(np.pi / (0.9 - 0.35), device=x.device, dtype=x.dtype)

    r = torch.sqrt(x**2 + y**2 + z**2)

    # u_true and forcing
    u_true = torch.exp(-t) * torch.sin(k * (r - r_in))

    u_t_true = -u_true
    u_r_true = torch.exp(-t) * k * torch.cos(k * (r - r_in))
    u_rr_true = -torch.exp(-t) * (k**2) * torch.sin(k * (r - r_in))
    lap_true = u_rr_true + (2.0 / r) * u_r_true  # r>=r_in>0
    forcing = u_t_true - alpha * lap_true

    # PDE residual
    outputs["f_u"] = u_t - alpha * lap_u - forcing

    # Boundary residual (mask)
    # Since boundary points were generated with exact radii, a small tol works.
    tol = torch.tensor(1e-4, device=x.device, dtype=x.dtype)
    mask_b = ((r - r_in).abs() < tol) | ((r - r_out).abs() < tol)
    outputs["f_bc"] = outputs["u"] * mask_b.to(outputs["u"].dtype)

    return outputs


# ---------------------------------------
# 5) Main training
# ---------------------------------------
@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    pinnstorch.utils.extras(cfg)

    metric_dict, _ = pinnstorch.train(
        cfg,
        read_data_fn=read_data_fn,
        pde_fn=pde_fn,
        output_fn=None,
    )

    metric_value = pinnstorch.utils.get_metric_value(
        metric_dict=metric_dict,
        metric_names=cfg.get("optimized_metric"),
    )
    return metric_value


if __name__ == "__main__":
    main()
