from typing import Dict, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

import pinnstorch


def read_data_fn(root_path):
    """
    Create a synthetic point cloud (x,y,t) for 2D Navier-Stokes PINN training.
    We return a dummy "solution" field because PointCloudData expects it, even
    if we do pure-residual training.
    """

    x_min, x_max = -1.0, 1.0
    y_min, y_max = -1.0, 1.0
    t_min, t_max = 0.0, 1.0

    N_space = 20_000
    N_time = 200

    # Spatial points: (N_space, 1)
    x = np.random.uniform(x_min, x_max, (N_space, 1)).astype(np.float32)
    y = np.random.uniform(y_min, y_max, (N_space, 1)).astype(np.float32)

    # Time points: (N_time, 1)
    t = np.linspace(t_min, t_max, N_time).reshape(-1, 1).astype(np.float32)

    # Dummy solution: shape must be (N_space, N_time)
    dummy = np.zeros((N_space, N_time), dtype=np.float32)

    solution = {
        "u": dummy,  # not used (pure residual), but required by PointCloudData
    }

    return pinnstorch.data.PointCloudData(
        spatial=[x, y],
        time=[t],
        solution=solution,
    )


def output_fn(
    outputs: Dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
):
    """
    Map NN outputs to physical variables using stream function psi:
      u = dpsi/dy
      v = -dpsi/dx
    This enforces incompressibility (div u = 0) by construction.
    """
    outputs["u"] = pinnstorch.utils.gradient(outputs["psi"], y)[0]
    outputs["v"] = -pinnstorch.utils.gradient(outputs["psi"], x)[0]
    return outputs


def pde_fn(
    outputs: Dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    extra_variables: Dict[str, torch.Tensor],
):
    """
    2D incompressible Navier–Stokes residuals (no forcing):

      u_t + l1*(u u_x + v u_y) + p_x - l2*(u_xx + u_yy) = 0
      v_t + l1*(u v_x + v v_y) + p_y - l2*(v_xx + v_yy) = 0

    where:
      l1 ~ convection coefficient (often 1)
      l2 ~ viscosity (nu)
    """

    # First derivatives
    u_x, u_y, u_t = pinnstorch.utils.gradient(outputs["u"], [x, y, t])
    v_x, v_y, v_t = pinnstorch.utils.gradient(outputs["v"], [x, y, t])

    # Second derivatives
    u_xx = pinnstorch.utils.gradient(u_x, x)[0]
    u_yy = pinnstorch.utils.gradient(u_y, y)[0]
    v_xx = pinnstorch.utils.gradient(v_x, x)[0]
    v_yy = pinnstorch.utils.gradient(v_y, y)[0]

    # Pressure gradients
    p_x, p_y = pinnstorch.utils.gradient(outputs["p"], [x, y])

    outputs["f_u"] = (
        u_t
        + extra_variables["l1"] * (outputs["u"] * u_x + outputs["v"] * u_y)
        + p_x
        - extra_variables["l2"] * (u_xx + u_yy)
    )

    outputs["f_v"] = (
        v_t
        + extra_variables["l1"] * (outputs["u"] * v_x + outputs["v"] * v_y)
        + p_y
        - extra_variables["l2"] * (v_xx + v_yy)
    )

    return outputs


@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # prints config, sets seed, etc.
    pinnstorch.utils.extras(cfg)

    metric_dict, _ = pinnstorch.train(
        cfg,
        read_data_fn=read_data_fn,  # ✅ passed to mesh via cfg.mesh.read_data_fn
        pde_fn=pde_fn,
        output_fn=output_fn,
    )

    metric_value = pinnstorch.utils.get_metric_value(
        metric_dict=metric_dict,
        metric_names=cfg.get("optimized_metric"),
    )

    return metric_value


if __name__ == "__main__":
    main()
