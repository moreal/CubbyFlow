from dataclasses import dataclass, field
from omegaconf import MISSING

@dataclass
class HWConfig:
    gpu_idx: str = "0"
    num_workers: int = 10


@dataclass
class NetworkConfig:  # flexible
    network: str = "fc_pt"
    checkpoint: str = ""
    # 2567 -> 4565
    in_size: int = 2657 * 3
    out_size: int = 4965 * 3


@dataclass
class DataConfig:
    ds_name: str = "PNT"
    from_dir: str = r"C:\Users\davin\project\CubbyFlow\win_build\bin\Release\HybridLiquidSim_output8"
    to_dir: str = r"C:\Users\davin\project\CubbyFlow\win_build\bin\Release\HybridLiquidSim_output10"


@dataclass
class OptConfig:  # flexible
    opt: str = "Adam"
    lr: float = 1e-3


@dataclass
class LogConfig:
    project_name: str = "with_aug"
    val_log_freq_epoch: int = 1
    epoch: int = 10

@dataclass
class DefaultConfig:
    hw: HWConfig = HWConfig()
    network: NetworkConfig = NetworkConfig()
    data: DataConfig = DataConfig()
    opt: OptConfig = OptConfig()
    log: LogConfig = LogConfig()
    seed: str = 42
