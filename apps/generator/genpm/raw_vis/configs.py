from dataclasses import dataclass


@dataclass
class RawVisConfig:
    raw_pm_data_path: str
    output_path: str
