import os
from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd
import numpy as np

@dataclass
class SurvivalLabel:
    time: float
    event: int  # 1=Dead, 0=Alive

class TCGAClinicalTable:
    def __init__(self, clin_path: str):
        # clinical.tsv 在 IMAGE_ROOT 下，tab 分隔
        self.df = pd.read_csv(clin_path, sep="\t")
        self.df["case_submitter_id"] = self.df["case_submitter_id"].astype(str)
        self.df = self.df.set_index("case_submitter_id")

    def get_survival_label(self, case_id: str) -> Optional[SurvivalLabel]:
        if case_id not in self.df.index:
            return None
        row = self.df.loc[case_id]
        vital = str(row.get("vital_status", "")).strip().lower()

        if vital == "dead":
            t = row.get("days_to_death", np.nan)
            e = 1
        else:
            t = row.get("days_to_last_follow_up", np.nan)
            e = 0

        if pd.isna(t):
            return None
        return SurvivalLabel(time=float(t), event=int(e))

    def get_row_dict(self, case_id: str) -> Optional[Dict]:
        if case_id not in self.df.index:
            return None
        return self.df.loc[case_id].to_dict()


def slide_id_from_path(svs_path: str) -> str:
    return os.path.basename(svs_path).replace(".svs", "")

def slide_to_case_id(slide_id: str) -> str:
    """
    TCGA：通常前 3 段就是 case_id，如：
    TCGA-S2-AA1A-01Z-00-DX1 -> TCGA-S2-AA1A
    """
    parts = slide_id.split("-")
    case_id = "-".join(parts[:3])
    return case_id