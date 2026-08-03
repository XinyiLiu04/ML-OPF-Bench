"""
DCOPF Path Configuration
========================

Centralized path management for datasets and constraint files.
Users only need to modify ROOT_DIR to adapt to different environments.

"""

import os

class PathConfig:

    # User Configuration (Only place to modify)
    ROOT_DIR = r"C:\Users\Aloha\Desktop\dataset"

    CONSTRAINTS_DIR = "DCOPF Constraints"
    DATASET_DIR = "DCOPF dataset"

    @classmethod
    def get_constraints_path(cls, case_short_name: str, is_api: bool = False) -> str:
        folder = f"{case_short_name}(api)" if is_api else case_short_name
        return os.path.join(cls.ROOT_DIR, cls.CONSTRAINTS_DIR, folder)

    @classmethod
    def get_dataset_path(cls, case_name: str, case_short_name: str,
                         variance: str = "v=0.12", is_api: bool = False) -> str:
        if is_api:
            dir_name = f"{case_short_name}(v=api)"
            file_suffix = "__api"  # Double underscore
        else:
            dir_name = f"{case_short_name}({variance})"
            file_suffix = ""

        filename = f"{case_name}{file_suffix}_dataset_with_duals.csv"
        return os.path.join(cls.ROOT_DIR, cls.DATASET_DIR, dir_name, filename)



