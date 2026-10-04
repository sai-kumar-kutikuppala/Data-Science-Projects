import pandas as pd


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same cleaning used in the notebook before modeling."""
    df = df.copy()

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace("Fe Male", "Female")
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace("Single", "Unmarried")

    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    if "NumberOfPersonVisiting" in df.columns and "NumberOfChildrenVisiting" in df.columns:
        df["TotalVisiting"] = df["NumberOfPersonVisiting"] + df["NumberOfChildrenVisiting"]
        df = df.drop(columns=["NumberOfPersonVisiting", "NumberOfChildrenVisiting"])

    return df
