"""
Data loading for FreshRetailNet-50K (Hugging Face).
Real dataset: Dingdong-Inc/FreshRetailNet-50K
https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple


def load_dataset_freshretail(
    max_train_rows: Optional[int] = None,
    max_eval_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load FreshRetailNet-50K from Hugging Face.
    No synthetic data or hardcoding - real download and real forecasting.

    Returns:
        tuple: (train_df, eval_df)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install Hugging Face datasets: pip install datasets"
        )

    print("Loading FreshRetailNet-50K from Hugging Face...")
    ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
    splits = list(ds.keys())
    eval_split = "eval" if "eval" in splits else ("validation" if "validation" in splits else splits[-1])
    if eval_split != "eval":
        print(f"  Using '{eval_split}' as evaluation split")

    def to_df(split: str) -> pd.DataFrame:
        data = ds[split]
        df = data.to_pandas()
        # Keep only scalar columns needed for tabular forecasting
        # (drop sequence columns: hours_sale, hours_stock_status)
        keep_cols = [
            "city_id", "store_id", "management_group_id",
            "first_category_id", "second_category_id", "third_category_id",
            "product_id", "dt", "sale_amount",
            "stock_hour6_22_cnt",
            "discount", "holiday_flag", "activity_flag",
            "precpt", "avg_temperature", "avg_humidity", "avg_wind_level",
        ]
        existing = [c for c in keep_cols if c in df.columns]
        df = df[existing].copy()
        df["dt"] = pd.to_datetime(df["dt"])
        df = df.sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True)
        return df

    train_df = to_df("train")
    eval_df = to_df(eval_split)

    if max_train_rows is not None and len(train_df) > max_train_rows:
        train_df = train_df.sample(n=max_train_rows, random_state=42).sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True)
        print(f"  Sampled train to {len(train_df)} rows (max_train_rows={max_train_rows})")
    if max_eval_rows is not None and len(eval_df) > max_eval_rows:
        eval_df = eval_df.sample(n=max_eval_rows, random_state=42).sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True)
        print(f"  Sampled eval to {len(eval_df)} rows (max_eval_rows={max_eval_rows})")

    print(f"Train shape: {train_df.shape}")
    print(f"Eval shape: {eval_df.shape}")
    print(f"Train period: {train_df['dt'].min()} to {train_df['dt'].max()}")
    print(f"Eval period: {eval_df['dt'].min()} to {eval_df['dt'].max()}")
    return train_df, eval_df
