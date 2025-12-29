from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


def _default_kernel():
    return RBF(length_scale=2.0, length_scale_bounds=(0.5, 10.0)) + WhiteKernel(
        noise_level=0.1, noise_level_bounds=(1e-5, 1.0)
    )


def _prepare_alpha(alpha_values: Iterable[float], min_alpha: float = 1e-4) -> np.ndarray:
    arr = np.array(list(alpha_values), dtype=float)
    arr[~np.isfinite(arr)] = 0.5
    arr = np.clip(arr, min_alpha, None)
    return arr


def fit_gpr_curve(
    data_points: pd.DataFrame,
    timeline: Optional[Iterable[float]] = None,
) -> Dict[str, list]:
    """
    对传入的数据点进行 Gaussian Process Regression，返回时间轴、均值曲线和置信区间。
    """
    if data_points.empty:
        return {"timeline": [], "mean_curve": [], "upper": [], "lower": []}

    if timeline is None:
        timeline = np.arange(6, 19, 1)
    timeline = np.array(list(timeline), dtype=float)

    df = data_points.dropna(subset=["age", "rate"]).copy()
    if df.empty:
        return {"timeline": timeline.tolist(), "mean_curve": [], "upper": [], "lower": []}

    X = df["age"].to_numpy(dtype=float).reshape(-1, 1)
    y = df["rate"].to_numpy(dtype=float)
    if "weight_alpha" in df.columns:
        alpha_series = df["weight_alpha"]
    else:
        alpha_series = pd.Series([0.5] * len(df))
    alpha = _prepare_alpha(alpha_series.fillna(0.5))

    kernel = _default_kernel()
    try:
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True)
        gpr.fit(X, y)
        y_mean, y_std = gpr.predict(timeline.reshape(-1, 1), return_std=True)
    except Exception:
        # 如果拟合失败，退回到简单平均
        mean_rate = float(np.mean(y))
        y_mean = np.full_like(timeline, mean_rate, dtype=float)
        y_std = np.full_like(timeline, 0.1, dtype=float)

    ci = 1.96 * y_std
    return {
        "timeline": timeline.tolist(),
        "mean_curve": y_mean.tolist(),
        "upper": (y_mean + ci).tolist(),
        "lower": (y_mean - ci).tolist(),
    }
