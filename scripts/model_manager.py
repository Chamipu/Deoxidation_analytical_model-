# -*- coding: utf-8 -*-
"""Adapter to choose between independent and coupled predictor implementations.

Provides a single entrypoint `apply_model` that keeps the GUI code stable
while allowing swapping underlying implementations.
"""
from __future__ import annotations

from typing import Mapping, Tuple
import pandas as pd

from scripts.independent_linear_predictor import pressure_predictor_lite as ind
from scripts.coupled_linear_predictor import pressure_predictor_coupled as coup


def apply_model(
    df_logs: pd.DataFrame,
    df_registry: pd.DataFrame,
    tank_config: Mapping[str, any],
    pipe_config: Mapping[str, any],
    flat_params_tank: Mapping[str, float],
    flat_params_pipe: Mapping[str, float],
    model: str = "independent",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply selected predictor and return updated (df_logs, df_registry).

    model: 'independent' or 'coupled'.
    - independent: calls the old `pressure_predictor_lite.apply_analytic_model` for
      tank and pipe separately (keeps legacy behaviour).
    - coupled: calls `pressure_predictor_coupled.apply_coupled_model` which computes
      both channels jointly.
    """
    model = (model or "independent").lower()

    if model == "coupled":
        return coup.apply_coupled_model(
            df_logs=df_logs,
            df_registry=df_registry,
            config_tank=tank_config,
            config_pipe=pipe_config,
            flat_params_tank=flat_params_tank,
            flat_params_pipe=flat_params_pipe,
        )

    # fallback: independent (legacy)
    df_logs, df_registry = ind.apply_analytic_model(df_logs, df_registry, tank_config, flat_params_tank)
    df_logs, df_registry = ind.apply_analytic_model(df_logs, df_registry, pipe_config, flat_params_pipe)
    return df_logs, df_registry
