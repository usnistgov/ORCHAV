"""Processing implementations selected by the sensing processor factory.

Coherent CPI processing consumes samples from one solve; sequential STFT
accumulates independent solves. The analytic path-Doppler oracle bins solver
path powers for diagnostics. Each entry point returns a sensing payload, or
``None`` when processing is disabled or usable input is unavailable.
"""

from .analytic_path_doppler_oracle import process_analytic_path_doppler_oracle
from .coherent_cpi_fft import process_coherent_cpi_fft
from .sequential_stft import process_sequential_stft

__all__ = [
    "process_analytic_path_doppler_oracle",
    "process_coherent_cpi_fft",
    "process_sequential_stft",
]
