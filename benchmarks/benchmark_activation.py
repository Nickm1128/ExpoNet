"""Measure ExpoActivation overhead against focused activation references."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn
from torch.utils import benchmark

from exponet import ExpoActivation

BATCH_SIZES = (1, 128)
WIDTHS = (64, 256)
VARIANTS = ("relu", "squared_relu", "sine", "learned_blend")
MODES = ("inference", "forward_backward")
NORMALIZATIONS = ("none", "layer")
SEED = 20260830


class SquaredReLU(nn.Module):
    """Native squared-ReLU timing reference."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply ReLU once and square its result."""
        u = torch.relu(x)
        return u * u


class Sine(nn.Module):
    """Plain sine timing reference."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply sine elementwise."""
        return torch.sin(x)


@dataclass(frozen=True)
class TimingResult:
    """One calibrated timing measurement."""

    section: str
    variant: str
    mode: str
    batch_size: int | None
    width: int
    normalization: str | None
    median_us: float
    iqr_us: float
    measurements: int
    iterations_per_measurement: int


def parse_args() -> argparse.Namespace:
    """Parse benchmark options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--min-run-time", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_threads <= 0:
        parser.error("--num-threads must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.min_run_time <= 0:
        parser.error("--min-run-time must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested, but torch.cuda.is_available() is False")
    return args


def deterministic_input(
    batch_size: int, width: int, *, device: torch.device, requires_grad: bool
) -> Tensor:
    """Create matched input data outside timed regions."""
    values = torch.linspace(
        -2.0,
        2.0,
        steps=batch_size * width,
        dtype=torch.float32,
        device=device,
    )
    return values.reshape(batch_size, width).requires_grad_(requires_grad)


def activation_module(variant: str, width: int, *, device: torch.device) -> nn.Module:
    """Build one isolated activation reference."""
    if variant == "relu":
        module: nn.Module = nn.ReLU()
    elif variant == "squared_relu":
        module = SquaredReLU()
    elif variant == "sine":
        module = Sine()
    elif variant == "learned_blend":
        module = ExpoActivation(num_features=width, blend_init=0.5)
    else:
        raise ValueError(f"Unknown activation variant: {variant}")
    return module.to(device=device, dtype=torch.float32)


def dense_block(
    variant: str, width: int, normalization: str, *, device: torch.device
) -> nn.Module:
    """Build a deterministic Linear -> normalization -> activation block."""
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    norm: nn.Module
    if normalization == "none":
        norm = nn.Identity()
    elif normalization == "layer":
        norm = nn.LayerNorm(width)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    module = nn.Sequential(
        nn.Linear(width, width),
        norm,
        activation_module(variant, width, device=device),
    )
    return module.to(device=device, dtype=torch.float32)


def synchronized(device: torch.device, operation: Callable[[], Tensor]) -> Tensor:
    """Run one operation with CUDA synchronization when required."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result


def measure(
    *,
    section: str,
    variant: str,
    mode: str,
    batch_size: int | None,
    width: int,
    normalization: str | None,
    operation: Callable[[], Tensor],
    device: torch.device,
    warmup: int,
    min_run_time: float,
    num_threads: int,
) -> TimingResult:
    """Warm up and collect a calibrated median and interquartile range."""

    def timed_operation() -> Tensor:
        return synchronized(device, operation)

    for _ in range(warmup):
        timed_operation()
    measurement = benchmark.Timer(
        stmt="timed_operation()",
        globals={"timed_operation": timed_operation},
        num_threads=num_threads,
    ).blocked_autorange(min_run_time=min_run_time)
    return TimingResult(
        section=section,
        variant=variant,
        mode=mode,
        batch_size=batch_size,
        width=width,
        normalization=normalization,
        median_us=measurement.median * 1_000_000,
        iqr_us=measurement.iqr * 1_000_000,
        measurements=len(measurement.raw_times),
        iterations_per_measurement=measurement.number_per_run,
    )


def inference_operation(module: nn.Module, x: Tensor) -> Callable[[], Tensor]:
    """Create an inference closure with setup excluded from timing."""
    module.eval()

    @torch.inference_mode()
    def run() -> Tensor:
        return module(x)

    return run


def backward_operation(module: nn.Module, x: Tensor) -> Callable[[], Tensor]:
    """Create a forward/backward closure with consistent gradient clearing."""
    module.train()

    def run() -> Tensor:
        module.zero_grad(set_to_none=True)
        x.grad = None
        output = module(x)
        output.sum().backward()
        return output

    return run


def mapping_operation(
    width: int, mode: str, *, device: torch.device
) -> Callable[[], Tensor]:
    """Create a standalone sigmoid-mapping timing closure."""
    theta = torch.zeros(
        width,
        dtype=torch.float32,
        device=device,
        requires_grad=mode == "forward_backward",
    )
    if mode == "inference":

        @torch.inference_mode()
        def run() -> Tensor:
            return torch.sigmoid(theta)

        return run

    def run() -> Tensor:
        theta.grad = None
        output = torch.sigmoid(theta)
        output.sum().backward()
        return output

    return run


def collect_results(args: argparse.Namespace) -> list[TimingResult]:
    """Run the complete early activation timing matrix."""
    device = torch.device(args.device)
    results: list[TimingResult] = []

    for width in WIDTHS:
        for mode in MODES:
            results.append(
                measure(
                    section="coefficient_mapping",
                    variant="sigmoid",
                    mode=mode,
                    batch_size=None,
                    width=width,
                    normalization=None,
                    operation=mapping_operation(width, mode, device=device),
                    device=device,
                    warmup=args.warmup,
                    min_run_time=args.min_run_time,
                    num_threads=args.num_threads,
                )
            )

    for batch_size in BATCH_SIZES:
        for width in WIDTHS:
            for variant in VARIANTS:
                for mode in MODES:
                    module = activation_module(variant, width, device=device)
                    x = deterministic_input(
                        batch_size,
                        width,
                        device=device,
                        requires_grad=mode == "forward_backward",
                    )
                    operation = (
                        inference_operation(module, x)
                        if mode == "inference"
                        else backward_operation(module, x)
                    )
                    results.append(
                        measure(
                            section="isolated_activation",
                            variant=variant,
                            mode=mode,
                            batch_size=batch_size,
                            width=width,
                            normalization=None,
                            operation=operation,
                            device=device,
                            warmup=args.warmup,
                            min_run_time=args.min_run_time,
                            num_threads=args.num_threads,
                        )
                    )

    for normalization in NORMALIZATIONS:
        for batch_size in BATCH_SIZES:
            for width in WIDTHS:
                for variant in VARIANTS:
                    for mode in MODES:
                        module = dense_block(
                            variant, width, normalization, device=device
                        )
                        x = deterministic_input(
                            batch_size,
                            width,
                            device=device,
                            requires_grad=mode == "forward_backward",
                        )
                        operation = (
                            inference_operation(module, x)
                            if mode == "inference"
                            else backward_operation(module, x)
                        )
                        results.append(
                            measure(
                                section="dense_block",
                                variant=variant,
                                mode=mode,
                                batch_size=batch_size,
                                width=width,
                                normalization=normalization,
                                operation=operation,
                                device=device,
                                warmup=args.warmup,
                                min_run_time=args.min_run_time,
                                num_threads=args.num_threads,
                            )
                        )
    return results


def environment(device: torch.device, num_threads: int) -> dict[str, object]:
    """Record the timing environment and relevant library versions."""
    details: dict[str, object] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": str(device),
        "dtype": "float32",
        "num_threads": num_threads,
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        details["cuda_device_name"] = torch.cuda.get_device_name(device)
        details["cuda_runtime"] = torch.version.cuda
    return details


def main() -> None:
    """Run the benchmark and write machine-readable results."""
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device)
    results = collect_results(args)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "environment": environment(device, args.num_threads),
        "protocol": {
            "seed": SEED,
            "batch_sizes": BATCH_SIZES,
            "widths": WIDTHS,
            "variants": VARIANTS,
            "modes": MODES,
            "normalizations": NORMALIZATIONS,
            "warmup_iterations": args.warmup,
            "minimum_run_time_seconds": args.min_run_time,
            "timer": "torch.utils.benchmark.Timer.blocked_autorange",
        },
        "results": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote={args.output.resolve()}")
    print(f"measurements={len(results)}")


if __name__ == "__main__":
    main()
