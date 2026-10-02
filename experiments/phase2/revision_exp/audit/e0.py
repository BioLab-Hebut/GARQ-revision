"""Runtime evidence for E0 without altering released GARQ semantics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model import GARQ, GARQuantizer, TransformerDecoder, TransformerEncoder
from revision_exp.utils.provenance import sha256_file, write_json


def _shape(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [_shape(item) for item in value]
    if value is None:
        return None
    return type(value).__name__


def capture_tensor_shapes() -> dict[str, Any]:
    torch.manual_seed(0)
    model = GARQ([11, 7], ["RNA", "ADT"], entry_num=5, entry_dim=8, k_knn=3)
    model.eval()
    events: list[dict[str, Any]] = []
    hooks = []

    def hook(name: str):
        def record(_module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            events.append({"module": name, "input": _shape(inputs), "output": _shape(output)})
        return record

    for name, module in model.named_modules():
        if isinstance(module, (TransformerEncoder, TransformerDecoder, torch.nn.TransformerEncoderLayer)):
            hooks.append(module.register_forward_hook(hook(name)))

    inputs = [torch.randn(6, 11), torch.randn(6, 7)]
    with torch.no_grad():
        hiddens = model(inputs)
        model.decode(hiddens)
        model.quantize(hiddens, return_assignment=True)
    for handle in hooks:
        handle.remove()
    return {
        "implementation_tag": "instrumented_legacy",
        "batch_size": 6,
        "events": events,
        "sequence_length_observed": sorted(
            {event["input"][0][1] for event in events if "transformer" in event["module"] and isinstance(event["input"], list)}
        ),
    }


def capture_parameter_registration() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(1)
    model = GARQ([6], ["RNA"], entry_num=4, entry_dim=8, k_knn=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    rows: list[dict[str, Any]] = []

    def snapshot(stage: str) -> None:
        for name, parameter in model.named_parameters():
            rows.append(
                {
                    "stage": stage,
                    "parameter_name": name,
                    "object_id": id(parameter),
                    "in_optimizer": id(parameter) in optimizer_ids,
                    "value": float(parameter.detach()) if parameter.numel() == 1 else "",
                }
            )

    snapshot("after_optimizer_before_forward")
    hidden = model([torch.randn(5, 6)])
    model.quantize(hidden)
    first_alpha_id = id(model.quantizer.alpha)
    snapshot("after_first_forward")
    hidden = model([torch.randn(5, 6)])
    model.quantize(hidden)
    second_alpha_id = id(model.quantizer.alpha)
    snapshot("after_second_forward")
    summary = {
        "alpha_present_before_forward": any(
            row["parameter_name"] == "quantizer.alpha" and row["stage"] == "after_optimizer_before_forward" for row in rows
        ),
        "alpha_in_optimizer_after_forward": any(
            row["parameter_name"] == "quantizer.alpha" and row["in_optimizer"] for row in rows
        ),
        "alpha_recreated_each_forward": first_alpha_id != second_alpha_id,
        "first_alpha_object_id": first_alpha_id,
        "second_alpha_object_id": second_alpha_id,
    }
    return rows, summary


def attention_dependency() -> list[dict[str, Any]]:
    torch.manual_seed(2)
    encoder = TransformerEncoder(5, 8)
    encoder.eval()
    target = torch.tensor([[1.0, -0.5, 0.2, 0.7, -1.1]])
    batch_a = torch.cat([target, torch.zeros(1, 5)], dim=0)
    batch_b = torch.cat([target, torch.full((1, 5), 100.0)], dim=0)
    with torch.no_grad():
        out_a = encoder(batch_a)[0]
        out_b = encoder(batch_b)[0]
    return [
        {
            "implementation_tag": "instrumented_legacy",
            "component": "TransformerEncoder",
            "inter_cell_input_changed": True,
            "target_output_max_abs_change": float(torch.max(torch.abs(out_a - out_b))),
            "sequence_length": 1,
        }
    ]


def batch_local_graph_dependency() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(3)
    for trial in range(1000):
        quantizer = GARQuantizer(entry_num=4, entry_dim=5, k_knn=2)
        quantizer.eval()
        with torch.no_grad():
            quantizer.anchors.weight.copy_(torch.randn(4, 5, generator=generator))
        target = torch.randn(1, 5, generator=generator)
        context_a = torch.randn(4, 5, generator=generator)
        context_b = torch.randn(4, 5, generator=generator) * 3
        sim_a, _ = quantizer.encode_relation(torch.cat([target, context_a]))
        assignment_a = int(sim_a[0].argmax())
        sim_b, _ = quantizer.encode_relation(torch.cat([target, context_b]))
        assignment_b = int(sim_b[0].argmax())
        if assignment_a != assignment_b:
            return {
                "implementation_tag": "instrumented_legacy",
                "trial": trial,
                "assignment_context_a": assignment_a,
                "assignment_context_b": assignment_b,
                "target_similarity_max_abs_change": float(torch.max(torch.abs(sim_a[0] - sim_b[0]))),
                "assignment_changed": True,
            }
    raise AssertionError("No context-dependent assignment found in 1000 deterministic trials")


def anchor_update_trace(steps: int = 100) -> list[dict[str, Any]]:
    torch.manual_seed(4)
    quantizer = GARQuantizer(entry_num=6, entry_dim=4, k_knn=2)
    quantizer.train()
    inputs = torch.randn(12, 4)
    rows = []
    for step in range(1, steps + 1):
        before = quantizer.anchors.weight.detach().clone()
        quantizer(inputs, return_assignment=False)
        displacement = torch.linalg.vector_norm(quantizer.anchors.weight.detach() - before, dim=1)
        usage = quantizer.anchor_usage.detach()
        rows.append(
            {
                "step": step,
                "usage_sum": float(usage.sum()),
                "usage_min": float(usage.min()),
                "usage_median": float(usage.median()),
                "usage_max": float(usage.max()),
                "scheduled_long_update_anchor_count": quantizer.entry_num,
                "effective_displacement_anchor_count": int(torch.sum(displacement > 0)),
                "all_anchors_displaced": bool(torch.all(displacement > 0)),
                "displacement_min": float(displacement.min()),
                "displacement_max": float(displacement.max()),
                "local_branch_condition": bool(float(usage.sum()) + 1e-4 >= 1.0),
            }
        )
    return rows


def faiss_backend_diagnostic() -> dict[str, Any]:
    import faiss

    rng = np.random.default_rng(5)
    x = rng.normal(size=(256, 8)).astype(np.float32)
    outcome: dict[str, Any] = {
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "standard_gpu_resources_available": hasattr(faiss, "StandardGpuResources"),
    }
    try:
        kmeans = faiss.Kmeans(8, 4, spherical=True, gpu=True, niter=2, verbose=False)
        kmeans.train(x)
        outcome.update({"released_gpu_true_call_succeeded": True, "index_type": type(kmeans.index).__name__})
    except Exception as error:  # evidence must retain the exact failure
        outcome.update({"released_gpu_true_call_succeeded": False, "error_type": type(error).__name__, "error": str(error)})
    return outcome


def source_manifest(repo_root: Path) -> list[dict[str, Any]]:
    suffixes = {".py", ".yaml", ".yml", ".toml", ".sh", ".R", ".ipynb"}
    return [
        {"path": str(path.relative_to(repo_root)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(repo_root.rglob("*"))
        if path.is_file() and ".git" not in path.parts and path.suffix in suffixes
    ]


def run_e0(output_dir: Path, repo_root: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shape_trace = capture_tensor_shapes()
    write_json(output_dir / "tensor_shapes.json", shape_trace)
    write_json(output_dir / "runtime_shape_trace.json", shape_trace)
    parameter_rows, parameter_summary = capture_parameter_registration()
    with (output_dir / "parameter_registration_before_after.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(parameter_rows[0]))
        writer.writeheader()
        writer.writerows(parameter_rows)
    write_json(output_dir / "parameter_registration_summary.json", parameter_summary)
    attention_rows = attention_dependency()
    with (output_dir / "attention_dependency_test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(attention_rows[0]))
        writer.writeheader()
        writer.writerows(attention_rows)
    write_json(output_dir / "batch_local_graph_dependency.json", batch_local_graph_dependency())
    trace = anchor_update_trace()
    with (output_dir / "anchor_dynamics_step.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    write_json(output_dir / "faiss_backend.json", faiss_backend_diagnostic())
    write_json(output_dir / "source_sha256.json", source_manifest(repo_root))
