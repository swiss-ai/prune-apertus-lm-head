#!/opt/venv/bin/python
"""Remove Apertus image/audio token rows from the output projection only."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


IMAGE_TOKEN_START = 131_272
IMAGE_TOKEN_END = 262_343
AUDIO_TOKEN_START = 262_344
AUDIO_TOKEN_END = 266_439
LM_HEAD_KEY = "lm_head.weight"
MODEL_INDEX_NAME = "model.safetensors.index.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an Apertus checkpoint whose lm_head cannot generate image "
            "or audio code token IDs. Input embeddings and the tokenizer are kept."
        )
    )
    parser.add_argument(
        "source", nargs="?", type=Path, help="Source Hugging Face checkpoint"
    )
    parser.add_argument(
        "output", nargs="?", type=Path, help="New converted checkpoint directory"
    )
    parser.add_argument(
        "--source",
        dest="source_option",
        type=Path,
        help="Source Hugging Face checkpoint",
    )
    parser.add_argument(
        "--output",
        dest="output_option",
        type=Path,
        help="New converted checkpoint directory",
    )
    parser.add_argument(
        "--copy-unchanged",
        action="store_true",
        help="Copy unchanged files instead of hard-linking them when possible.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the source checkpoint and report the planned conversion.",
    )
    args = parser.parse_args()
    if args.source is not None and args.source_option is not None:
        parser.error("provide the source path once, either positionally or with --source")
    if args.output is not None and args.output_option is not None:
        parser.error("provide the output path once, either positionally or with --output")
    args.source = args.source or args.source_option
    args.output = args.output or args.output_option
    if args.source is None or args.output is None:
        parser.error("both source and output paths are required")
    return args


def copy_unchanged(source: Path, destination: Path, *, force_copy: bool) -> None:
    """Copy a file, preferring hard links to avoid duplicating unchanged weights."""
    if force_copy:
        shutil.copy2(source, destination)
        return

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def copy_non_model_files(
    source: Path,
    destination: Path,
    model_shards: set[str],
    *,
    force_copy: bool,
) -> None:
    for item in source.iterdir():
        if item.name in model_shards or item.name in {MODEL_INDEX_NAME, "config.json"}:
            continue
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(
                item,
                target,
                copy_function=lambda src, dst: copy_unchanged(
                    Path(src), Path(dst), force_copy=force_copy
                ),
            )
        else:
            copy_unchanged(item, target, force_copy=force_copy)


def inspect_source(source: Path) -> tuple[dict, dict, Path, int, int, int]:
    index_path = source / MODEL_INDEX_NAME
    config_path = source / "config.json"
    if not source.is_dir():
        raise ValueError(f"Source checkpoint directory does not exist: {source}")
    if not index_path.is_file():
        raise ValueError(f"Missing sharded safetensors index: {index_path}")
    if not config_path.is_file():
        raise ValueError(f"Missing config.json: {config_path}")

    index = json.loads(index_path.read_text())
    config = json.loads(config_path.read_text())
    try:
        head_shard_name = index["weight_map"][LM_HEAD_KEY]
    except KeyError as error:
        raise ValueError(f"{LM_HEAD_KEY} is not listed in {index_path}") from error

    head_shard = source / head_shard_name
    if not head_shard.is_file():
        raise ValueError(f"lm_head shard does not exist: {head_shard}")

    with safe_open(head_shard, framework="pt", device="cpu") as weights:
        if LM_HEAD_KEY not in weights.keys():
            raise ValueError(f"{LM_HEAD_KEY} is not present in {head_shard}")
        shape = weights.get_slice(LM_HEAD_KEY).get_shape()
        if len(shape) != 2:
            raise ValueError(f"Expected a 2-D {LM_HEAD_KEY}, found shape {shape}")
        rows, columns = shape
        tensor = weights.get_tensor(LM_HEAD_KEY)
        element_size = tensor.element_size()
        del tensor

    if rows <= IMAGE_TOKEN_START:
        raise ValueError(
            f"{LM_HEAD_KEY} has {rows} rows; it does not contain image code IDs "
            f"starting at {IMAGE_TOKEN_START}."
        )
    if config.get("vocab_size") != rows:
        raise ValueError(
            "config.json vocab_size does not match lm_head rows "
            f"({config.get('vocab_size')} != {rows})."
        )
    return index, config, head_shard, rows, columns, element_size


def write_pruned_head(source: Path, destination: Path) -> None:
    """Write one safetensors shard, retaining only text/control-token lm-head rows."""
    tensors = {}
    with safe_open(source, framework="pt", device="cpu") as weights:
        metadata = weights.metadata()
        for key in weights.keys():
            tensor = weights.get_tensor(key)
            if key == LM_HEAD_KEY:
                tensor = tensor[:IMAGE_TOKEN_START].contiguous()
            tensors[key] = tensor
    save_file(tensors, destination, metadata=metadata)


def convert(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    output = args.output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Output must not be the source directory or one of its children.")
    if output.exists():
        raise ValueError(f"Output path already exists: {output}")

    index, config, head_shard, rows, columns, element_size = inspect_source(source)
    removed_rows = rows - IMAGE_TOKEN_START
    removed_bytes = removed_rows * columns * element_size
    print(
        f"{LM_HEAD_KEY}: {rows}x{columns} -> {IMAGE_TOKEN_START}x{columns} "
        f"(removes {removed_rows:,} rows / {removed_bytes / 2**30:.2f} GiB)"
    )
    print(
        f"Removed token-ID interval: {IMAGE_TOKEN_START}-{rows - 1}; "
        f"this includes image codes {IMAGE_TOKEN_START}-{IMAGE_TOKEN_END} and "
        f"audio codes {AUDIO_TOKEN_START}-{AUDIO_TOKEN_END}."
    )
    if args.dry_run:
        return

    temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir(parents=True)
    try:
        model_shards = set(index["weight_map"].values())
        copy_non_model_files(
            source, temporary, model_shards, force_copy=args.copy_unchanged
        )

        for shard_name in sorted(model_shards):
            source_shard = source / shard_name
            target_shard = temporary / shard_name
            if source_shard == head_shard:
                write_pruned_head(source_shard, target_shard)
            else:
                copy_unchanged(source_shard, target_shard, force_copy=args.copy_unchanged)

        config["output_vocab_size"] = IMAGE_TOKEN_START
        (temporary / "config.json").write_text(json.dumps(config, indent=2) + "\n")

        metadata = index.setdefault("metadata", {})
        if "total_size" in metadata:
            metadata["total_size"] -= removed_bytes
        (temporary / MODEL_INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n")

        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    args = parse_args()
    try:
        convert(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
