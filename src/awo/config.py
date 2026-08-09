"""Configuration loading and deterministic fingerprints."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

import yaml

ENV_PATTERN = re.compile(r"\$ENV\{([A-Z_][A-Z0-9_]*)\}")


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ConfigError(f"Environment variable {name!r} is required")
            return os.environ[name]

        return ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Top-level configuration must be a mapping: {path}")
    return loaded


def _load_with_extends(path: Path, seen: set[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(str(item) for item in [*seen, resolved])
        raise ConfigError(f"Configuration inheritance cycle: {chain}")

    seen.add(resolved)
    raw = _read_yaml(resolved)
    parent_ref = raw.pop("extends", None)
    merged: dict[str, Any] = {}
    if parent_ref is not None:
        refs = [parent_ref] if isinstance(parent_ref, str) else parent_ref
        if not isinstance(refs, list) or not all(isinstance(item, str) for item in refs):
            raise ConfigError("'extends' must be a path or a list of paths")
        for ref in refs:
            merged = _deep_merge(merged, _load_with_extends(resolved.parent / ref, seen))
    merged = _deep_merge(merged, raw)
    seen.remove(resolved)
    return merged


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML, resolve inheritance/model config, and expand explicit env references."""

    resolved = Path(path).resolve()
    config = _expand_env(_load_with_extends(resolved, set()))
    if config.get("schema_version") != 1:
        raise ConfigError("Only schema_version=1 is supported")

    model_ref = config.get("model_config")
    if model_ref is not None:
        if not isinstance(model_ref, str):
            raise ConfigError("'model_config' must be a relative YAML path")
        model = _expand_env(_read_yaml((resolved.parent / model_ref).resolve()))
        if model.get("schema_version") != 1:
            raise ConfigError("Model configuration must use schema_version=1")
        config["model"] = model

    config["_config_path"] = str(resolved)
    return config


def public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove runtime-only metadata before hashing or persisting a resolved config."""

    return {key: value for key, value in config.items() if not key.startswith("_")}


def config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        public_config(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
