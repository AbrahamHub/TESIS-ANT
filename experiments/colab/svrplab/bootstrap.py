"""Arranque del entorno (Google Colab / local) para el pipeline EHBG-FACS.

Este módulo resuelve, de forma idempotente, todo lo necesario para que los cinco
notebooks de paradigma corran sobre **el mismo** ecosistema:

  1. Clona (o localiza) el repositorio **oficial** de SVRPBench
     (https://github.com/yehias21/svrpbench) y lo agrega al ``sys.path`` con los
     dos estilos de import que el repo mezcla (paquete ``vrp_bench.core`` + módulos
     planos ``city``/``time_windows_generator``/``aco_solver``/...).
  2. Detecta y reporta la GPU disponible (Colab Pro / Pro+), fija la precisión de
     matmul de PyTorch y devuelve el ``device`` recomendado.
  3. Centraliza las rutas de artefactos (``data/``, ``results/``, ``figures/``,
     ``models/``) bajo una raíz única, persistible en Google Drive.
  4. Siembra de forma global ``random``/``numpy``/``torch`` para reproducibilidad.

Todas las funciones son seguras de llamar varias veces (no duplican rutas en
``sys.path`` ni vuelven a clonar).
"""
from __future__ import annotations

import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

OFFICIAL_REPO_URL = "https://github.com/yehias21/svrpbench"

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #


@dataclass
class Paths:
    """Rutas canónicas del pipeline. ``root`` es persistible (p. ej. Google Drive)."""

    root: Path
    official_repo: Path
    data: Path = field(init=False)
    instances: Path = field(init=False)
    results: Path = field(init=False)
    figures: Path = field(init=False)
    models: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data = self.root / "data"
        self.instances = self.data / "instances"
        self.results = self.root / "results"
        self.figures = self.root / "figures"
        self.models = self.data / "models"
        for p in (self.data, self.instances, self.results, self.figures, self.models):
            p.mkdir(parents=True, exist_ok=True)


def _detect_root(explicit: Optional[str]) -> Path:
    """Raíz de artefactos. Prioridad: argumento explícito > Google Drive montado >
    carpeta local ``./svrplab_runs``."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    drive = Path("/content/drive/MyDrive")
    if drive.exists():
        return drive / "EHBG_FACS"
    return (Path.cwd() / "svrplab_runs").resolve()


# --------------------------------------------------------------------------- #
# Repositorio oficial de SVRPBench
# --------------------------------------------------------------------------- #


def ensure_official_repo(dest: Path, *, url: str = OFFICIAL_REPO_URL,
                         verbose: bool = True) -> Path:
    """Clona el repo oficial en ``dest`` si no existe; devuelve la ruta del repo.

    Acepta que ``dest`` ya contenga un repo (carpeta ``vrp_bench`` presente), en
    cuyo caso no hace nada. Esto permite reutilizar un clon previo (Drive) o uno
    incluido en la tesis (``experiments/svrp/third_party/svrpbench``)."""
    pkg = dest / "vrp_bench"
    if pkg.exists():
        if verbose:
            print(f"[bootstrap] repo oficial ya presente en {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[bootstrap] clonando {url} -> {dest}")
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
    if not pkg.exists():
        raise RuntimeError(
            f"El clon de {url} no contiene 'vrp_bench/'. Revisa la URL/red.")
    return dest


def _add_official_to_path(repo: Path) -> None:
    pkg = repo / "vrp_bench"
    for p in (str(repo), str(pkg)):  # paquete moderno + módulos planos heredados
        if p not in sys.path:
            sys.path.insert(0, p)


# --------------------------------------------------------------------------- #
# GPU / PyTorch
# --------------------------------------------------------------------------- #


def device_report(verbose: bool = True) -> str:
    """Devuelve ``"cuda"`` si hay GPU, ``"cpu"`` en caso contrario, e imprime un
    informe del hardware. Ajusta la precisión de matmul para aprovechar Tensor
    Cores (A100/L4/T4) en Colab Pro/Pro+."""
    try:
        import torch
    except ImportError:
        if verbose:
            print("[bootstrap] PyTorch no instalado (ok para paradigmas 1-2).")
        return "cpu"

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        if verbose:
            print(f"[bootstrap] GPU detectada: {name} ({mem:.1f} GB) | "
                  f"torch {torch.__version__} CUDA {torch.version.cuda}")
        return "cuda"
    if verbose:
        print(f"[bootstrap] sin GPU; usando CPU | torch {torch.__version__}")
    return "cpu"


def seed_everything(seed: int = 12345) -> None:
    """Siembra global de ``random``/``numpy``/``torch`` (+ CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Entrada única
# --------------------------------------------------------------------------- #


@dataclass
class Env:
    paths: Paths
    device: str
    official_repo: Path


def init(root: Optional[str] = None,
         official_repo: Optional[str] = None,
         *, seed: int = 12345, verbose: bool = True) -> Env:
    """Punto de entrada de los notebooks. Devuelve un ``Env`` con rutas, ``device``
    y la ubicación del repo oficial, ya en el ``sys.path`` y con semillas fijadas.

    Parámetros
    ----------
    root : str, opcional
        Raíz de artefactos (resultados/figuras/datos). Por defecto, Google Drive si
        está montado, o ``./svrplab_runs``. Usar la **misma** raíz en los 5 notebooks
        garantiza que comparten dataset y acumulan resultados comparables.
    official_repo : str, opcional
        Ruta donde clonar/buscar el repo oficial. Por defecto ``<root>/svrpbench``.
    """
    base = _detect_root(root)
    base.mkdir(parents=True, exist_ok=True)
    repo = Path(official_repo).expanduser().resolve() if official_repo else (base / "svrpbench")
    repo = ensure_official_repo(repo, verbose=verbose)
    _add_official_to_path(repo)
    paths = Paths(root=base, official_repo=repo)
    dev = device_report(verbose=verbose)
    seed_everything(seed)
    if verbose:
        print(f"[bootstrap] raíz de artefactos: {paths.root}")
        print(f"[bootstrap] results/ figures/ data/ listos | seed={seed} device={dev}")
    return Env(paths=paths, device=dev, official_repo=repo)
