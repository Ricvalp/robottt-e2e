"""
QuickDraw dataset processing package.

This package provides utilities to preprocess Quick, Draw! sketches, build
K-shot imitation learning episodes, store processed data efficiently, and
serve it through PyTorch `Dataset` interfaces.
"""

from .diffusion import (
    ContextQueryInContextDiffusionCollator,
    MAMLDiffusionCollator,
)
from .episode_builder import (
    Episode,
    EpisodeBuilder,
    EpisodeBuilderSimilar,
    EpisodeBuilderSimilarMAML,
    MAMLEpisode,
)
from .loader import QuickDrawEpisodes, QuickDrawEpisodesMAML, QuickDrawSketches
from .preprocess import (
    ProcessedSketch,
    QuickDrawPreprocessor,
    RawSketch,
    load_binary_sketches,
    load_ndjson_sketches,
)
from .rasterize import RasterizerConfig, rasterize_absolute_points
from .storage import DatasetManifest, EpisodeStorage, SketchStorage, StorageConfig

__all__ = [
    "QuickDrawPreprocessor",
    "ProcessedSketch",
    "RawSketch",
    "load_ndjson_sketches",
    "load_binary_sketches",
    "EpisodeBuilder",
    "EpisodeBuilderSimilar",
    "EpisodeBuilderSimilarMAML",
    "Episode",
    "MAMLEpisode",
    "StorageConfig",
    "DatasetManifest",
    "SketchStorage",
    "EpisodeStorage",
    "QuickDrawEpisodes",
    "QuickDrawEpisodesMAML",
    "QuickDrawSketches",
    "MAMLDiffusionCollator",
    "RasterizerConfig",
    "rasterize_absolute_points",
    "ContextQueryInContextDiffusionCollator",
]
