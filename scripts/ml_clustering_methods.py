"""Deterministic unsupervised methods for the exploratory ML extension."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KMeansResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    iterations: int
    selected_initialization: int


@dataclass(frozen=True)
class SpectralEmbedding:
    values: np.ndarray
    bandwidth: float
    laplacian_eigenvalues: np.ndarray


@dataclass(frozen=True)
class PCAEmbedding:
    values: np.ndarray
    component_count: int
    cumulative_explained_variance: float
    explained_variance_ratios: np.ndarray


def array_sha256(values: np.ndarray) -> str:
    """Hash an array after conversion to a stable contiguous float64 representation."""

    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _validated_correlation(correlation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(correlation, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("correlation matrix must be square with at least two securities")
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
        raise ValueError("correlation matrix must be finite and symmetric")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("correlation matrix must have a unit diagonal")
    if bool(((matrix < -1.0 - 1e-12) | (matrix > 1.0 + 1e-12)).any()):
        raise ValueError("correlations must lie in [-1, 1]")
    return np.clip(matrix, -1.0, 1.0)


def _orient_columns(values: np.ndarray) -> np.ndarray:
    """Resolve arbitrary eigenvector/SVD column signs deterministically."""

    oriented = np.asarray(values, dtype=float).copy()
    for column in range(oriented.shape[1]):
        pivot = int(np.argmax(np.abs(oriented[:, column])))
        if oriented[pivot, column] < 0:
            oriented[:, column] *= -1.0
    return oriented


def spectral_embedding(correlation: np.ndarray, cluster_count: int) -> SpectralEmbedding:
    """Construct the frozen normalized-Laplacian spectral embedding."""

    matrix = _validated_correlation(correlation)
    security_count = len(matrix)
    if not 1 < cluster_count < security_count:
        raise ValueError("spectral cluster count must lie in [2, security_count)")
    squared_distance = np.maximum(0.0, (1.0 - matrix) / 2.0)
    distance = np.sqrt(squared_distance)
    off_diagonal = distance[np.triu_indices(security_count, k=1)]
    positive = off_diagonal[off_diagonal > 0]
    if positive.size == 0:
        raise ValueError("spectral affinity requires at least one positive pair distance")
    bandwidth = float(np.median(positive))
    affinity = np.exp(-squared_distance / (2.0 * bandwidth**2))
    np.fill_diagonal(affinity, 0.0)
    degrees = affinity.sum(axis=1)
    if not np.isfinite(degrees).all() or bool((degrees <= 0).any()):
        raise ValueError("spectral affinity graph has an invalid degree")
    inverse_root_degree = 1.0 / np.sqrt(degrees)
    normalized_affinity = affinity * np.outer(inverse_root_degree, inverse_root_degree)
    laplacian = np.eye(security_count) - normalized_affinity
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (laplacian + laplacian.T))
    embedding = _orient_columns(eigenvectors[:, :cluster_count])
    row_norms = np.linalg.norm(embedding, axis=1)
    if not np.isfinite(row_norms).all() or bool((row_norms <= 0).any()):
        raise ValueError("spectral embedding contains a zero or non-finite row")
    embedding = embedding / row_norms[:, None]
    return SpectralEmbedding(
        values=embedding,
        bandwidth=bandwidth,
        laplacian_eigenvalues=eigenvalues[:cluster_count],
    )


def pca_correlation_profile_embedding(
    correlation: np.ndarray,
    *,
    explained_variance_threshold: float = 0.80,
    minimum_components: int = 2,
) -> PCAEmbedding:
    """Embed centered stock correlation profiles with the frozen PCA rule."""

    matrix = _validated_correlation(correlation).copy()
    security_count = len(matrix)
    if not 0 < explained_variance_threshold < 1:
        raise ValueError("PCA explained-variance threshold must lie in (0, 1)")
    if not 1 <= minimum_components < security_count:
        raise ValueError("PCA minimum components must lie in [1, security_count)")
    for column in range(security_count):
        off_diagonal = np.delete(matrix[:, column], column)
        matrix[column, column] = float(np.mean(off_diagonal))
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2
    total = float(variance.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("PCA correlation profiles have no finite variation")
    ratios = variance / total
    cumulative = np.cumsum(ratios)
    component_count = max(
        minimum_components,
        int(np.searchsorted(cumulative, explained_variance_threshold, side="left") + 1),
    )
    component_count = min(component_count, security_count - 1)
    embedding = _orient_columns(left[:, :component_count]) * singular_values[:component_count]
    if not np.isfinite(embedding).all():
        raise ValueError("PCA embedding is non-finite")
    return PCAEmbedding(
        values=embedding,
        component_count=component_count,
        cumulative_explained_variance=float(cumulative[component_count - 1]),
        explained_variance_ratios=ratios[:component_count],
    )


def _squared_distances(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)


def _kmeans_plus_plus(
    values: np.ndarray, cluster_count: int, generator: np.random.Generator
) -> np.ndarray:
    selected = [int(generator.integers(0, len(values)))]
    closest = _squared_distances(values, values[selected]).min(axis=1)
    while len(selected) < cluster_count:
        available = np.ones(len(values), dtype=bool)
        available[selected] = False
        weights = np.where(available, closest, 0.0)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            candidates = np.flatnonzero(available)
            if not len(candidates):
                raise ValueError("k-means++ could not select distinct centers")
            candidate = int(candidates[0])
        else:
            threshold = float(generator.random()) * total
            candidate = int(np.searchsorted(np.cumsum(weights), threshold, side="right"))
            if candidate >= len(values) or not available[candidate]:
                candidate = int(np.flatnonzero(available)[-1])
        selected.append(candidate)
        closest = np.minimum(closest, _squared_distances(values, values[[candidate]])[:, 0])
    return values[selected].copy()


def _repair_empty_clusters(
    labels: np.ndarray, distances: np.ndarray, cluster_count: int
) -> np.ndarray:
    repaired = labels.copy()
    counts = np.bincount(repaired, minlength=cluster_count)
    for empty in np.flatnonzero(counts == 0):
        assigned_distance = distances[np.arange(len(repaired)), repaired]
        eligible = counts[repaired] > 1
        if not eligible.any():
            raise ValueError("cannot repair an empty cluster without emptying another")
        score = np.where(eligible, assigned_distance, -np.inf)
        moved = int(np.argmax(score))
        donor = int(repaired[moved])
        repaired[moved] = int(empty)
        counts[donor] -= 1
        counts[empty] += 1
    return repaired


def _partition_signature(labels: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(
        sorted(tuple(np.flatnonzero(labels == cluster).tolist()) for cluster in np.unique(labels))
    )


def _fit_kmeans_initialization(
    values: np.ndarray,
    cluster_count: int,
    generator: np.random.Generator,
    initialization: int,
    maximum_iterations: int,
    convergence_tolerance: float,
) -> KMeansResult | None:
    centers = _kmeans_plus_plus(values, cluster_count, generator)
    previous_labels: np.ndarray | None = None
    for iteration in range(1, maximum_iterations + 1):
        distances = _squared_distances(values, centers)
        labels = _repair_empty_clusters(np.argmin(distances, axis=1), distances, cluster_count)
        updated = np.vstack(
            [values[labels == cluster].mean(axis=0) for cluster in range(cluster_count)]
        )
        shift = float(np.max(np.linalg.norm(updated - centers, axis=1)))
        stable = previous_labels is not None and np.array_equal(labels, previous_labels)
        centers = updated
        previous_labels = labels.copy()
        if stable or shift <= convergence_tolerance:
            inertia = float(np.sum((values - centers[labels]) ** 2))
            return KMeansResult(labels, centers, inertia, iteration, initialization)
    return None


def _candidate_is_better(
    candidate: KMeansResult,
    signature: tuple[tuple[int, ...], ...],
    best: KMeansResult | None,
    best_signature: tuple[tuple[int, ...], ...] | None,
) -> bool:
    if best is None:
        return True
    if candidate.inertia < best.inertia - 1e-12:
        return True
    return bool(
        abs(candidate.inertia - best.inertia) <= 1e-12
        and best_signature is not None
        and signature < best_signature
    )


def deterministic_kmeans(
    features: np.ndarray,
    cluster_count: int,
    *,
    seed_components: Sequence[int],
    n_init: int = 100,
    maximum_iterations: int = 500,
    convergence_tolerance: float = 1e-10,
) -> KMeansResult:
    """Run deterministic multi-start k-means++ and retain the best partition."""

    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or len(values) <= cluster_count or cluster_count < 2:
        raise ValueError("k-means features or cluster count have invalid dimensions")
    if not np.isfinite(values).all():
        raise ValueError("k-means features must be finite")
    if n_init <= 0 or maximum_iterations <= 0 or convergence_tolerance < 0:
        raise ValueError("k-means controls are invalid")
    seeds = [int(value) for value in seed_components]
    if not seeds or any(value < 0 for value in seeds):
        raise ValueError("k-means seed components must be nonnegative integers")
    generator = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(seeds)))
    best: KMeansResult | None = None
    best_signature: tuple[tuple[int, ...], ...] | None = None
    for initialization in range(n_init):
        candidate = _fit_kmeans_initialization(
            values,
            cluster_count,
            generator,
            initialization,
            maximum_iterations,
            convergence_tolerance,
        )
        if candidate is None:
            continue
        signature = _partition_signature(candidate.labels)
        if _candidate_is_better(candidate, signature, best, best_signature):
            best = candidate
            best_signature = signature
    if best is None:
        raise RuntimeError("no k-means initialization converged")
    return best


def canonical_cluster_labels(
    names: Sequence[str], labels: np.ndarray, *, prefix: str
) -> dict[str, str]:
    """Map transient integer labels to stable lexicographically ordered IDs."""

    ordered_names = [str(name) for name in names]
    values = np.asarray(labels, dtype=int).reshape(-1)
    if len(ordered_names) != len(values) or len(set(ordered_names)) != len(ordered_names):
        raise ValueError("names and labels must have equal length with unique names")
    partitions = sorted(
        tuple(sorted(ordered_names[index] for index in np.flatnonzero(values == cluster)))
        for cluster in np.unique(values)
    )
    if any(not members for members in partitions):
        raise ValueError("canonical labels cannot contain an empty cluster")
    return {
        member: f"{prefix}_{index:02d}"
        for index, members in enumerate(partitions, start=1)
        for member in members
    }


def labels_sha256(labels: dict[str, str]) -> str:
    payload = json.dumps(labels, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
