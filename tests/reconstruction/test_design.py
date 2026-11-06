"""Blocs de colonnes : troncature par ordre, alignement temporel, poolage, cache."""

from __future__ import annotations

import numpy as np
import pytest

from qgf.reconstruction.design import (
    TIME_STATIC,
    DesignBlock,
    build_pooled_design,
    build_sample_design,
    cached_pooled_block,
    check_finite,
    concat_blocks,
    pooled_groups,
    pooled_targets,
    resolve_family,
    stack_blocks,
    with_constant,
)


def _block(num_rows: int = 4, prefix: str = "a") -> DesignBlock:
    matrix = np.arange(num_rows * 3, dtype=float).reshape(num_rows, 3)
    return DesignBlock(
        matrix=matrix,
        labels=(f"{prefix}_0", f"{prefix}_1", f"{prefix}_2"),
        orders=np.array([0, 1, 2]),
    )


def test_block_rejects_inconsistent_metadata():
    with pytest.raises(ValueError, match="labels"):
        DesignBlock(matrix=np.zeros((2, 3)), labels=("a",), orders=np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="orders"):
        DesignBlock(matrix=np.zeros((2, 2)), labels=("a", "b"), orders=np.array([0]))


def test_default_time_index_is_static():
    block = _block()
    assert np.all(block.time_index == TIME_STATIC)
    assert not block.has_time_columns


def test_select_keeps_columns_up_to_order():
    block = _block()
    assert block.select(0).labels == ("a_0",)
    assert block.select(1).labels == ("a_0", "a_1")
    assert block.select(9).num_columns == 3


def test_select_time_keeps_static_and_matching_columns():
    """A time-aligned feature exposes only one column per time step."""
    matrix = np.arange(8, dtype=float).reshape(2, 4)
    block = DesignBlock(
        matrix=matrix,
        labels=("static", "t0", "t1", "t2"),
        orders=np.zeros(4, dtype=np.int64),
        time_index=np.array([TIME_STATIC, 0, 1, 2]),
    )
    assert block.has_time_columns
    assert block.select_time(1).labels == ("static", "t1")
    assert block.select_time(5).labels == ("static",)


def test_concat_and_stack_propagate_metadata():
    left, right = _block(prefix="a"), _block(prefix="b")
    wide = concat_blocks([left, right])
    assert wide.num_columns == 6
    assert wide.labels[3] == "b_0"
    tall = stack_blocks([left, left])
    assert tall.num_rows == 8
    assert tall.labels == left.labels


def test_stack_rejects_heterogeneous_columns():
    with pytest.raises(ValueError, match="Heterogeneous"):
        stack_blocks([_block(prefix="a"), _block(prefix="b")])


def test_concat_rejects_heterogeneous_rows():
    with pytest.raises(ValueError, match="different heights"):
        concat_blocks([_block(num_rows=3), _block(num_rows=4)])


def test_with_constant_prepends_ones():
    block = with_constant(_block())
    assert block.labels[0] == "const"
    assert np.all(block.matrix[:, 0] == 1.0)


def test_check_finite_reports_offending_column():
    block = DesignBlock(
        matrix=np.array([[1.0, np.inf]]),
        labels=("ok", "overflow"),
        orders=np.array([0, 1]),
    )
    with pytest.raises(FloatingPointError, match="overflow"):
        check_finite(block, "test")


def test_resolve_family_parses_composites():
    assert resolve_family("walk") == ("walk",)
    assert resolve_family("walk+laplacian") == ("walk", "laplacian")
    with pytest.raises(ValueError, match="Unknown families"):
        resolve_family("walk+nope")


def test_pooled_groups_and_targets_align(tiny_samples):
    groups = pooled_groups(tiny_samples)
    targets = pooled_targets(tiny_samples)
    assert groups.size == targets.shape[0]
    assert groups.size == sum(sample.num_pairs for sample in tiny_samples)
    assert set(np.unique(groups)) == {sample.graph_id for sample in tiny_samples}


def test_pooled_design_rows_match_pair_total(tiny_samples, context):
    design = build_pooled_design(
        tiny_samples,
        "walk+laplacian",
        context,
        dataset="TEST",
        cache_dir=None,
    )
    assert design.num_rows == sum(sample.num_pairs for sample in tiny_samples)
    assert design.num_columns == (context.k_max + 1) + 3 * context.k_max


def test_pooled_design_row_order_follows_samples(tiny_samples, context):
    design = build_pooled_design(tiny_samples, "walk", context, dataset="TEST", cache_dir=None)
    offset = 0
    for sample in tiny_samples:
        local = build_sample_design(sample, "walk", context)
        assert np.allclose(design.matrix[offset : offset + sample.num_pairs], local.matrix)
        offset += sample.num_pairs


def test_design_cache_round_trip(tiny_samples, context, tmp_path):
    """The disk cache returns exactly the same design."""
    direct = cached_pooled_block(tiny_samples, "walk", context, "TEST", cache_dir=None)
    written = cached_pooled_block(
        tiny_samples,
        "walk",
        context,
        "TEST",
        cache_dir=str(tmp_path),
    )
    reloaded = cached_pooled_block(
        tiny_samples,
        "walk",
        context,
        "TEST",
        cache_dir=str(tmp_path),
    )
    assert np.allclose(direct.matrix, reloaded.matrix)
    assert written.labels == reloaded.labels
    assert np.array_equal(written.orders, reloaded.orders)


def test_design_cache_skips_oversized_blocks(tiny_samples, context, tmp_path):
    block = cached_pooled_block(
        tiny_samples,
        "walk",
        context,
        "TEST",
        cache_dir=str(tmp_path),
        max_mb=0.0,
    )
    assert block.num_rows > 0
    assert not list(tmp_path.rglob("*.npz"))


def test_quench_sensitive_bases_get_distinct_cache_entries(tiny_samples, tmp_path):
    """The two-body design depends on the quench: its cache must separate them."""
    from qgf.reconstruction.config import AFM_QUENCH, FM_QUENCH, BasisContext
    from qgf.reconstruction.design import is_quench_sensitive

    assert is_quench_sensitive("twobody")
    assert is_quench_sensitive("twobody_curve")
    assert not is_quench_sensitive("walk")

    times = (100.0, 200.0, 300.0, 400.0)
    afm = cached_pooled_block(
        tiny_samples,
        "twobody",
        BasisContext(k_max=1, quench=AFM_QUENCH, times_ns=times),
        "TEST",
        cache_dir=str(tmp_path),
    )
    fm = cached_pooled_block(
        tiny_samples,
        "twobody",
        BasisContext(k_max=1, quench=FM_QUENCH, times_ns=times),
        "TEST",
        cache_dir=str(tmp_path),
    )
    assert len(list(tmp_path.rglob("*.npz"))) == 2
    assert not np.allclose(afm.matrix, fm.matrix)
