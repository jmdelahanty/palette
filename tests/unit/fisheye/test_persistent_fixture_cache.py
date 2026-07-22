from __future__ import annotations

from pathlib import Path

from tests.persistent_fixture_cache import persistent_directory_fixture


def test_persistent_directory_fixture_reuses_valid_content(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("VERSION = 1\n", encoding="utf-8")
    builds: list[Path] = []

    def build(destination: Path) -> None:
        builds.append(destination)
        destination.mkdir()
        (destination / "value.txt").write_text("canonical\n", encoding="utf-8")

    def validate(destination: Path) -> None:
        assert (destination / "value.txt").read_text(encoding="utf-8") == "canonical\n"

    first = persistent_directory_fixture(
        namespace="example",
        schema_version="v1",
        source_paths=(source,),
        dependency_versions={"demo": "1"},
        build=build,
        validate=validate,
        cache_root=tmp_path / "cache",
    )
    second = persistent_directory_fixture(
        namespace="example",
        schema_version="v1",
        source_paths=(source,),
        dependency_versions={"demo": "1"},
        build=build,
        validate=validate,
        cache_root=tmp_path / "cache",
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.path == second.path
    assert len(builds) == 1


def test_persistent_directory_fixture_invalidates_source_and_corruption(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("VERSION = 1\n", encoding="utf-8")
    build_count = 0

    def build(destination: Path) -> None:
        nonlocal build_count
        build_count += 1
        destination.mkdir()
        (destination / "value.txt").write_text("canonical\n", encoding="utf-8")

    def resolve():
        return persistent_directory_fixture(
            namespace="example",
            schema_version="v1",
            source_paths=(source,),
            dependency_versions={"demo": "1"},
            build=build,
            validate=lambda path: (path / "value.txt").read_text(encoding="utf-8"),
            cache_root=tmp_path / "cache",
        )

    first = resolve()
    (first.path / "value.txt").write_text("corrupt\n", encoding="utf-8")
    repaired = resolve()
    source.write_text("VERSION = 2\n", encoding="utf-8")
    invalidated = resolve()

    assert repaired.cache_key == first.cache_key
    assert repaired.cache_hit is False
    assert invalidated.cache_key != first.cache_key
    assert invalidated.cache_hit is False
    assert build_count == 3
