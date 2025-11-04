#!/usr/bin/env python3
"""
Test script to verify pose schema can be loaded and has the correct structure.
This validates the changes made to store and propagate pose_schema attributes.
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fisheye.pose.schema import schema_from_package, schema_from_metadata

def test_schema_loading():
    """Test that traditional_v1 schema can be loaded from package."""
    print("Testing schema_from_package('traditional_v1')...")

    schema = schema_from_package("traditional_v1")

    # Verify schema structure
    assert schema.name == "traditional_v1", f"Expected name 'traditional_v1', got '{schema.name}'"
    assert schema.num_keypoints == 3, f"Expected 3 keypoints, got {schema.num_keypoints}"
    assert schema.node_names == ["bladder", "eye_left", "eye_right"], \
        f"Unexpected node names: {schema.node_names}"
    assert schema.edges == [[0, 1], [0, 2]], \
        f"Unexpected edges: {schema.edges}"

    print(f"✓ Schema loaded successfully:")
    print(f"  Name: {schema.name}")
    print(f"  Nodes: {schema.node_names}")
    print(f"  Edges: {schema.edges}")
    print(f"  Num keypoints: {schema.num_keypoints}")

    return schema


def test_schema_metadata_format():
    """Test that schema can be serialized and reconstructed from metadata dict."""
    print("\nTesting schema serialization/deserialization...")

    # Load original schema
    original = schema_from_package("traditional_v1")

    # Create metadata dict (what we store in zarr attrs)
    metadata = {
        "name": original.name,
        "nodes": original.node_names,
        "edges": original.edges,
        "metadata": original.metadata,
        "source": "configs/fisheye/pose_schemas/traditional_v1.json"
    }

    print(f"  Metadata dict: {metadata}")

    # Reconstruct from metadata
    reconstructed = schema_from_metadata(metadata)

    # Verify reconstruction
    assert reconstructed.name == original.name
    assert reconstructed.node_names == original.node_names
    assert reconstructed.edges == original.edges
    assert reconstructed.num_keypoints == original.num_keypoints

    print(f"✓ Schema successfully reconstructed from metadata")
    print(f"  Reconstructed nodes: {reconstructed.node_names}")
    print(f"  Reconstructed edges: {reconstructed.edges}")


def test_schema_index_lookup():
    """Test that node index lookup works correctly."""
    print("\nTesting node index lookup...")

    schema = schema_from_package("traditional_v1")

    # Test index lookup
    bladder_idx = schema.index("bladder")
    eye_left_idx = schema.index("eye_left")
    eye_right_idx = schema.index("eye_right")

    assert bladder_idx == 0, f"Expected bladder at index 0, got {bladder_idx}"
    assert eye_left_idx == 1, f"Expected eye_left at index 1, got {eye_left_idx}"
    assert eye_right_idx == 2, f"Expected eye_right at index 2, got {eye_right_idx}"

    print(f"✓ Node lookup works correctly:")
    print(f"  bladder -> {bladder_idx}")
    print(f"  eye_left -> {eye_left_idx}")
    print(f"  eye_right -> {eye_right_idx}")


if __name__ == "__main__":
    try:
        test_schema_loading()
        test_schema_metadata_format()
        test_schema_index_lookup()
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nThe pose schema can be:")
        print("  1. Loaded from package files")
        print("  2. Serialized to zarr attributes")
        print("  3. Reconstructed from metadata")
        print("  4. Used for node index lookups")
        print("\nYour visualization agent can load schema from zarr like this:")
        print("  schema_meta = keypoint_group.attrs.get('pose_schema')")
        print("  schema = schema_from_metadata(schema_meta)")
        print("  labels = schema.node_names")
        print("  edges = schema.edges")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
