# test_detect_quality.py
from fisheye.refinement.detect_quality import analyze_detect_quality, save_quality_report

zarr_path = "/nvme1/sesh2/2025-09-23T21-56-53Z_arena_4_chaser_arena4.zarr"

# Run the analysis
report = analyze_detect_quality(
    zarr_path=zarr_path,
    run_name=None,  # Uses latest
    jump_threshold=100.0
)

# Print results
print("\n" + "="*60)
print("DETECTION QUALITY REPORT")
print("="*60)

print(f"\nSource run: {report['source_run']}")

print("\nCOVERAGE:")
cov = report['coverage']
print(f"  Total frames: {cov['total_frames']}")
print(f"  Frames with detections: {cov['present_frames']} ({cov['coverage_percent']:.1f}%)")
print(f"  Multi-detection frames: {cov['multi_detection_frames']}")

print("\nGAPS:")
gaps = cov['gaps']
print(f"  Total gaps: {gaps['total_count']}")
print(f"  Longest gap: {gaps['longest_gap']} frames")
print(f"  Mean gap: {gaps['mean_gap_size']:.1f} frames")
print(f"\n  By category:")
for cat, count in gaps['categories'].items():
    print(f"    {cat}: {count}")

print("\nARTIFACTS:")
art = report['artifacts']
print(f"  Islands: {len(art['islands'])}")
print(f"  Blips: {len(art['blips'])}")
print(f"  Jumps: {len(art['jumps'])}")
print(f"  Total: {art['total_artifacts']}")

print("\nBBOX VALIDATION:")
bbox = report['bbox_validation']
print(f"  Total bboxes: {bbox['total_bboxes']}")
print(f"  Out of range: {bbox['out_of_range']}")
print(f"  Size outliers: {bbox['size_outliers']}")
print(f"  Malformed: {bbox['malformed']}")
print(f"  Size CV: {bbox['size_cv']:.3f}")

print("\nQUALITY SCORE:")
score = report['quality_score']
print(f"  Coverage: {score['coverage_score']:.1f}/100")
print(f"  Artifact: {score['artifact_score']:.1f}/100")
print(f"  Bbox: {score['bbox_score']:.1f}/100")
print(f"  Overall: {score['overall_score']:.1f}/100")
print(f"  Grade: {score['grade']}")

# Save to zarr
saved_path = save_quality_report(zarr_path, report)
print(f"\n[Saved to: {saved_path}]")

# Show artifact frames for visualization
if report['artifacts']['total_artifacts'] > 0:
    print("\nARTIFACT FRAMES (for visualization):")
    if report['artifacts']['islands']:
        print(f"  Islands at frames: {report['artifacts']['islands']}")
    if report['artifacts']['blips']:
        print(f"  Blips at frames: {report['artifacts']['blips']}")
    if report['artifacts']['jumps']:
        print(f"  Jumps at frames: {report['artifacts']['jumps']}")