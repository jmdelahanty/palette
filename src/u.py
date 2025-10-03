import zarr
root = zarr.open('/nvme1/sesh1/2025-09-23T21-41-12Z_arena_4_chaser_arena4.zarr', mode='r')
assign_runs = root['id_assignment_runs']
latest = assign_runs.attrs.get('latest')
print(f"Latest: {latest}")

if latest:
    run = assign_runs[latest]
    print(f"\nRun attrs: {dict(run.attrs)}")
    print(f"\nArrays: {list(run.array_keys())}")
    
    # Check the arrays
    detection_ids = run['detection_ids']
    n_detections_per_mask = run['n_detections_per_mask']
    
    print(f"\ndetection_ids shape: {detection_ids.shape}")
    print(f"n_detections_per_mask shape: {n_detections_per_mask.shape}")
    print(f"\nSample detection_ids (first 10): {detection_ids[:10]}")