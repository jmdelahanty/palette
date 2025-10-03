import zarr
root = zarr.open_group('/nvme1/sesh1/2025-09-23T21-41-12Z_arena_4_chaser_arena4.zarr', mode='r')
assign_runs = root['id_assignment_runs']
latest = assign_runs.attrs.get('latest')
print(f"Latest: {latest}")

if latest:
    run = assign_runs[latest]
    print(f"Run group attrs: {dict(run.attrs)}")
    print(f"Arrays in run: {list(run.array_keys())}")
    print(f"Groups in run: {list(run.group_keys())}")