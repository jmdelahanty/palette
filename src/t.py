# In Python
import zarr
root = zarr.open_group('/nvme1/sesh1/2025-09-23T21-41-12Z_arena_4_chaser_arena4.zarr', mode='a')
# Delete the incomplete run
del root['id_assignment_runs/id_assignment_2025-10-03_02-09-32']
# Clear the latest pointer
root['id_assignment_runs'].attrs['latest'] = None