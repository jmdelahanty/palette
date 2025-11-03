#!/usr/bin/env python
"""Quick script to check heading data in keypoints run."""

import zarr
import numpy as np

root = zarr.open('/nvme1/sesh3/2025-09-23T22-11-11Z_arena_4_chaser_arena4.zarr/', mode='r')
kp = root['refined_keypoints_runs/refined_keypoints_2025-10-25_16-24-12']
heading = kp['heading'][:]

print(f'Total headings: {len(heading)}')
print(f'Finite headings: {np.isfinite(heading).sum()}')
print(f'NaN headings: {np.isnan(heading).sum()}')
print(f'Percentage NaN: {np.isnan(heading).sum() / len(heading) * 100:.1f}%')

# Check if refined_success exists
if 'refined_success' in kp:
    refined_success = kp['refined_success'][:]
    print(f'\nRefined success: {refined_success.sum()} / {len(refined_success)}')
    print(f'Percentage refined success: {refined_success.sum() / len(refined_success) * 100:.1f}%')

    # Check correlation
    has_heading = np.isfinite(heading)
    print(f'\nHeading available when refined_success=True: {(has_heading & refined_success).sum()} / {refined_success.sum()}')
    print(f'Heading available when refined_success=False: {(has_heading & ~refined_success).sum()} / {(~refined_success).sum()}')

print('\nFirst 10 headings:', heading[:10])
print('Indices with finite heading:', np.where(np.isfinite(heading))[0][:20])
