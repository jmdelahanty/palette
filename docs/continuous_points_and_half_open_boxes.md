# Continuous points and half-open bounding boxes

This guide explains two image-coordinate conventions used by Palette:

- continuous points for locations such as fish positions, centroids, eyes,
  keypoints, and contour samples; and
- half-open pixel-edge bounding boxes for rectangular extents, crops, and mask
  bounds.

These conventions answer different questions. A point answers **where is this
location?** A bounding box answers **which rectangular region is covered?**
Although both contain `x` and `y` numbers, they are not interchangeable.

## Start with the image array

An image with height `H` and width `W` is usually accessed as:

```python
pixel = image[row, column]
pixel = image[y, x]
```

The valid integer array indices are:

```text
x = 0, 1, ..., W - 1
y = 0, 1, ..., H - 1
```

Palette's source-camera image profile has its origin at the top left. Positive
`x` goes right and positive `y` goes down. This is an image coordinate system,
not the usual mathematical Cartesian system in which positive `y` goes up.

An array index identifies one sampled pixel. Image geometry also needs to
represent locations between samples and boundaries around sets of samples.
That is why it cannot use integer indices alone.

## Continuous points

A continuous point is a floating-point location in an image plane:

```text
(x, y) = (128.37, 941.82)
```

It may fall at a pixel sample location, between samples, or at the center of a
larger object. It has no width, height, or area. There is consequently no
question about whether a maximum endpoint is included.

Examples of point-valued data include:

- a fish centroid;
- a tracked position;
- the center of an eye ellipse;
- a pose keypoint;
- a point sampled from a tail spline; and
- the center derived from a detection box.

"Continuous" does not mean continuous in time. It means that the spatial
coordinate is not restricted to discrete integer array indices. The exact
relationship between a point coordinate and the source pixels is part of the
persisted coordinate-frame and transform authority; consumers should not infer
it from an array name alone.

## Half-open pixel-edge bounding boxes

Palette represents an image-aligned bounding box as:

```text
[x_min, y_min, x_max, y_max)
```

The opening bracket means that the minimum edges are included. The closing
parenthesis means that the maximum edges are excluded. More explicitly, the
box covers the region satisfying:

```text
x_min <= x < x_max
y_min <= y < y_max
```

For example:

```text
box = [10, 20, 13, 22)
```

covers columns 10, 11, and 12 and rows 20 and 21. Its dimensions are exactly:

```python
width = x_max - x_min    # 3
height = y_max - y_min  # 2
```

It maps directly to NumPy slicing:

```python
crop = image[20:22, 10:13]
```

The word **edge** is important. `x_max` and `y_max` describe outer boundaries,
not pixel indices. For a `4512 x 4512` image, the complete image box is:

```text
[0, 0, 4512, 4512)
```

`4512` is not a valid pixel index, but it is the valid outer edge after the
last pixel. This is why code must not validate a half-open box endpoint using
the rules for a point or an array index.

## A small visual example

Consider a one-row image containing four pixel cells:

```text
edge coordinate     0       1       2       3       4
                    | cell0 | cell1 | cell2 | cell3 |
full image box      [                               )
                    [0, 4)
```

The box `[1, 3)` contains `cell1` and `cell2`. The adjacent boxes `[0, 2)` and
`[2, 4)` meet at edge 2 without overlapping and without leaving a gap.

This composability is one reason half-open intervals are widely used in array
programming.

## Computing a continuous center from a box

The geometric center of a half-open box is a continuous point:

```python
center_x = (x_min + x_max) / 2
center_y = (y_min + y_max) / 2
```

For the earlier box:

```text
box    = [10, 20, 13, 22)
center = (11.5, 21.0)
```

The fractional `x` value is expected. A center is a geometric position, not a
request to access `image[21, 11.5]`.

Palette persists both the box and the derived center in some detection
products:

```text
bbox_norm_coords  normalized cx, cy, width, height
        |
        | sealed normalized-to-pixel projection
        v
bbox_img_xyxy     half-open source-camera pixel edges
        |
        | sealed exact midpoint derivation
        v
centers_img_xy    continuous source-camera points
```

The derivation records are digest-bound so a consumer can distinguish a
genuine persisted relationship from two arrays that merely contain
plausible-looking numbers.

## Deriving a half-open box from a binary mask

Suppose a mask contains foreground pixels at integer coordinates `xs` and
`ys`. The smallest half-open bounding box around the foreground is:

```python
x_min = xs.min()
y_min = ys.min()
x_max = xs.max() + 1
y_max = ys.max() + 1
```

The `+ 1` converts the last included pixel index into the excluded outer edge.
Forgetting it is a classic off-by-one error: the crop loses its final row and
column, and its reported width and height are each one pixel too small.

For an empty mask there is no valid foreground box. Palette should represent
that through its validity/QC contract rather than inventing a zero-area box
that looks real.

## Why an inclusive maximum is troublesome

An alternative convention sometimes found in older computer-vision data is:

```text
[x_min, y_min, x_max_inclusive, y_max_inclusive]
```

Under that convention, width becomes:

```python
width = x_max_inclusive - x_min + 1
```

Mixing inclusive and half-open conventions silently changes dimensions and
centers. For example, the three columns 10, 11, and 12 can be written as:

```text
inclusive: [10, 12]
half-open:  [10, 13)
```

If `[10, 12]` is accidentally interpreted as half-open, column 12 disappears.
If `[10, 13)` is accidentally interpreted as inclusive, a nonexistent fourth
column is implied.

Palette therefore records the convention rather than relying on field names
such as `xyxy` to imply it.

## Why points and boxes need separate frame authorities

The two geometries may share the same image dimensions, origin, and axis
directions, but their transformation rules are not always identical.

### Translation

For a simple crop translated back into its source image, both points and box
edges may receive the same `(x_offset, y_offset)`. Palette still validates the
two directed transform chains separately. Equal numerical offsets do not make
the coordinate authorities interchangeable.

### Resizing

When an image is resized, an edge-to-edge transform must preserve the outer
extent: the old right boundary maps to the new right boundary. A point/sample
transform may instead require center-aware alignment. Reusing an edge transform
as a point transform, or vice versa, can introduce a half-pixel shift.

### Cropping

A half-open box maps mechanically to an array slice. A point inside that crop
is transformed as a location. Treating a box endpoint as a point can wrongly
reject `x_max == W`; treating a point as an edge can shift overlays or derived
centers.

For these reasons, current Palette coordinate contracts persist distinct
source-camera authorities for:

```text
continuous point geometry
half-open pixel-edge bbox geometry
```

A reader verifies that both authorities refer to the same acquisition extent,
but it does not collapse them into one authority.

## How this applies to behavioral analysis

Different downstream calculations should consume the appropriate geometry:

| Quantity | Geometry |
|---|---|
| Fish position or centroid | Continuous point |
| Eye center | Continuous point |
| Pose keypoint | Continuous point |
| Tail spline sample | Continuous point |
| Detection extent | Half-open bbox |
| Crop extent | Half-open bbox |
| Mask component bounds | Half-open bbox |
| Distance between fish | Continuous points, after proving a common frame |
| Heading or velocity | Differences between time-aligned continuous points |
| Image extraction | Half-open bbox converted to an array slice |

Distance, heading, and velocity code should never be handed a box endpoint as
though it were an animal position. Conversely, crop extraction should not
round a continuous centroid and pretend that the result defines an extent.

## What schema v2 protects

In Palette's current canonical coordinate work, relevant v2 records make the
point/edge distinction explicit. They bind items such as:

- the coordinate profile and pixel convention;
- the exact source-camera acquisition extent;
- the direction and identity of transforms;
- observation-row and acquisition-frame identity;
- the array payload and derivation records; and
- cryptographic digests of those records.

This is more than descriptive metadata. Normal readers validate it and fail
closed when an expected authority is absent, ambiguous, stale, or from a
different archive.

Historical data must not simply be relabeled as v2 because its numbers look
reasonable. It needs either a versioned historical reader that verifies the
old semantics exactly or an explicit migration/new publication that proves the
new semantics from authoritative inputs.

## Common mistakes

1. Treating `x_max` or `y_max` as an included pixel index.
2. Rejecting `x_max == W` even though it is a valid outer edge.
3. Forgetting `+ 1` when deriving a box from the maximum foreground index.
4. Adding `+ 1` to a box that is already half-open.
5. Computing `width = x_max - x_min + 1` for a half-open box.
6. Rounding centroids or keypoints before analysis.
7. Applying a bbox-edge resize transform to point coordinates.
8. Inferring coordinate semantics from names such as `bbox`, `xyxy`, or
   `positions_px` without validating their persisted authority.
9. Mixing top-left/y-down image coordinates with mathematical y-up coordinates
   when calculating angles.
10. Assuming matching shapes or plausible numeric ranges prove that two arrays
    share a coordinate frame.

## A practical checklist

Before consuming image geometry, ask:

1. Is this a point, vector, bbox, contour, or other geometry?
2. Which coordinate frame owns it: ROI, source camera, canvas, arena, or body?
3. What is its pixel convention?
4. Are maximum box endpoints inclusive or half-open?
5. What are the axis directions and origin?
6. Which persisted transform produced it, and in which direction?
7. Does it share the exact row/time identity required by the other input?
8. Is its source and derivation authority sealed and valid?

If these questions cannot be answered from the persisted contract, Palette's
normal scientific readers should consider the geometry unavailable rather than
guessing.

## Repository references

The current implementation is centered in:

- `src/fisheye/shared/pixel_frame_authority.py` for typed pixel-frame and
  acquisition authority;
- `src/fisheye/shared/observation_coordinate_publication.py` for detection and
  crop point/bbox publication; and
- `src/fisheye/shared/subject_shape_coordinate_publication.py` for subject-shape
  point and bbox transformations.

The broader design and cross-repository implications are recorded in:

- `docs/diagnostics/coordinate_contract_audit_2026-07-19.md`; and
- `docs/diagnostics/crimson_coordinate_implementation_work_package_2026-07-19.md`.
