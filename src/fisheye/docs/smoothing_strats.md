Here are a few smoothing approaches you can adopt, from simplest to more sophisticated:

Moving Average / Rolling Mean
Average the last N samples for each eye before drawing. You can implement this with a fixed-size buffer (e.g. 5–15 frames depending on frame rate) or use pandas.Series.rolling().mean() if you’re working in Python.

Exponential Moving Average (EMA)
Maintain smoothed = α * new_value + (1 - α) * smoothed, where α (0–1) controls responsiveness. EMAs react quicker to sudden changes than a plain rolling window.

Median Filter
For short bursts of noise/outliers, take the median over a small window. It preserves step changes better than a mean but still knocks down spikes.

Savitzky–Golay Filter
If you want a smooth curve that preserves peaks/slopes, the SG filter fits a polynomial over a sliding window. Available in scipy.signal.savgol_filter.

Low-pass Butterworth / One-pole filter
For more control, treat the signal as time-series data and apply a low-pass filter (e.g., first-order IIR), tuned to pass the frequencies you care about.

Kalman Filter
If you have a motion model (e.g. assume smooth changes in eye angle), a simple 1D Kalman filter can fuse measurements and maintain an estimate with uncertainty handling.

Spline / Polynomial Fit for post-processing
For offline plots, fitting a spline or a low-order polynomial through the sequence can give a smooth curve, though it’s heavier for real-time use.

Remember to apply smoothing separately to left/right or to the derived vergence; also mask out invalid frames (valid_mask) before smoothing so grossly bad detections don’t bias the average.