#!/bin/bash
# Simple test focusing on the problematic last frames

VIDEO="$1"

if [ -z "$VIDEO" ]; then
    echo "Usage: bash test_last_frames_simple.sh /path/to/video.mp4"
    exit 1
fi

echo "=========================================="
echo "TESTING LAST FRAMES"
echo "=========================================="
echo "Video: $VIDEO"
echo ""

# Test 1: Extract last frame using time-based seeking (more reliable)
echo "1. Extracting last frame using time-based seek..."
timeout 10s ffmpeg -sseof -0.016 -i "$VIDEO" -frames:v 1 -f null - 2>&1 | tail -5
RESULT1=$?
if [ $RESULT1 -eq 0 ]; then
    echo "   ✓ Last frame decoded successfully"
elif [ $RESULT1 -eq 124 ]; then
    echo "   ❌ TIMEOUT - ffmpeg hung trying to decode last frame"
else
    echo "   ❌ Failed with error code $RESULT1"
fi
echo ""

# Test 2: Decode last 10 seconds
echo "2. Decoding last 10 seconds of video..."
timeout 30s ffmpeg -sseof -10 -i "$VIDEO" -f null - 2>&1 | grep "frame=" | tail -1
RESULT2=$?
if [ $RESULT2 -eq 0 ]; then
    echo "   ✓ Last 10 seconds decoded successfully"
elif [ $RESULT2 -eq 124 ]; then
    echo "   ❌ TIMEOUT - ffmpeg hung on last 10 seconds"
else
    echo "   ❌ Failed with error code $RESULT2"
fi
echo ""

# Test 3: Decode from specific frame near end
echo "3. Decoding from frame 45700 onwards (53 frames)..."
timeout 20s ffmpeg -i "$VIDEO" -vf "select='gte(n,45700)'" -vsync 0 -f null - 2>&1 | grep "frame=" | tail -1
RESULT3=$?
if [ $RESULT3 -eq 0 ]; then
    echo "   ✓ Frames 45700+ decoded successfully"
elif [ $RESULT3 -eq 124 ]; then
    echo "   ❌ TIMEOUT - ffmpeg hung on frames near end"
else
    echo "   ❌ Failed with error code $RESULT3"
fi
echo ""

# Test 4: Check for corruption by validating the entire file
echo "4. Running full file validation (fast check)..."
timeout 60s ffmpeg -v error -i "$VIDEO" -f null - 2>&1 | head -20
RESULT4=$?
if [ $RESULT4 -eq 0 ]; then
    echo "   ✓ No errors found in entire file"
elif [ $RESULT4 -eq 124 ]; then
    echo "   ❌ TIMEOUT during full validation"
else
    echo "   ⚠️  Some errors detected (see above)"
fi
echo ""

# Test 5: Get info about last GOP (Group of Pictures)
echo "5. Checking GOP structure near end..."
ffprobe -select_streams v:0 -show_frames -read_intervals %+#50 "$VIDEO" 2>&1 | grep -E "pict_type|pkt_pts_time" | tail -20
echo ""

echo "=========================================="
echo "SUMMARY"
echo "=========================================="
if [ $RESULT1 -eq 124 ] || [ $RESULT2 -eq 124 ] || [ $RESULT3 -eq 124 ]; then
    echo "❌ VIDEO FILE HAS ISSUES AT END"
    echo ""
    echo "FFmpeg (CPU decoder) hangs when trying to decode frames near EOF."
    echo "This confirms the video file itself has problems, not just decord."
    echo ""
    echo "Possible causes:"
    echo "  - Video recording was interrupted/incomplete"
    echo "  - File corruption at end of file"
    echo "  - HEVC encoding issue with last GOP"
    echo ""
    echo "RECOMMENDATION:"
    echo "  Re-encode the video to fix corruption:"
    echo "  ffmpeg -i $VIDEO -c:v libx265 -crf 23 -c:a copy fixed_video.mp4"
else
    echo "✓ VIDEO FILE IS OK"
    echo ""
    echo "FFmpeg can decode all frames including the last ones."
    echo "The issue is specifically with decord's GPU decoder."
    echo ""
    echo "RECOMMENDATION:"
    echo "  Use CPU decoding (--force-cpu) or fall back to CPU for last batch"
fi