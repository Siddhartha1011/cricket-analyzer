# Cricket Cover Drive Analysis

Real-time biomechanical analysis of cricket cover drive technique using computer vision and pose estimation.

##  Features

### Original Version (`cover_drive_rt.py`)
- Basic MediaPipe pose detection
- Core biomechanical metrics (elbow angle, spine lean, head position)
- Real-time video processing with overlays
- YouTube video download support

### Enhanced Robust Version (`robust_cover_drive_rt.py`)  **Recommended**
- **Temporal smoothing** to handle MediaPipe tracking issues
- **Knee position estimation** when direct detection fails
- **Confidence-based validation** for cricket-specific poses
- **Quality assessment** and reporting
- **Enhanced visualizations** with reliability indicators
- **Comprehensive analysis reports**

##  Quick Start

### 1. Environment Setup

**Option A: Using pip**
```bash
# Create virtual environment
python -m venv cricket-analysis
source cricket-analysis/bin/activate  # On Windows: cricket-analysis\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

**Analyze a YouTube video:**
```bash
python robust_cover_drive_rt.py --source "https://youtube.com/shorts/vSX3IRxGnNY"
```

**Analyze local video file:**
```bash
python robust_cover_drive_rt.py --source "/path/to/your/video.mp4"
```

**With live preview:**
```bash
python robust_cover_drive_rt.py --source "your_video.mp4" --show 1
```

##  Output Files

All outputs are saved to `./output/` directory:

- **`robust_annotated.mp4`** - Video with pose analysis overlays
- **`robust_metrics.csv`** - Frame-by-frame biomechanical data
- **`robust_evaluation.json`** - Technique scores and recommendations  
- **`quality_report.json`** - Pose detection quality assessment

##  Configuration Options

```bash
python robust_cover_drive_rt.py \
    --source "video_url_or_path" \
    --stance auto \                    # {right, left, auto}
    --confidence_threshold 0.5 \       # 0.3-0.8, lower for difficult poses
    --smoothing_window 5 \             # 3-10, more smoothing = more stable
    --target_fps 30 \                  # Output video FPS
    --width 720 \                      # Output resolution
    --show 1                           # Live preview: 1=on, 0=off
```

### Parameter Recommendations

**For high-quality videos:**
- `--confidence_threshold 0.6`
- `--smoothing_window 3`

**For challenging poses/lighting:**
- `--confidence_threshold 0.4` 
- `--smoothing_window 7`
- `--width 480` (smaller = more stable detection)

**For real-time preview:**
- `--show 1`
- `--target_fps 15` (lower FPS for smoother preview)

##  Biomechanical Metrics

### Core Measurements
- **Front Elbow Angle** - Shoulder→Elbow→Wrist angle (ideal: 100-150°)
- **Front Knee Angle** - Hip→Knee→Ankle angle (ideal: 120-170°)
- **Spine Lean** - Body forward lean angle (ideal: 8-22°)
- **Head Position** - Head alignment over front knee (ideal: <0.2 hip-widths)
- **Foot Direction** - Front toe pointing angle (ideal: <25°)

### Quality Indicators
- **Pose Quality** - Overall pose detection confidence (0-1)
- **Joint Confidence** - Individual joint tracking reliability
- **Data Coverage** - Percentage of frames with valid measurements

##  Understanding Results

### Technique Scores (1-10 scale)
- **Swing Control** - Based on elbow angle consistency
- **Footwork** - Foot direction and knee stability combined
- **Head Position** - Head-over-knee alignment
- **Balance** - Spine lean and overall posture
- **Overall Stability** - Knee bend consistency

### Quality Report
- **Excellent (>0.8)** - Highly reliable measurements
- **Good (0.6-0.8)** - Generally trustworthy data
- **Fair (0.4-0.6)** - Use with caution
- **Poor (<0.4)** - Unreliable, consider different camera angle

##  Video Requirements

### Optimal Setup
- **Camera Position**: Side-on view, 45° angle to batting direction
- **Distance**: 3-5 meters from batter
- **Height**: Waist level of batter
- **Lighting**: Even, avoid strong shadows
- **Background**: Uncluttered, contrasting with player
- **Resolution**: 720p minimum, 1080p preferred

### Supported Formats
- **YouTube URLs** (shorts, regular videos)
- **Local files**: MP4, AVI, MOV, MKV
- **Frame rates**: 24-60 FPS (30 FPS optimal)

##  Troubleshooting

### Common Issues

**"No pose detected"**
- Check camera angle (side-on view works best)
- Improve lighting conditions
- Reduce `--confidence_threshold` to 0.3-0.4
- Try smaller resolution (`--width 480`)

**"Knee tracking unreliable"**
- Normal for cricket batting poses - algorithm uses estimation
- Check quality report for data reliability
- Consider multiple camera angles for critical analysis

**"Elbow angle calculation failed"**
- Ensure arms are visible throughout swing
- Avoid loose clothing that obscures arm movement
- Try higher resolution or better camera position

**Performance issues**
- Reduce `--width` (e.g., 480 or 360)
- Lower `--target_fps` (e.g., 15)
- Use `--show 0` for batch processing
- Close other applications

### MediaPipe Limitations
- Struggles with bent/crouched cricket poses
- Knee detection often fails during stride
- Requires clear view of full body
- Performance degrades with complex backgrounds

##  Cricket-Specific Notes

### Batting Stance Detection
- **Right-handed**: Front foot = left side
- **Left-handed**: Front foot = right side  
- **Auto mode**: Uses hip/shoulder positioning to determine stance

### Technique Analysis Focus
- **Contact phase** metrics most important
- **Stride phase** may have unreliable knee tracking
- **Follow-through** captures swing completion
- **Setup phase** good for posture assessment

### Coaching Applications
- **Technique comparison** across multiple shots
- **Progress tracking** over training sessions  
- **Specific drill feedback** (footwork, head position)
- **Injury prevention** (spine lean, knee stress indicators)

##  Analysis Workflow

### 1. Data Collection
```bash
# Process multiple videos
for video in videos/*.mp4; do
    python robust_cover_drive_rt.py --source "$video" --show 0
done
```

### 2. Batch Analysis
```python
import pandas as pd
import glob

# Load all metrics files
all_metrics = []
for file in glob.glob("output/*_metrics.csv"):
    df = pd.read_csv(file)
    df['session'] = file.split('/')[-1]
    all_metrics.append(df)

combined = pd.concat(all_metrics)
```

### 3. Progress Tracking
- Compare technique scores across sessions
- Monitor consistency metrics (standard deviation)
- Track improvement in specific areas
- Identify persistent technique issues

##  Assumptions & Limitations

### Assumptions
- **Single person** in frame during analysis
- **Side-on camera view** for optimal pose detection
- **Consistent lighting** throughout video
- **Standard cricket batting technique** (not switch hits, etc.)
- **Adult proportions** for knee estimation algorithm

### Technical Limitations
- **MediaPipe dependency** - inherits its pose detection limitations
- **2D analysis only** - no depth/3D information
- **Frame-by-frame** - no temporal context beyond smoothing window
- **Generic pose model** - not cricket-specific training data
- **Occlusion handling** - limited ability to handle blocked body parts

### Biomechanical Limitations
- **Simplified joint models** - treats joints as simple hinges
- **No force analysis** - only kinematic measurements
- **No equipment interaction** - bat position not tracked
- **Static thresholds** - optimal ranges may vary by individual
- **No comparison to professional standards** - uses general guidelines

### Data Quality Considerations
- **Confidence scores essential** - always check quality metrics
- **Multiple angles recommended** for critical analysis
- **Lighting conditions critical** for consistent results
- **Background complexity** affects pose detection accuracy
- **Clothing choice matters** - avoid loose/flowing garments

##  Citation & Credits

This project uses:
- **MediaPipe** (Google) for pose estimation
- **OpenCV** for video processing  
- **yt-dlp** for YouTube video download
- Cricket biomechanics principles from sports science literature

##  Support

For issues related to:
- **Installation problems**: Check dependencies and Python version
- **Video processing errors**: Verify video format and accessibility
- **Analysis quality**: Review camera setup and video requirements
- **Results interpretation**: Consult biomechanics literature or coaching resources

##  Version History

- **v1.0** - Basic pose analysis with MediaPipe
- **v2.0** - Enhanced robust analysis with quality assessment
- **v2.1** - Improved cricket-specific pose handling
- **Current** - Comprehensive analysis with detailed reporting

---

** Note**: This tool provides automated analysis to assist coaching and technique development. It should complement, not replace, qualified coaching expertise and professional biomechanical analysis.
