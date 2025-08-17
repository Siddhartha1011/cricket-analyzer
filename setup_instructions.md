# Quick Setup Guide

## 5-Minute Setup

### Step 1: Download Files
```bash
# Create project directory
mkdir cricket-analysis
cd cricket-analysis

# Save these files:
# - robust_cover_drive_rt.py (the main script)
# - requirements.txt
```

### Step 2: Install Dependencies

**Option A: Using pip (Recommended)**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```


### Step 3: Test Installation
```bash
# Quick test
python robust_cover_drive_rt.py --help
```

### Step 4: Run Analysis
```bash
# Basic usage
python robust_cover_drive_rt.py --source "https://youtube.com/shorts/vSX3IRxGnNY"

# With preview (press 'q' to quit)
python robust_cover_drive_rt.py --source "your_video.mp4" --show 1
```

##  Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All packages installed without errors
- [ ] Help command works
- [ ] `output/` directory created after first run
- [ ] Video analysis completes successfully

##  Common Setup Issues

**Import Error: cv2**
```bash
pip install opencv-python
```

**Import Error: mediapipe**  
```bash
pip install mediapipe
```

**YouTube download fails**
```bash
pip install --upgrade yt-dlp
```

**Permission errors (Windows)**
```bash
# Run as administrator or use:
pip install --user -r requirements.txt
```

**M1/M2 Mac issues**
```bash
# Use conda for better ARM support
conda install opencv mediapipe numpy pandas
pip install yt-dlp
```


