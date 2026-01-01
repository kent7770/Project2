# PCA-Based Object Tracking

A Principal Component Analysis (PCA) based object tracking system for image sequences. This implementation uses PCA to build appearance models that represent objects and enable robust tracking across frames with motion prediction, occlusion handling, and multi-scale search.

## Features

- **PCA Appearance Model**: Builds compact representations of object appearance using Principal Component Analysis
- **Multi-Scale Tracking**: Searches at multiple scales to handle object size changes
- **Motion Prediction**: Uses velocity estimation for better search region prediction
- **Occlusion Handling**: Detects and recovers from occlusions using reconstruction error
- **Multiple Similarity Metrics**: Euclidean, correlation, Mahalanobis, reconstruction error, or combined
- **HOG Features**: Optional Histogram of Oriented Gradients for more robust features
- **Adaptive Updates**: Confidence-based appearance model updates
- **Interactive Initialization**: Select objects interactively with OpenCV's ROI selector
- **Trajectory Visualization**: Draw object trajectories on frames
- **Evaluation Metrics**: Precision, success rate, IoU, center error against ground truth
- **Video Output**: Save results as MP4 video with confidence visualization

## Installation

1. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

This project uses the OTB100 object tracking benchmark.

- **Download**: [OTB100 (Google Drive)](https://drive.google.com/drive/u/0/folders/1I-pD7n7TZ-FNsEEYS_2yID4m9cwh-4bk)
- **Extract to**: `OTB100/` in the project root
- **Curated subset for quick testing**: David, Girl, FaceOcc1, Car4, Walking2, Basketball, Biker, Bolt

After extraction, run:
```bash
python main.py --input OTB100/David/img --ground-truth OTB100/David/groundtruth_rect.txt --display --fast
```

## Quick Start

### Interactive Tracking

```bash
python main.py --input path/to/images --display
```

### Track with Predefined Bounding Box

```bash
python main.py --input path/to/images --bbox "100,50,80,120" --output results/
```

### Evaluate Against Ground Truth

```bash
python main.py --input path/to/images --ground-truth gt.txt --output results/
```

### Included OTB100 Sequences (Curated Subset)

This project includes a small subset of OTB100 sequences for quick testing and demos:

- `OTB100/David`
- `OTB100/Girl`
- `OTB100/FaceOcc1`
- `OTB100/Car4`
- `OTB100/Walking2`
- `OTB100/Basketball`
- `OTB100/Biker`
- `OTB100/Bolt`

## Command Line Options

### Input/Output

| Option | Description |
|--------|-------------|
| `--input, -i` | Input directory containing image sequence (required) |
| `--output, -o` | Output directory for results |
| `--bbox, -b` | Initial bounding box as "x,y,width,height" |
| `--ground-truth, -gt` | Ground truth file for evaluation |

### Tracker Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `--patch-size H W` | 64 64 | Patch size for feature extraction |
| `--n-components` | auto | Number of PCA components |
| `--variance-threshold` | 0.95 | Variance threshold for auto components |
| `--search-radius` | 40 | Search radius in pixels |
| `--similarity` | combined | Similarity metric (see below) |
| `--update-rate` | 3 | Update model every N frames |
| `--learning-rate` | 0.05 | Learning rate for updates |
| `--occlusion-threshold` | 0.15 | Threshold for occlusion detection (lower = more lenient) |

### Similarity Metrics

- `euclidean`: Euclidean distance in PCA space
- `correlation`: Normalized correlation coefficient
- `mahalanobis`: Mahalanobis distance with variance weighting
- `reconstruction`: Based on PCA reconstruction error
- `combined`: Weighted combination of euclidean and reconstruction (recommended)

### Feature Options

| Option | Description |
|--------|-------------|
| `--use-hog` | Use HOG features instead of raw pixels |
| `--no-motion` | Disable motion prediction |
| `--no-multiscale` | Disable multi-scale search |
| `--fast` | **Enable fast mode (5-10x faster, good accuracy)** |

### Display/Save Options

| Option | Description |
|--------|-------------|
| `--display` | Display tracking in real-time |
| `--display-fps` | Limit display playback FPS (useful for slower machines) |
| `--display-scale` | Scale the display window (e.g., `1.5` or `2.0`) |
| `--no-save-frames` | Don't save individual frames |
| `--save-video` | Save result as MP4 video |
| `--fps` | FPS for output video (default: 30) |

### Memory Options

| Option | Description |
|--------|-------------|
| `--lazy-load` | Load images lazily (memory efficient) |
| `--max-frames` | Maximum frames to process |

## Examples

### Basic Tracking with Display

```bash
python main.py --input images/sequence --display
```

### Fast Mode (Recommended for Real-Time)

```bash
python main.py --input OTB100/Walking2/img --display --fast
```

### HOG Features with Custom Parameters

```bash
python main.py --input images/sequence --use-hog --n-components 30 --search-radius 50 --display
```

### Full Evaluation with Video Output

```bash
python main.py --input images/sequence \
    --ground-truth groundtruth.txt \
    --output results/ \
    --save-video \
    --similarity combined \
    --display
```

### Memory-Efficient Processing

```bash
python main.py --input large_sequence/ --lazy-load --max-frames 1000 --output results/
```

## Keyboard Controls (Display Mode)

| Key | Action |
|-----|--------|
| `q` | Quit tracking |
| `p` | Pause/Resume |
| `s` | Save current frame snapshot |

## Architecture

```
┌─────────────────┐
│  Image Sequence │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature        │ ─── Extract patches, normalize, optional HOG
│  Extractor      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PCA Appearance │ ─── Build/update PCA model, compute similarity
│  Model          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PCA Tracker    │ ─── Motion prediction, multi-scale search,
│                 │     occlusion handling, confidence tracking
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Results        │ ─── Bounding boxes, trajectory, video, metrics
└─────────────────┘
```

## How It Works

### Initialization
1. User selects or provides initial bounding box
2. Extract object patch and variations (position + scale jittering)
3. Build PCA model from collected patches
4. Initialize motion model and confidence tracking

### Tracking Loop
1. **Motion Prediction**: Predict next position using velocity estimate
2. **Candidate Generation**: Extract patches at multiple positions and scales around predicted position
3. **Similarity Computation**: Project candidates to PCA space, compute similarity with reference
4. **Best Match Selection**: Select candidate with highest similarity
5. **Occlusion Detection**: If similarity below threshold, mark as occluded
6. **Model Update**: Update appearance model if confident enough
7. **Motion Update**: Update velocity estimate from position change

### Occlusion Handling
- Uses reconstruction error to detect occlusions
- Maintains motion during short occlusions
- Expands search radius when recovering from occlusion
- Gives up after max_occlusion_frames (default: 10)

## Output Files

When `--output` is specified:

- `tracked_frame_XXXX.jpg`: Frames with bounding boxes and trajectory
- `tracking_results.txt`: Bounding box coordinates per frame
- `tracking_result.mp4`: Video (if `--save-video` is used)

## Ground Truth Format

Ground truth files should have one bounding box per line:
- Comma-separated: `x,y,width,height`
- Space-separated: `x y width height`

## Evaluation Metrics

When ground truth is provided, the following metrics are computed:

- **Precision**: Percentage of frames with center location error < 20 pixels
- **Success Rate**: Percentage of frames with IoU > 0.5
- **Average IoU**: Mean Intersection over Union
- **Average Center Error**: Mean Euclidean distance between centers

## Dependencies

- `numpy>=1.21.0`: Numerical computations
- `opencv-python>=4.5.0`: Image processing and visualization
- `scikit-learn>=1.0.0`: PCA implementation
- `matplotlib>=3.5.0`: Optional visualization
- `Pillow>=9.0.0`: Image I/O

## Algorithm Details

### PCA Appearance Model

The appearance model uses Principal Component Analysis to create a low-dimensional representation of the object's appearance:

1. Collect N patches from the object (initial + variations)
2. Flatten patches to feature vectors
3. Compute PCA to find principal components
4. Keep components explaining 95% of variance (or fixed number)
5. Project new patches to PCA space for matching

### Similarity Computation

The `combined` similarity metric (default) uses:

```
similarity = 0.7 * pca_distance_similarity + 0.3 * reconstruction_similarity
```

Where:
- `pca_distance_similarity = 1 / (1 + ||proj - ref||)`
- `reconstruction_similarity = 1 / (1 + normalized_reconstruction_error)`

### Motion Model

Simple velocity-based prediction with exponential moving average:

```
velocity = 0.7 * velocity + 0.3 * (current_position - previous_position)
predicted_position = current_position + velocity
```

## Tips for Best Results

1. **Patch Size**: Larger patches (96x96 or 128x128) for larger objects
2. **HOG Features**: Use `--use-hog` for objects with strong edges
3. **Search Radius**: Increase for fast-moving objects
4. **Update Rate**: Lower for stable objects, higher for changing appearance
5. **Occlusion Threshold**: Lower for stricter occlusion detection

## License

This project is provided for educational and research purposes.
