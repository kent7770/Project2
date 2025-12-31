"""
Main application for PCA-based object tracking.
Processes image sequences and tracks objects using PCA appearance model.
"""

import argparse
import cv2
import numpy as np
import sys
from pathlib import Path
from src.pca_tracker import PCATracker, TrackingResult
from src.utils import (
    load_image_sequence,
    load_image_sequence_lazy,
    load_ground_truth,
    draw_bbox,
    draw_trajectory,
    save_tracking_results,
    create_tracking_video,
    display_frame,
    evaluate_tracking,
    get_bbox_center,
    print_progress
)


def select_bbox_interactive(frame: np.ndarray) -> tuple:
    """
    Interactive bounding box selection using OpenCV's built-in ROI selector.
    
    Args:
        frame: First frame image
    
    Returns:
        Bounding box as (x, y, width, height)
    """
    print("\n" + "="*50)
    print("OBJECT SELECTION")
    print("="*50)
    print("Draw a rectangle around the object to track.")
    print("Press ENTER or SPACE to confirm selection.")
    print("Press 'c' to cancel and redraw.")
    print("Press 'q' to quit.")
    print("="*50 + "\n")
    
    # Use OpenCV's selectROI for better UX
    window_name = "Select Object to Track"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Resize window if frame is large
    h, w = frame.shape[:2]
    if w > 1280 or h > 720:
        scale = min(1280 / w, 720 / h)
        cv2.resizeWindow(window_name, int(w * scale), int(h * scale))
    
    bbox = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)
    
    if bbox[2] == 0 or bbox[3] == 0:
        return None
    
    return tuple(int(v) for v in bbox)


def run_tracking(args) -> dict:
    """
    Run the tracking pipeline.
    
    Args:
        args: Command line arguments
    
    Returns:
        Dictionary with tracking results and metrics
    """
    results = {
        'success': False,
        'frames_processed': 0,
        'bboxes': [],
        'confidences': [],
        'metrics': None
    }
    
    # Load image sequence
    print(f"\nLoading image sequence from: {args.input}")
    
    try:
        if args.lazy_load:
            # Count frames first
            images_list = list(load_image_sequence_lazy(args.input))
            if len(images_list) == 0:
                print("Error: No images found in directory")
                return results
            images = images_list
        else:
            images = load_image_sequence(args.input, max_frames=args.max_frames)
            if len(images) == 0:
                print("Error: No images found in directory")
                return results
        
        print(f"Loaded {len(images)} images")
        print(f"Frame size: {images[0].shape[1]}x{images[0].shape[0]}")
        
    except Exception as e:
        print(f"Error loading images: {e}")
        return results
    
    # Load ground truth if provided
    ground_truth = None
    if args.ground_truth:
        try:
            ground_truth = load_ground_truth(args.ground_truth)
            print(f"Loaded {len(ground_truth)} ground truth annotations")
        except Exception as e:
            print(f"Warning: Could not load ground truth: {e}")
    
    # Get initial bounding box
    first_frame = images[0]
    
    if args.bbox:
        # Parse bbox from command line
        try:
            bbox_values = [int(x) for x in args.bbox.split(',')]
            if len(bbox_values) != 4:
                raise ValueError("Bbox must have 4 values")
            initial_bbox = tuple(bbox_values)
        except Exception as e:
            print(f"Error parsing bbox: {e}")
            return results
    elif ground_truth and len(ground_truth) > 0:
        # Use first ground truth as initial bbox
        initial_bbox = ground_truth[0]
        print("Using first ground truth annotation as initial bbox")
    else:
        # Interactive selection
        initial_bbox = select_bbox_interactive(first_frame)
        if initial_bbox is None:
            print("Error: No bounding box selected")
            return results
    
    print(f"Initial bounding box: x={initial_bbox[0]}, y={initial_bbox[1]}, "
          f"w={initial_bbox[2]}, h={initial_bbox[3]}")
    
    # Initialize tracker
    print("\nInitializing PCA tracker...")
    
    tracker = PCATracker(
        patch_size=tuple(args.patch_size),
        n_components=args.n_components,
        variance_threshold=args.variance_threshold,
        search_radius=args.search_radius,
        similarity_method=args.similarity,
        update_rate=args.update_rate,
        learning_rate=args.learning_rate,
        occlusion_threshold=args.occlusion_threshold,
        use_motion_model=not args.no_motion,
        use_multiscale=not args.no_multiscale,
        use_hog=args.use_hog
    )
    
    success = tracker.initialize(first_frame, initial_bbox)
    if not success:
        print("Error: Failed to initialize tracker")
        return results
    
    print("Tracker initialized successfully!")
    explained_variance = tracker.appearance_model.get_explained_variance()
    n_components = tracker.appearance_model.get_n_components()
    print(f"PCA Model: {n_components} components, {explained_variance:.1%} variance explained")
    
    # Track through sequence
    print("\n" + "="*50)
    print("TRACKING IN PROGRESS")
    print("="*50)
    
    if args.display:
        print("Press 'q' to stop, 'p' to pause/resume, 's' to save current frame")
    
    tracked_bboxes = []
    confidences = []
    trajectory = []
    paused = False
    
    for i, frame in enumerate(images):
        if i == 0:
            # First frame - use initial bbox
            result = TrackingResult(
                bbox=initial_bbox,
                confidence=1.0,
                similarity=1.0,
                is_occluded=False,
                scale=1.0
            )
        else:
            # Update tracker
            result = tracker.update(frame)
        
        if result is not None:
            tracked_bboxes.append(result.bbox)
            confidences.append(result.confidence)
            
            center = get_bbox_center(result.bbox)
            trajectory.append(center)
        else:
            tracked_bboxes.append(None)
            confidences.append(0.0)
        
        # Display if requested
        if args.display:
            if result is not None and result.bbox is not None:
                # Color based on confidence
                conf = result.confidence
                color = (0, int(255 * conf), int(255 * (1 - conf)))
                
                frame_display = draw_bbox(
                    frame, result.bbox, 
                    color=color,
                    label=f"Frame {i+1}",
                    confidence=conf
                )
                
                # Draw trajectory
                if len(trajectory) > 1:
                    frame_display = draw_trajectory(frame_display, trajectory)
                
                # Add status overlay
                status = "TRACKING" if not result.is_occluded else "OCCLUDED"
                status_color = (0, 255, 0) if not result.is_occluded else (0, 165, 255)
                cv2.putText(frame_display, status, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            else:
                frame_display = frame.copy()
                cv2.putText(frame_display, "LOST", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            key = display_frame(frame_display, wait_key=1 if not paused else 0)
            
            if key == ord('q'):
                print("\nTracking stopped by user")
                break
            elif key == ord('p'):
                paused = not paused
            elif key == ord('s') and args.output:
                save_path = Path(args.output) / f"snapshot_{i+1:04d}.jpg"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), frame_display)
                print(f"\nSaved snapshot: {save_path}")
        
        # Progress
        if not args.display or (i + 1) % 10 == 0:
            extra = ""
            if result is not None:
                extra = f"conf={result.confidence:.2f}"
                if result.is_occluded:
                    extra += " [OCCLUDED]"
            print_progress(i + 1, len(images), "Tracking", extra_info=extra)
    
    print("\n" + "="*50)
    print("TRACKING COMPLETE")
    print("="*50)
    
    # Cleanup display
    if args.display:
        cv2.destroyAllWindows()
    
    # Update results
    results['success'] = True
    results['frames_processed'] = len(tracked_bboxes)
    results['bboxes'] = tracked_bboxes
    results['confidences'] = confidences
    
    # Evaluate if ground truth available
    if ground_truth:
        metrics = evaluate_tracking(tracked_bboxes, ground_truth)
        results['metrics'] = metrics
        
        print("\nEvaluation Metrics:")
        print(f"  Precision (CLE < 20px): {metrics.precision:.1%}")
        print(f"  Success Rate (IoU > 0.5): {metrics.success_rate:.1%}")
        print(f"  Average IoU: {metrics.avg_iou:.3f}")
        print(f"  Average Center Error: {metrics.avg_center_error:.1f}px")
        print(f"  Frames Tracked: {metrics.frames_tracked}")
        print(f"  Frames Lost: {metrics.frames_lost}")
    
    # Summary
    valid_tracks = sum(1 for bbox in tracked_bboxes if bbox is not None)
    avg_confidence = np.mean([c for c in confidences if c > 0]) if confidences else 0
    
    print(f"\nSummary:")
    print(f"  Frames Processed: {len(tracked_bboxes)}")
    print(f"  Frames Tracked: {valid_tracks} ({valid_tracks/len(tracked_bboxes)*100:.1f}%)")
    print(f"  Average Confidence: {avg_confidence:.2f}")
    
    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving results to: {args.output}")
        
        # Save tracked frames
        if not args.no_save_frames:
            save_tracking_results(
                args.output, images, tracked_bboxes,
                draw_trajectory_on_frames=True
            )
            print("  - Saved tracked frames")
        
        # Save video
        if args.save_video:
            video_path = output_dir / "tracking_result.mp4"
            create_tracking_video(
                images, tracked_bboxes, str(video_path),
                fps=args.fps, show_trajectory=True, confidences=confidences
            )
            print(f"  - Saved video: {video_path}")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='PCA-based Object Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive tracking with display
  python main.py --input images/sequence --display
  
  # Track with predefined bounding box
  python main.py --input images/sequence --bbox "100,50,80,120" --output results/
  
  # Evaluate against ground truth
  python main.py --input images/sequence --ground-truth gt.txt --output results/
  
  # Use HOG features with custom parameters
  python main.py --input images/sequence --use-hog --n-components 30 --display
        """
    )
    
    # Input/Output
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input directory containing image sequence')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--bbox', '-b', type=str, default=None,
                       help='Initial bounding box as "x,y,width,height"')
    parser.add_argument('--ground-truth', '-gt', type=str, default=None,
                       help='Ground truth file for evaluation')
    
    # Tracker parameters
    parser.add_argument('--patch-size', type=int, nargs=2, default=[64, 64],
                       metavar=('H', 'W'), help='Patch size (default: 64 64)')
    parser.add_argument('--n-components', type=int, default=None,
                       help='Number of PCA components (default: auto)')
    parser.add_argument('--variance-threshold', type=float, default=0.95,
                       help='Variance threshold for PCA (default: 0.95)')
    parser.add_argument('--search-radius', type=int, default=30,
                       help='Search radius in pixels (default: 30)')
    parser.add_argument('--similarity', type=str, default='combined',
                       choices=['euclidean', 'correlation', 'mahalanobis', 'reconstruction', 'combined'],
                       help='Similarity metric (default: combined)')
    parser.add_argument('--update-rate', type=int, default=3,
                       help='Update appearance model every N frames (default: 3)')
    parser.add_argument('--learning-rate', type=float, default=0.05,
                       help='Learning rate for model updates (default: 0.05)')
    parser.add_argument('--occlusion-threshold', type=float, default=0.3,
                       help='Threshold for occlusion detection (default: 0.3)')
    
    # Feature options
    parser.add_argument('--use-hog', action='store_true',
                       help='Use HOG features instead of raw pixels')
    parser.add_argument('--no-motion', action='store_true',
                       help='Disable motion prediction')
    parser.add_argument('--no-multiscale', action='store_true',
                       help='Disable multi-scale search')
    
    # Display/Save options
    parser.add_argument('--display', action='store_true',
                       help='Display tracking results in real-time')
    parser.add_argument('--no-save-frames', action='store_true',
                       help='Do not save individual frames')
    parser.add_argument('--save-video', action='store_true',
                       help='Save tracking result as video')
    parser.add_argument('--fps', type=int, default=30,
                       help='FPS for output video (default: 30)')
    
    # Memory options
    parser.add_argument('--lazy-load', action='store_true',
                       help='Load images lazily (memory efficient)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process')
    
    args = parser.parse_args()
    
    # Run tracking
    results = run_tracking(args)
    
    if not results['success']:
        sys.exit(1)
    
    return results


if __name__ == '__main__':
    main()
