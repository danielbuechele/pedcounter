import cv2
import numpy as np
import argparse
import sqlite3
import time
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass
from ultralytics import YOLO
from ultralytics.engine.results import Results
from cuid2 import cuid_wrapper
from picamera2 import Picamera2
import time
from dotenv import load_dotenv
import psycopg2
from psycopg2 import pool

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"


# Define the counting line in normalized coordinates (0-1)
# (x1, y1), (x2, y2) where x and y are between 0 and 1
LINE_NORMALIZED: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.5, 0), (0.5, 1))

# Initialize CUID generator
cuid_generator = cuid_wrapper()

# Create output directory for PNG files
OUTPUT_DIR = "logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Type definitions
CrossingState = Literal['crossing', 'crossed_left', 'crossed_right']
EventType = Literal['in', 'out']

@dataclass
class TrackState:
    state: CrossingState
    first_seen: float

@dataclass
class Track:
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    track_id: int

@dataclass
class Counts:
    in_count: int = 0
    out_count: int = 0
    
    def increment(self, event_type: EventType) -> None:
        """Increment the appropriate counter"""
        if event_type == 'in':
            self.in_count += 1
        else:
            self.out_count += 1

# Type aliases
TrackStates = Dict[int, TrackState]
Tracks = List[Track]
Line = Tuple[Tuple[int, int], Tuple[int, int]]
NormalizedCoords = Tuple[float, float]
BoundingBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)

# Global state variables
db_conn: Optional[sqlite3.Connection] = None
track_states: TrackStates = {}
counts: Counts = Counts()
tracks: Tracks = []
line: Line = ((0, 0), (0, 0))
frame: Optional[np.ndarray] = None
results: Optional[List[Results]] = None
left_is_in: bool = True

def init_neon_connection_pool() -> None:
    """Initialize the Neon connection pool"""
    load_dotenv()

    connection_string = os.getenv('DATABASE_URL')

    try:
        global connection_pool
        connection_pool = pool.SimpleConnectionPool(
            1, 10, connection_string
        )
        if connection_pool:
            print("Neon connection pool created successfully")
    except Exception as e:
        print(f"Could not connect to Neon DB: {e}")
        connection_pool = None

def upload_unsynced_events():
    global db_conn
    if not db_conn or not connection_pool:
        return

    cursor = db_conn.cursor()
    cursor.execute("SELECT id, timestamp, track_id, confidence, event_type, duration_seconds FROM events WHERE uploaded = 0")
    rows = cursor.fetchall()

    for row in rows:
        event = {
            'id': row[0],
            'timestamp': row[1],
            'track_id': row[2],
            'confidence': row[3],
            'event_type': row[4],
            'duration_seconds': row[5]
        }
        success = push_event_to_neon(event)
        if success:
            cursor.execute("UPDATE events SET uploaded = 1 WHERE id = ?", (event['id'],))
            db_conn.commit()
            print(f"uploaded: {event['id'][:8]}...")
        else:
            print(f"upload failed: {event['id'][:8]}...")


def close_neon_connection_pool() -> None:
    """Close the Neon connection pool"""
    if connection_pool:
        connection_pool.closeall()
        print("Neon connection pool closed")

def init_database() -> None:
    """Initialize SQLite database and create events table"""
    global db_conn
    db_conn = sqlite3.connect('logs/_logs.db')
    cursor = db_conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            timestamp DATETIME NOT NULL,
            track_id INTEGER NOT NULL,
            confidence REAL NOT NULL,
            event_type TEXT NOT NULL,
            duration_seconds REAL,
            uploaded BOOLEAN DEFAULT 0
        )
    ''')
    
    db_conn.commit()
    print("Database initialized: logs/_logs.db")

def close_database() -> None:
    """Close the database connection"""
    global db_conn
    if db_conn:
        db_conn.close()
        print("Database connection closed")

def push_event_to_neon(event: dict) -> bool:
    """Try pushing an event to the Neon database"""
    global connection_pool
    if not connection_pool:
        return False
    try:
        conn = connection_pool.getconn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO events (id, timestamp, track_id, confidence, event_type, duration_seconds)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            event['id'], event['timestamp'], event['track_id'], event['confidence'],
            event['event_type'], event['duration_seconds']
        ))
        conn.commit()
        cur.close()
        connection_pool.putconn(conn)
        return True
    except Exception as e:
        print(f"❌ Neon push failed: {e}")
        return False

def save_event_snapshot(event_id: str, trigger_track_id: int) -> None:
    """Save a PNG snapshot of the visualization for an event"""
    global frame, tracks, line, counts, track_states
    if frame is None:
        return
    vis_frame = draw_visualization(frame.copy(), trigger_track_id)
    filename = os.path.join(OUTPUT_DIR, f"{event_id}.png")
    cv2.imwrite(filename, vis_frame)
    print(f"Saved event snapshot: {filename}")

def log_event(track_id: int, confidence: float, event_type: EventType, duration_seconds: Optional[float] = None) -> None:
    global db_conn
    if not db_conn:
        return

    cursor = db_conn.cursor()
    timestamp = datetime.now().isoformat()
    event_id = cuid_generator()
    
    # try to push to neon
    event_data = {
        'id': event_id,
        'timestamp': timestamp,
        'track_id': track_id,
        'confidence': confidence,
        'event_type': event_type,
        'duration_seconds': duration_seconds
    }
    uploaded = push_event_to_neon(event_data)

    # save to local db
    cursor.execute('''
        INSERT INTO events (id, timestamp, track_id, confidence, event_type, duration_seconds, uploaded)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (event_id, timestamp, track_id, confidence, event_type, duration_seconds, int(uploaded)))
    db_conn.commit()

    print(f"Logged {event_type} event {event_id[:8]}... (uploaded: {uploaded})")
    save_event_snapshot(event_id, track_id)


def normalized_to_pixel(normalized_coords: NormalizedCoords, frame_width: int, frame_height: int) -> Tuple[int, int]:
    """Convert normalized coordinates (0-1) to pixel coordinates"""
    x, y = normalized_coords
    pixel_x = int(x * frame_width)
    pixel_y = int(y * frame_height)
    return (pixel_x, pixel_y)

def is_bbox_left_of_line(bbox: BoundingBox, line: Line) -> bool:
    """Helper function to check if a bounding box is completely to the left or right of the line"""
    (x1, y1), (x2, y2) = line
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
    
    # For a vertical line, check if entire bbox is to the left
    if abs(x2 - x1) < abs(y2 - y1):  # Vertical line
        return bbox_x2 < x1  # Entire bbox is to the left
    else:  # Horizontal line
        return bbox_y2 < y1  # Entire bbox is above

def is_bbox_crossing_line(bbox: BoundingBox, line: Line) -> bool:
    """Helper function to check if a bounding box is crossing the line (partially on both sides)"""
    (x1, y1), (x2, y2) = line
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
    
    # For a vertical line, check if bbox spans the line
    if abs(x2 - x1) < abs(y2 - y1):  # Vertical line
        return bbox_x1 < x1 and bbox_x2 > x1  # Bbox spans the vertical line
    else:  # Horizontal line
        return bbox_y1 < y1 and bbox_y2 > y1  # Bbox spans the horizontal line

def draw_visualization(frame: np.ndarray, highlight_track_id: Optional[int] = None) -> np.ndarray:
    """Draw visualization with bounding boxes, line, counts, and track states"""
    global tracks, line, counts, track_states
    
    # Draw line
    cv2.line(frame, line[0], line[1], (0, 255, 255), 2)
    # Draw counts
    cv2.putText(frame, f"In: {counts.in_count}  Out: {counts.out_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    # Draw bounding boxes
    for track in tracks:
        # If highlight_track_id is specified, only draw that track
        if highlight_track_id is not None and track.track_id != highlight_track_id:
            continue
            
        x1, y1, x2, y2 = track.bbox
        track_id = track.track_id
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        
        # Get current state for this track
        track_id_int = int(track_id)
        track_state = track_states.get(track_id_int)
        current_state = track_state.state if track_state else 'unknown'
        label = f"{track_id}: {current_state}"
        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return frame

def get_track_confidence(track_id: int) -> float:
    """Get the confidence score for a specific track_id"""
    global results
    if results is None:
        return 0.0
    track_ids = results[0].boxes.id.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    track_mask = track_ids == track_id
    if track_mask.any():
        return float(confidences[track_mask].mean())
    return 0.0

def get_crossing_event_type(from_left: bool) -> EventType:
    """Determine if a crossing is an 'in' or 'out' event based on direction and configuration"""
    global left_is_in
    if from_left:
        # Moving from left to right
        return 'out' if left_is_in else 'in'
    else:
        # Moving from right to left
        return 'in' if left_is_in else 'out'

def handle_crossing_event(track_id: int, from_left: bool) -> None:
    """Handle a crossing event for either direction"""
    global counts, track_states
    event_type = get_crossing_event_type(from_left)
    counts.increment(event_type)
    
    direction_str = "left to right" if from_left else "right to left"
    print(f"Track {track_id} {'entered' if event_type == 'in' else 'exited'} (went {direction_str})")
    confidence = get_track_confidence(track_id)
    
    # Calculate duration since first seen
    duration_seconds: Optional[float] = None
    if track_id in track_states and hasattr(track_states[track_id], 'first_seen'):
        duration_seconds = time.time() - track_states[track_id].first_seen
    
    log_event(track_id, confidence, event_type, duration_seconds)

def main() -> None:
    global track_states, counts, tracks, line, frame, results, left_is_in
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, choices=['picamera', 'video'], default=None, help='Path to video file or picamera. If not set, webcam will be used.')
    parser.add_argument('--in-side', type=str, choices=['left', 'right'], default='left',
                       help='Which side counts as "in" (default: left)')
    parser.add_argument('--show-viz', action='store_true', help='Show visualization window (default: False)')
    args = parser.parse_args()

    # Update global configuration
    left_is_in = args.in_side == 'left'
    print(f"Configuration: {args.in_side.title()} side counts as 'in'")
    if args.show_viz:
        print("Visualization: Enabled")
    else:
        print("Visualization: Disabled (use --show-viz to enable)")
    

    try:
        #cap = cv2.VideoCapture(0 if args.video is None else args.video)
        if args.input == 'picamera':
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
            picam2.configure(config)
            picam2.start()
        elif args.input == 'video':
            cap = cv2.VideoCapture(args.video)
        else:
            cap = cv2.VideoCapture(0)

        model = YOLO('yolov8n.pt', verbose=False)  # Use YOLOv8 nano model for speed
        frame_count = 0
        MAX_FRAMES_TO_SAVE = 0
        saved_frames = 0


        while True:
            if args.input == 'picamera':
                frame = picam2.capture_array()
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Mirror the webcam horizontally
            frame = cv2.flip(frame, 1)

            # Convert normalized line coordinates to pixel coordinates
            if frame is not None:
                frame_height, frame_width = frame.shape[:2]
            line = (
                normalized_to_pixel(LINE_NORMALIZED[0], frame_width, frame_height),
                normalized_to_pixel(LINE_NORMALIZED[1], frame_width, frame_height)
            )

            # Run YOLOv8 detection with built-in ByteTrack
            results = model.track(
                frame, 
                persist=True, 
                conf=0.1, 
                tracker="bytetrack.yaml",
                classes=[0],  # Only detect persons (class 0)
                verbose=False
            )
            
            tracks = []
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    # Convert numpy scalars to Python scalars to avoid boolean ambiguity
                    tracks.append(Track(bbox=(float(x1), float(y1), float(x2), float(y2)), track_id=int(track_id)))

            # Count line crossings
            for track in tracks:
                track_id: int = track.track_id
                bbox: Tuple[float, float, float, float] = track.bbox
                
                # Determine current state
                current_state: CrossingState
                if is_bbox_crossing_line(bbox, line):
                    current_state = 'crossing'  # Still crossing, don't count yet
                elif is_bbox_left_of_line(bbox, line):
                    current_state = 'crossed_left'
                else:
                    current_state = 'crossed_right'
                
                # State transition logic - track last stable state and count transitions
                if track_id not in track_states:
                    track_states[track_id] = TrackState(state=current_state, first_seen=time.time())
                else:
                    prev_state = track_states[track_id].state
                    
                    # Only count when transitioning from a stable state to the opposite stable state
                    if prev_state == 'crossed_left' and current_state == 'crossed_right':
                        # Person moved from left to right
                        handle_crossing_event(track_id, True)
                    elif prev_state == 'crossed_right' and current_state == 'crossed_left':
                        # Person moved from right to left
                        handle_crossing_event(track_id, False)
                    
                    # Update state (ignore crossing state for tracking)
                    if current_state != 'crossing':
                        track_states[track_id].state = current_state

            # Draw visualization
            if args.show_viz and frame is not None:
                vis_frame = draw_visualization(frame.copy())
                cv2.imshow('Pedestrian Counter', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if args.input == 'picamera':
            picam2.stop()  # stop picamera
        if args.show_viz:
            cv2.destroyAllWindows()
        close_database()
        close_neon_connection_pool()
        print("Cleanup completed")

if __name__ == '__main__':
    init_database()
    init_neon_connection_pool()
    upload_unsynced_events()
    main() 

