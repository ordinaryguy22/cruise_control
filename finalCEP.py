import cv2
import numpy as np
import time
from collections import deque

### 1. CONFIGURATION CONSTANTS ###
# Core parameters
TARGET_RESOLUTION = (640, 360)
COVERAGE_HEIGHT_RATIO = 0.50
BAND_WIDTH = 10

# Line detection
HOUGH_THRESHOLD = 15
MIN_LINE_LENGTH = 100
MAX_LINE_GAP = 75

# Line stability
MAX_SLOPE_CHANGE = 10
MAX_INTERCEPT_CHANGE = 50
LEAST_SLOPE_CHANGE = 0.01
LEAST_INTERCEPT_CHANGE = 3

# Frame Reusal
LINE_REUSE_INTERVAL = 5

# Visualization
LANE_MASK_PADDING = 3
MAX_DEVIATION = 20
DEVIATION_COLOR = (255, 255, 0)
TURN_BOX_COLOR = (50, 50, 50)
DEVIATION_BOX_COLOR = (50, 50, 50)

# Object detection
OBJECT_DETECTION_THRESHOLD = 10
MIN_OBJECT_AREA = 20
OBJECT_COLOR = (0, 0, 255)

### 2. GLOBAL STATE ###
prev_left_line = None
prev_right_line = None
frame_counter = 0
line_history = deque(maxlen=5)
performance_metrics = {
    'total_time': 0,
    'function_times': {},
    'frame_count': 0,
    'fps': 0,
    'last_frame_time': 0
}


### MAIN PROCESSING CHAIN ###

def main(video_path):
    reset_global_state()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    cv2.namedWindow('Lane and Object Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Lane and Object Detection', *TARGET_RESOLUTION)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, TARGET_RESOLUTION)
        processed_frame = process_frame(frame_resized)

        cv2.imshow('Lane and Object Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nFinal Performance Analysis:")
    print(f"Total frames processed: {performance_metrics['frame_count']}")
    print(f"Average FPS: {performance_metrics['fps']:.1f}")

    total_time = sum(performance_metrics['function_times'].values())
    if total_time > 0:
        print("\nFunction Time Distribution:")
        for func_name, func_time in performance_metrics['function_times'].items():
            percentage = (func_time / total_time) * 100
            print(f"{func_name}: {percentage:.1f}% ({func_time:.1f}ms)")


def process_frame(frame):
    global prev_left_line, prev_right_line, frame_counter
    frame_counter += 1

    if frame_counter % LINE_REUSE_INTERVAL != 0 or prev_left_line is None or prev_right_line is None:
        mask = generate_final_mask(frame)
        new_left_line, new_right_line = detect_lane_lines(mask)
        left_line, right_line = get_stable_lines(new_left_line, new_right_line)

        if left_line is not None and right_line is not None:
            prev_left_line, prev_right_line = left_line, right_line
    else:
        left_line, right_line = prev_left_line, prev_right_line

    frame_height, frame_width = frame.shape[:2]
    left_points, right_points = extend_lines_to_coverage(
        prev_left_line, prev_right_line, frame_height, frame_width)

    deviation, direction, center_point = calculate_deviation(
        left_points, right_points, frame_width)

    lane_mask = create_lane_mask(frame.shape, left_points, right_points)
    detected_objects = detect_objects_in_lane(frame, lane_mask)
    shaded_frame = draw_lane_overlay(frame, left_points, right_points)

    if left_points is not None and right_points is not None:
        shaded_frame = draw_object_detection(shaded_frame, detected_objects)

    shaded_frame = draw_deviation_info(shaded_frame, deviation, direction, center_point)
    update_performance_metrics()
    display_performance_info(shaded_frame)

    return shaded_frame


def generate_final_mask(frame):
    """Generate binary mask highlighting lane markings"""
    frame_resized = cv2.resize(frame, TARGET_RESOLUTION)
    height = frame_resized.shape[0]
    coverage_start = int(height * (1 - COVERAGE_HEIGHT_RATIO))
    coverage_roi = frame_resized[coverage_start:, :]

    hsv_roi = cv2.cvtColor(coverage_roi, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([15, 20, 200])
    upper_yellow = np.array([25, 55, 255])
    lower_white = np.array([130, 3, 200])
    upper_white = np.array([140, 7, 255])

    yellow_mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow) | \
                  cv2.inRange(hsv_roi, lower_white, upper_white)

    restricted_mask = restrict_white_pixel_band_center(yellow_mask)

    final_mask = np.zeros_like(frame_resized[:, :, 0])
    final_mask[coverage_start:, :] = restricted_mask

    return final_mask


def restrict_white_pixel_band_center(image, band_width=BAND_WIDTH):
    """Keeps white pixels within ±band_width of center white pixels"""
    height, width = image.shape
    output = np.zeros_like(image)
    center = width // 2
    white_mask = (image == 255)
    cols = np.arange(width)

    left_cols = cols[:center + 1][::-1]
    left_white = np.argmax(white_mask[:, left_cols], axis=1)
    left_white[left_white == 0] = center + 1
    left_white = center - left_white

    right_cols = cols[center:]
    right_white = np.argmax(white_mask[:, right_cols], axis=1)
    right_white[right_white == 0] = width - center + 1
    right_white = center + right_white

    x = np.arange(width)
    for y in range(height):
        if left_white[y] <= center:
            left_band = (x >= max(left_white[y] - band_width, 0)) & \
                        (x <= min(left_white[y] + band_width, width - 1))
            output[y] = np.where(left_band & white_mask[y], 255, output[y])

        if right_white[y] >= center and right_white[y] != left_white[y]:
            right_band = (x >= max(right_white[y] - band_width, 0)) & \
                         (x <= min(right_white[y] + band_width, width - 1))
            output[y] = np.where(right_band & white_mask[y], 255, output[y])

    return output


def detect_lane_lines(binary_mask):
    """Detect lane lines using Hough transform"""
    lines = cv2.HoughLinesP(binary_mask, 1, np.pi / 180,
                            threshold=HOUGH_THRESHOLD,
                            minLineLength=MIN_LINE_LENGTH,
                            maxLineGap=MAX_LINE_GAP)
    return average_slope_intercept(lines, binary_mask.shape)


def average_slope_intercept(lines, frame_shape):
    """Convert lines from Hough transform to average slope and intercept"""
    if lines is None:
        return None, None

    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < 1e-6:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))

    left_avg = np.mean(left_lines, axis=0) if left_lines else None
    right_avg = np.mean(right_lines, axis=0) if right_lines else None

    return left_avg, right_avg


def get_stable_lines(new_left_line, new_right_line):
    """Get stable lines using history and reuse strategy"""
    global prev_left_line, prev_right_line, frame_counter

    if new_left_line is not None and new_right_line is not None:
        line_history.append((new_left_line, new_right_line))

    if new_left_line is None or new_right_line is None:
        if line_history:
            return line_history[-1]
        return None, None

    if prev_left_line is None or prev_right_line is None:
        return new_left_line, new_right_line

    left_slope_change = abs(new_left_line[0] - prev_left_line[0])
    left_intercept_change = abs(new_left_line[1] - prev_left_line[1])
    right_slope_change = abs(new_right_line[0] - prev_right_line[0])
    right_intercept_change = abs(new_right_line[1] - prev_right_line[1])

    left_acceptable = (LEAST_SLOPE_CHANGE <= left_slope_change <= MAX_SLOPE_CHANGE and
                       LEAST_INTERCEPT_CHANGE <= left_intercept_change <= MAX_INTERCEPT_CHANGE)
    right_acceptable = (LEAST_SLOPE_CHANGE <= right_slope_change <= MAX_SLOPE_CHANGE and
                        LEAST_INTERCEPT_CHANGE <= right_intercept_change <= MAX_INTERCEPT_CHANGE)

    if left_acceptable and right_acceptable:
        return new_left_line, new_right_line
    else:
        return prev_left_line, prev_right_line


def extend_lines_to_coverage(left_line, right_line, frame_height, frame_width):
    """Extend both lines to cover specified portion of frame"""
    left_points = calculate_line_points(left_line, frame_height, frame_width)
    right_points = calculate_line_points(right_line, frame_height, frame_width)
    return left_points, right_points


def calculate_line_points(line_params, frame_height, frame_width):
    """Calculate line endpoints based on slope and intercept"""
    if line_params is None:
        return None

    m, b = line_params
    target_y = int(frame_height * (1 - COVERAGE_HEIGHT_RATIO))

    bottom_x = int((frame_height - 1 - b) / m)
    top_x = int((target_y - b) / m)

    return ((bottom_x, frame_height - 1), (top_x, target_y))


def create_lane_mask(frame_shape, left_points, right_points):
    """Create a binary mask of the lane area with padding"""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    if left_points is not None and right_points is not None:
        adjusted_left_bottom = (left_points[0][0] + LANE_MASK_PADDING, left_points[0][1] - LANE_MASK_PADDING)
        adjusted_right_bottom = (right_points[0][0] - LANE_MASK_PADDING, right_points[0][1] - LANE_MASK_PADDING)
        adjusted_right_top = (right_points[1][0] - LANE_MASK_PADDING, right_points[1][1] + LANE_MASK_PADDING)
        adjusted_left_top = (left_points[1][0] + LANE_MASK_PADDING, left_points[1][1] + LANE_MASK_PADDING)

        polygon_pts = np.array([
            adjusted_left_bottom, adjusted_right_bottom,
            adjusted_right_top, adjusted_left_top
        ])
        cv2.fillPoly(mask, [polygon_pts], 255)
    return mask


def detect_objects_in_lane(frame, lane_mask):
    """Detect objects within the lane area with improved filtering"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=lane_mask)
    thresh = cv2.adaptiveThreshold(masked_gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_objects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_OBJECT_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if 0.2 < aspect_ratio < 5:
                hull = cv2.convexHull(cnt)
                detected_objects.append((hull, (x, y, w, h)))

    return detected_objects


def draw_lane_overlay(frame, left_points, right_points):
    """Draw lane markings and overlay on frame"""
    frame_height, frame_width = frame.shape[:2]
    overlay = frame.copy()
    mask = np.zeros_like(frame)

    if left_points is None or right_points is None:
        return frame

    polygon_pts = np.array([
        left_points[0], right_points[0],
        right_points[1], left_points[1]
    ])

    cv2.fillPoly(mask, [polygon_pts], color=(0, 255, 0))
    cv2.line(overlay, left_points[0], left_points[1], (0, 0, 255), 2)
    cv2.line(overlay, right_points[0], right_points[1], (0, 0, 255), 2)
    cv2.line(overlay, (0, int(frame_height * (1 - COVERAGE_HEIGHT_RATIO))),
             (frame_width, int(frame_height * (1 - COVERAGE_HEIGHT_RATIO))),
             (255, 0, 255), 1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 1.0, mask, 0.4, 0)


def draw_object_detection(frame, objects):
    """Draw detected objects on the frame"""
    for hull, (x, y, w, h) in objects:
        cv2.drawContours(frame, [hull], -1, OBJECT_COLOR, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), OBJECT_COLOR, 2)
        cv2.putText(frame, "OBSTACLE", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, OBJECT_COLOR, 2)

    status = "STOP" if objects else "CLEAR"
    color = (0, 0, 255) if objects else (0, 255, 0)
    cv2.rectangle(frame, (frame.shape[1] - 150, 90), (frame.shape[1] - 10, 130), color, -1)
    cv2.putText(frame, status, (frame.shape[1] - 120, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame


def calculate_deviation(left_points, right_points, frame_width):
    """Calculate deviation from center and return direction advice"""
    if left_points is None or right_points is None:
        return 0, "No lanes", None

    top_center_x = (left_points[1][0] + right_points[1][0]) // 2
    top_center_y = left_points[1][1]
    frame_center_x = frame_width // 2
    deviation = top_center_x - frame_center_x
    abs_deviation = abs(deviation)

    if abs_deviation < MAX_DEVIATION:
        direction = "Center"
    elif deviation > 0:
        direction = "Move LEFT"
    else:
        direction = "Move RIGHT"

    return deviation, direction, (top_center_x, top_center_y)


def draw_deviation_info(frame, deviation, direction, center_point):
    """Draw deviation visualization and info boxes"""
    frame_height, frame_width = frame.shape[:2]

    cv2.line(frame, (frame_width // 2, 0), (frame_width // 2, frame_height),
             (0, 255, 255), 1, cv2.LINE_AA)

    if center_point is not None:
        cross_size = 10
        cv2.line(frame,
                 (center_point[0] - cross_size, center_point[1]),
                 (center_point[0] + cross_size, center_point[1]),
                 DEVIATION_COLOR, 2)
        cv2.line(frame,
                 (center_point[0], center_point[1] - cross_size),
                 (center_point[0], center_point[1] + cross_size),
                 DEVIATION_COLOR, 2)
        cv2.line(frame, center_point, (frame_width // 2, center_point[1]),
                 DEVIATION_COLOR, 1, cv2.LINE_AA)

    abs_deviation = abs(deviation)
    deviation_color = (0, 255, 0) if abs_deviation < MAX_DEVIATION else (0, 0, 255)
    cv2.rectangle(frame, (10, 60), (200, 100), (50, 50, 50), -1)
    cv2.putText(frame, f"Deviation: {abs_deviation:.1f} px",
                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, deviation_color, 2)

    box_width = 140
    box_height = 40
    right_margin = 10
    turn_box_top = 10

    if abs_deviation < MAX_DEVIATION:
        direction_text = "CENTER"
        direction_color = (0, 255, 0)
    elif deviation > 0:
        direction_text = "LEFT"
        direction_color = (0, 165, 255)
    else:
        direction_text = "RIGHT"
        direction_color = (0, 165, 255)

    cv2.rectangle(frame,
                  (frame_width - box_width - right_margin, turn_box_top),
                  (frame_width - right_margin, turn_box_top + box_height),
                  (50, 50, 50), -1)
    cv2.putText(frame, direction_text,
                (frame_width - box_width - right_margin + 30, turn_box_top + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, direction_color, 2)

    return frame


def update_performance_metrics():
    """Update and display performance metrics"""
    current_time = time.time()
    frame_time = (current_time - performance_metrics['last_frame_time']) * 1000
    performance_metrics['last_frame_time'] = current_time
    performance_metrics['frame_count'] += 1

    if performance_metrics['frame_count'] > 1:
        performance_metrics['fps'] = 1000 / frame_time


def display_performance_info(frame):
    """Display performance information on frame"""
    frame_height, frame_width = frame.shape[:2]

    cv2.rectangle(frame, (10, 10), (200, 50), (50, 50, 50), -1)
    cv2.putText(frame, f"FPS: {performance_metrics['fps']:.1f}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


### SUPPORTING FUNCTIONS ###

def reset_global_state():
    """Reset all global variables to their initial state"""
    global prev_left_line, prev_right_line, frame_counter, line_history, performance_metrics
    prev_left_line = None
    prev_right_line = None
    frame_counter = 0
    line_history = deque(maxlen=5)
    performance_metrics = {
        'total_time': 0,
        'function_times': {},
        'frame_count': 0,
        'fps': 0,
        'last_frame_time': 0
    }


def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = (time.time() - start_time) * 1000
        func_name = func.__name__
        performance_metrics['function_times'][func_name] = \
            performance_metrics['function_times'].get(func_name, 0) + elapsed
        return result

    return wrapper


def line_parameters_within_stable_range(new_line, prev_line):
    """Check if line parameters changed too little from previous frame"""
    if prev_line is None or new_line is None:
        return False

    new_slope, new_intercept = new_line
    prev_slope, prev_intercept = prev_line

    return (abs(new_slope - prev_slope) < LEAST_SLOPE_CHANGE and
            abs(new_intercept - prev_intercept) < LEAST_INTERCEPT_CHANGE)


def line_parameters_exceed_limit(new_line, prev_line):
    """Check if line parameters changed too much from previous frame"""
    if prev_line is None or new_line is None:
        return False

    new_slope, new_intercept = new_line
    prev_slope, prev_intercept = prev_line

    return (abs(new_slope - prev_slope) > MAX_SLOPE_CHANGE or
            abs(new_intercept - prev_intercept) > MAX_INTERCEPT_CHANGE)


def update_line_history(left_line, right_line):
    """Update line history with current frame's lines"""
    if left_line is not None and right_line is not None:
        line_history.append((left_line, right_line))


if __name__ == "__main__":
    video_file = 'PXL_20250325_043754655.TS.mp4'
    main(video_file)