import cv2
import numpy as np
from collections import defaultdict
import time
from ultralytics import YOLO
from scipy.spatial.distance import pdist, squareform

vehicle_positions_history = defaultdict(list)
congestion_zones = []
space_pockets = []
priority_vehicles = []
latest_analysis = []
frame_count = 0
HISTORY_FRAMES = 20

yolo_model = None

def initialize_yolo():
    global yolo_model
    try:
        yolo_model = YOLO('yolov8n.pt')
        print("YOLOv8n model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading YOLOv8n model: {e}")
        return False

def detect_vehicles(frame):
    global yolo_model, frame_count
    
    if yolo_model is None:
        if not initialize_yolo():
            return []
    
    frame_count += 1
    vehicles = []
    
    try:
        results = yolo_model(frame, verbose=False)
        
        vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck',
            1: 'bicycle'
        }
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id in vehicle_classes and confidence > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w, h = x2 - x1, y2 - y1
                        center_x, center_y = x1 + w // 2, y1 + h // 2
                        area = w * h
                        
                        vehicles.append({
                            'id': f'V{frame_count}_{i}',
                            'bbox': (x1, y1, w, h),
                            'center': (center_x, center_y),
                            'area': area,
                            'class': vehicle_classes[class_id],
                            'confidence': confidence,
                            'class_id': class_id
                        })
        
    except Exception as e:
        print(f"Error in vehicle detection: {e}")
    
    return vehicles

def analyze_vehicle_spacing_and_congestion(frame, vehicles):
    global congestion_zones, space_pockets
    
    if len(vehicles) < 2:
        congestion_zones = []
        space_pockets = []
        return [], []
    
    height, width = frame.shape[:2]
    
    # Get vehicle centers for distance analysis
    vehicle_centers = [vehicle['center'] for vehicle in vehicles]
    vehicle_data = {i: vehicle for i, vehicle in enumerate(vehicles)}
    
    # Calculate distances between all vehicles
    distances = pdist(vehicle_centers, metric='euclidean')
    distance_matrix = squareform(distances)
    
    # Define congestion and spacing thresholds
    CONGESTION_THRESHOLD = 80  # pixels - vehicles closer than this are congested
    SPACE_THRESHOLD = 150      # pixels - gaps larger than this are space pockets
    MIN_CLUSTER_SIZE = 3       # minimum vehicles to form a congestion zone
    
    # Find congested vehicle clusters
    congestion_zones = []
    visited = set()
    
    for i, vehicle in enumerate(vehicles):
        if i in visited:
            continue
            
        # Find all vehicles close to this one
        cluster = [i]
        queue = [i]
        visited.add(i)
        
        while queue:
            current = queue.pop(0)
            for j in range(len(vehicles)):
                if j not in visited and distance_matrix[current][j] < CONGESTION_THRESHOLD:
                    cluster.append(j)
                    queue.append(j)
                    visited.add(j)
        
        # If cluster is large enough, mark as congestion zone
        if len(cluster) >= MIN_CLUSTER_SIZE:
            cluster_vehicles = [vehicle_data[idx] for idx in cluster]
            
            # Calculate bounding box of the cluster
            min_x = min(v['center'][0] - v['bbox'][2]//2 for v in cluster_vehicles)
            max_x = max(v['center'][0] + v['bbox'][2]//2 for v in cluster_vehicles)
            min_y = min(v['center'][1] - v['bbox'][3]//2 for v in cluster_vehicles)
            max_y = max(v['center'][1] + v['bbox'][3]//2 for v in cluster_vehicles)
            
            # Calculate cluster density
            cluster_area = (max_x - min_x) * (max_y - min_y)
            density = len(cluster) / max(cluster_area, 1)
            
            # Calculate average spacing within cluster
            cluster_distances = []
            for idx1 in cluster:
                for idx2 in cluster:
                    if idx1 < idx2:
                        cluster_distances.append(distance_matrix[idx1][idx2])
            
            avg_spacing = np.mean(cluster_distances) if cluster_distances else 0
            
            congestion_zones.append({
                'id': f'CZ{len(congestion_zones) + 1}',
                'vehicles': cluster_vehicles,
                'bbox': (min_x, min_y, max_x - min_x, max_y - min_y),
                'center': ((min_x + max_x) // 2, (min_y + max_y) // 2),
                'vehicle_count': len(cluster),
                'density': density,
                'avg_spacing': avg_spacing,
                'congestion_level': 'HIGH' if avg_spacing < 50 else 'MEDIUM' if avg_spacing < 80 else 'LOW'
            })
    
    # Find space pockets (large gaps between vehicle groups)
    space_pockets = []
    
    # Create a grid to analyze empty spaces
    grid_size = 30
    grid_height = height // grid_size
    grid_width = width // grid_size
    occupancy_grid = np.zeros((grid_height, grid_width))
    
    # Mark grid cells occupied by vehicles
    for vehicle in vehicles:
        cx, cy = vehicle['center']
        w, h = vehicle['bbox'][2], vehicle['bbox'][3]
        
        # Mark area around vehicle as occupied
        radius = max(w, h) // 2
        grid_x = cx // grid_size
        grid_y = cy // grid_size
        
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                gx, gy = grid_x + dx, grid_y + dy
                if 0 <= gx < grid_width and 0 <= gy < grid_height:
                    occupancy_grid[gy][gx] = 1
    
    # Find connected empty regions (space pockets)
    visited_grid = np.zeros_like(occupancy_grid)
    
    def flood_fill(start_y, start_x):
        """Find connected empty region using flood fill"""
        if (visited_grid[start_y][start_x] == 1 or 
            occupancy_grid[start_y][start_x] == 1):
            return []
        
        region = []
        queue = [(start_y, start_x)]
        visited_grid[start_y][start_x] = 1
        
        while queue:
            y, x = queue.pop(0)
            region.append((y, x))
            
            # Check 4-connected neighbors
            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if (0 <= ny < grid_height and 0 <= nx < grid_width and
                    visited_grid[ny][nx] == 0 and occupancy_grid[ny][nx] == 0):
                    visited_grid[ny][nx] = 1
                    queue.append((ny, nx))
        
        return region
    
    # Find all space pockets
    for y in range(grid_height):
        for x in range(grid_width):
            if visited_grid[y][x] == 0 and occupancy_grid[y][x] == 0:
                region = flood_fill(y, x)
                
                if len(region) > 8:  # Only consider significant empty regions
                    # Convert grid coordinates back to pixel coordinates
                    pixel_region = [(y * grid_size, x * grid_size) for y, x in region]
                    
                    # Calculate bounding box
                    min_x = min(x for y, x in pixel_region)
                    max_x = max(x for y, x in pixel_region) + grid_size
                    min_y = min(y for y, x in pixel_region)
                    max_y = max(y for y, x in pixel_region) + grid_size
                    
                    # Ensure within frame bounds
                    min_x = max(0, min_x)
                    max_x = min(width, max_x)
                    min_y = max(0, min_y)
                    max_y = min(height, max_y)
                    
                    area = (max_x - min_x) * (max_y - min_y)
                    
                    if area > 5000:  # Only consider large enough spaces
                        space_pockets.append({
                            'id': f'SP{len(space_pockets) + 1}',
                            'bbox': (min_x, min_y, max_x - min_x, max_y - min_y),
                            'center': ((min_x + max_x) // 2, (min_y + max_y) // 2),
                            'area': area,
                            'grid_cells': len(region)
                        })
    
    return congestion_zones, space_pockets

def detect_priority_vehicles(vehicles, congestion_zones, frame):
    global priority_vehicles
    priority_vehicles = []
    
    for vehicle in vehicles:
        x, y, w, h = vehicle['bbox']
        roi = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Check for red and blue emergency lights
        red_mask = cv2.inRange(hsv, (0, 120, 120), (10, 255, 255))
        blue_mask = cv2.inRange(hsv, (100, 120, 120), (130, 255, 255))
        
        has_emergency_lights = (np.sum(red_mask) > 100) or (np.sum(blue_mask) > 100)
        
        if has_emergency_lights and vehicle['class'] in ['truck', 'bus']:  # Potential ambulance/firetruck
            priority_vehicles.append({
                'vehicle': vehicle,
                'type': 'Emergency Vehicle',
                'has_emergency_lights': True
            })
    
    return priority_vehicles

def generate_traffic_analysis():
    global latest_analysis, congestion_zones, space_pockets, priority_vehicles
    analysis = []
    total_vehicles = sum(zone['vehicle_count'] for zone in congestion_zones)
    if not congestion_zones:
        total_vehicles = 0
    analysis.append(f"TOTAL VEHICLES DETECTED: {total_vehicles}")
    
    if congestion_zones:
        high_congestion_zones = [z for z in congestion_zones if z['congestion_level'] == 'HIGH']
        medium_congestion_zones = [z for z in congestion_zones if z['congestion_level'] == 'MEDIUM']
        
        if high_congestion_zones:
            analysis.append("HIGH CONGESTION ZONES DETECTED")
            for zone in high_congestion_zones:
                analysis.append(f"Zone {zone['id']}: {zone['vehicle_count']} vehicles, avg spacing: {zone['avg_spacing']:.0f}px")
        
        if medium_congestion_zones:
            analysis.append("MEDIUM CONGESTION ZONES DETECTED")
            for zone in medium_congestion_zones:
                analysis.append(f"Zone {zone['id']}: {zone['vehicle_count']} vehicles, avg spacing: {zone['avg_spacing']:.0f}px")
        
        if not high_congestion_zones and not medium_congestion_zones:
            analysis.append("LOW CONGESTION - Good vehicle spacing")
    else:
        analysis.append("NO CONGESTION DETECTED - Optimal spacing")
    
    if space_pockets:
        total_space_area = sum(pocket['area'] for pocket in space_pockets)
        analysis.append(f"AVAILABLE SPACE POCKETS: {len(space_pockets)}")
        analysis.append(f"Total available space: {total_space_area}px²")
        
        for i, pocket in enumerate(space_pockets[:3]):
            x, y = pocket['center']
            area = pocket['area']
            analysis.append(f"Pocket {pocket['id']}: Center({x},{y}), Area={area}px²")
    else:
        analysis.append("NO SIGNIFICANT SPACE POCKETS - Limited rerouting options")
    
    if priority_vehicles:
        analysis.append(f"PRIORITY VEHICLES: {len(priority_vehicles)}")
        
        for i, priority_info in enumerate(priority_vehicles[:3]):
            vehicle = priority_info['vehicle']
            score = priority_info['priority_score']
            reasons = priority_info['reasons']
            
            analysis.append(f"  {vehicle['class'].upper()} (Score: {score})")
            analysis.append(f"    Position: {vehicle['center']}")
            analysis.append(f"    Reasons: {', '.join(reasons)}")
    else:
        analysis.append("No priority vehicles requiring immediate attention")
    
    analysis.append("TRAFFIC MANAGEMENT RECOMMENDATIONS:")
    
    if congestion_zones:
        high_congestion = [z for z in congestion_zones if z['congestion_level'] == 'HIGH']
        if high_congestion:
            analysis.append("1. Disperse high congestion zones immediately")
            analysis.append("2. Implement traffic light optimization")
        analysis.append("3. Guide vehicles to available space pockets")
    
    if priority_vehicles:
        analysis.append("4. Clear priority vehicles first")
    
    if space_pockets and congestion_zones:
        analysis.append("5. Redirect traffic to space pockets")
        analysis.append("6. Activate dynamic route guidance")
    
    if not congestion_zones and not priority_vehicles:
        analysis.append("• Traffic flow optimal - maintain current state")
    
    latest_analysis = analysis
    return analysis

def gen_frames():
    global congestion_zones, space_pockets, priority_vehicles
    
    cap = cv2.VideoCapture("sample.mp4") #tcp://192.168.137.110:5000
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect vehicles
        vehicles = detect_vehicles(frame)
        
        # Analyze spacing and congestion
        congestion_zones, space_pockets = analyze_vehicle_spacing_and_congestion(frame, vehicles)
        
        # Detect priority vehicles
        priority_vehicles = detect_priority_vehicles(vehicles, congestion_zones, frame)  # Added frame parameter
        
        # Generate analysis
        generate_traffic_analysis()
        
        # Draw congestion zones
        for zone in congestion_zones:
            x, y, w, h = zone['bbox']
            
            # Color based on congestion level
            if zone['congestion_level'] == 'HIGH':
                color = (0, 0, 255)  # Red
                thickness = 4
            elif zone['congestion_level'] == 'MEDIUM':
                color = (0, 165, 255)  # Orange
                thickness = 3
            else:
                color = (0, 255, 255)  # Yellow
                thickness = 2
            
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
            cv2.putText(frame, f"CONGESTION {zone['id']} - {zone['congestion_level']}", 
                       (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"{zone['vehicle_count']} vehicles", 
                       (int(x), int(y + h + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw space pockets
        for pocket in space_pockets:
            x, y, w, h = pocket['bbox']
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), 3)
            cv2.putText(frame, f"SPACE {pocket['id']}", (int(x), int(y - 5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw vehicles with priority highlighting
        priority_vehicle_ids = [pv['vehicle']['id'] for pv in priority_vehicles]
        
        for vehicle in vehicles:
            x, y, w, h = vehicle['bbox']
            center_x, center_y = vehicle['center']
            
            # Different colors for priorities
            if vehicle['id'] in priority_vehicle_ids:
                color = (0, 0, 255)  # Red for priority
                thickness = 3
            elif vehicle['class'] in ['bus', 'truck']:
                color = (0, 255, 255)  # Yellow for large vehicles
                thickness = 2
            else:
                color = (0, 255, 0)  # Green for regular vehicles
                thickness = 2
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Vehicle label
            label = f"{vehicle['class']}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Center point
            cv2.circle(frame, (center_x, center_y), 3, color, -1)
        
        # Add system info
        info_y = 30
        cv2.putText(frame, "VEHICLE SPACING & CONGESTION ANALYSIS", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Vehicles: {len(vehicles)} | Congestion Zones: {len(congestion_zones)}", 
                   (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Priority: {len(priority_vehicles)} | Space Pockets: {len(space_pockets)}", 
                   (10, info_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

def get_steps():
    return latest_analysis

def switch_priority():
    pass

def flag_priority():
    pass