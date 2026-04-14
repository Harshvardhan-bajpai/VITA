import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import threading
import queue
import sounddevice as sd
from scipy.fftpack import fft

class SmartTrafficSystem:
    def __init__(self, video_sources=None):
        try:
            self.model = YOLO("yolov8n.pt")
        except Exception as e:
            raise Exception("Failed to load YOLO model. Make sure you have installed ultralytics properly.") from e
        
        if video_sources is None:
            video_sources = ["lane1.mp4", "lane2.mp4", "lane3.mp4", "lane4.mp4"]
        
        self.caps = []
        for source in video_sources:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"Warning: Could not open {source}, using dummy feed")
                cap = None
            self.caps.append(cap)
        
        self.target_width = 770
        self.target_height = 770
        self.stop_line_y = int(self.target_height * 0.8)
        
        self.current_green_lane = 0
        self.next_green_lane = 1
        self.traffic_state = "GREEN"
        self.state_start_time = time.time()
        self.lane_wait_times = [0, 0, 0, 0]
        
        self.min_green_time = 5
        self.max_wait_time = 180
        self.yellow_time = 3
        self.emergency_override = False
        self.emergency_lane = -1
        
        self.upstream_flow_data = {}
        self.communication_queue = queue.Queue()
        
        self.emergency_vehicles = ["ambulance", "fire truck", "police"]
        
        self.violation_folder = "traffic_violations"
        os.makedirs(self.violation_folder, exist_ok=True)

        self.audio_buffer = queue.Queue()
        self.setup_audio_detection()
        
    def setup_audio_detection(self):
        def audio_callback(indata, frames, time, status):
            self.audio_buffer.put(indata.copy())
        
        self.audio_stream = sd.InputStream(
            channels=1,
            samplerate=44100,
            callback=audio_callback
        )
        self.audio_stream.start()
        
    def detect_siren(self):
        if self.audio_buffer.empty():
            return False
            
        audio_data = self.audio_buffer.get()
        fft_data = np.abs(fft(audio_data[:, 0]))
        
        ambulance_freq = [800, 1000]
        fire_truck_freq = [600, 800]  
        
        peaks = np.argmax(fft_data[:len(fft_data)//2])
        normalized_freq = peaks * 44100 / len(fft_data)
        
        is_ambulance = any(abs(normalized_freq - f) < 50 for f in ambulance_freq)
        is_fire_truck = any(abs(normalized_freq - f) < 50 for f in fire_truck_freq)
        
        return is_ambulance or is_fire_truck

    def detect_vehicles(self, frame):
        if frame is None:
            return []
        
        results = self.model(frame, verbose=False)[0]
        vehicles = []
        emergency_detected = False
        
        for box in results.boxes:
            cls = int(box.cls[0])
            label = self.model.names[cls]
            confidence = float(box.conf[0])
            
            if label in ["car", "truck", "bus", "motorbike", "bicycle"] and confidence > 0.5:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width, height = int(x2 - x1), int(y2 - y1)
                
                if label in ["truck", "bus"] and confidence > 0.8:
                    emergency_detected = self.is_emergency_vehicle(frame, int(x1), int(y1), width, height)
                
                vehicles.append({
                    'bbox': (int(x1), int(y1), width, height),
                    'label': label,
                    'confidence': confidence,
                    'emergency': emergency_detected
                })
        
        return vehicles

    def is_emergency_vehicle(self, frame, x, y, w, h):
        roi = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        red_mask = cv2.inRange(hsv, (0, 120, 120), (10, 255, 255))
        blue_mask = cv2.inRange(hsv, (100, 120, 120), (130, 255, 255))
        
        has_emergency_lights = (np.sum(red_mask) > 100) or (np.sum(blue_mask) > 100)
        
        has_siren = self.detect_siren()
        
        return has_emergency_lights or has_siren

    def calculate_lane_priority(self, lane_idx, vehicles):
        if not vehicles:
            return 0
        
        vehicle_count = len(vehicles)
        
        weighted_count = 0
        for vehicle in vehicles:
            if vehicle['label'] == 'bus':
                weighted_count += 3
            elif vehicle['label'] == 'truck':
                weighted_count += 2
            elif vehicle['label'] in ['motorbike', 'bicycle']:
                weighted_count += 0.5
            else:
                weighted_count += 1
        
        distance_score = 0
        for vehicle in vehicles:
            x, y, w, h = vehicle['bbox']
            distance_from_stop = abs((y + h) - self.stop_line_y)
            distance_score += max(0, 100 - distance_from_stop) / 100
        
        wait_factor = min(self.lane_wait_times[lane_idx] / 60, 2)
        
        upstream_factor = self.upstream_flow_data.get(lane_idx, 1.0)
        
        total_score = (weighted_count * 10) + distance_score + (wait_factor * 5) + (upstream_factor * 3)
        
        return total_score

    def check_emergency_override(self, all_vehicles):
        for lane_idx, vehicles in enumerate(all_vehicles):
            for vehicle in vehicles:
                if vehicle.get('emergency', False):
                    print(f"EMERGENCY VEHICLE DETECTED in Lane {lane_idx + 1}!")
                    self.emergency_override = True
                    self.emergency_lane = lane_idx
                    return True
        return False

    def update_traffic_state(self, all_vehicles):
        current_time = time.time()
        elapsed = current_time - self.state_start_time
        
        for i in range(len(self.lane_wait_times)):
            if i != self.current_green_lane:
                self.lane_wait_times[i] += 1/30
            else:
                self.lane_wait_times[i] = 0
        
        if self.check_emergency_override(all_vehicles):
            if self.emergency_lane != self.current_green_lane and elapsed >= 3:
                print(f"SWITCHING TO EMERGENCY LANE {self.emergency_lane + 1}")
                self.current_green_lane = self.emergency_lane
                self.state_start_time = current_time
                self.emergency_override = False
                return
        
        if elapsed < self.min_green_time:
            return
        
        max_wait_lane = max(range(len(self.lane_wait_times)), key=lambda i: self.lane_wait_times[i])
        if self.lane_wait_times[max_wait_lane] >= self.max_wait_time:
            print(f"FORCE SWITCH: Lane {max_wait_lane + 1} waited {self.lane_wait_times[max_wait_lane]:.1f} seconds")
            self.current_green_lane = max_wait_lane
            self.state_start_time = current_time
            return
        
        lane_priorities = []
        for i, vehicles in enumerate(all_vehicles):
            priority = self.calculate_lane_priority(i, vehicles)
            lane_priorities.append(priority)
        
        current_priority = lane_priorities[self.current_green_lane]
        max_priority_lane = max(range(len(lane_priorities)), key=lambda i: lane_priorities[i])
        max_priority = lane_priorities[max_priority_lane]
        
        should_switch = False
        
        if len(all_vehicles[self.current_green_lane]) == 0 and max_priority > 5:
            should_switch = True
            print("Switching: Current lane empty, others have traffic")
        
        elif max_priority > current_priority + 15 and elapsed >= self.min_green_time:
            should_switch = True
            print(f"Switching: Higher priority lane detected (Current: {current_priority:.1f}, Max: {max_priority:.1f})")
        
        elif elapsed >= 45:
            should_switch = True
            print("Switching: Maximum green time reached")
        
        if should_switch:
            self.traffic_state = "YELLOW"
            self.next_green_lane = max_priority_lane
            self.state_start_time = current_time
            
            if elapsed >= self.yellow_time:
                self.traffic_state = "GREEN"
                self.current_green_lane = self.next_green_lane
                self.state_start_time = current_time

    def draw_traffic_light(self, frame, is_green):
        light_x, light_y = 10, 10
        cv2.rectangle(frame, (light_x, light_y), (light_x+40, light_y+100), (50, 50, 50), -1)
        
        red_color = (100, 100, 100) if is_green else (0, 0, 255)
        cv2.circle(frame, (light_x+20, light_y+20), 15, red_color, -1)
        
        green_color = (0, 255, 0) if is_green else (100, 100, 100)
        cv2.circle(frame, (light_x+20, light_y+70), 15, green_color, -1)
        
        return frame

    def check_violations(self, frame, vehicles, lane_idx, is_green):
        if is_green:
            return
        
        for vehicle in vehicles:
            x, y, w, h = vehicle['bbox']
            bottom_y = y + h
            
            if bottom_y >= self.stop_line_y:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"{self.violation_folder}/lane{lane_idx+1}_violation_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"VIOLATION: {vehicle['label']} jumped red light in Lane {lane_idx+1}")

    def process_frame(self, frame, lane_idx):
        if frame is None:
            frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
            cv2.putText(frame, f"Lane {lane_idx+1}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(frame, "No Camera", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            return frame, []
        
        frame = cv2.resize(frame, (self.target_width, self.target_height))
        vehicles = self.detect_vehicles(frame)
        
        for vehicle in vehicles:
            x, y, w, h = vehicle['bbox']
            color = (0, 0, 255) if vehicle.get('emergency', False) else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, vehicle['label'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        is_green = (lane_idx == self.current_green_lane)
        frame = self.draw_traffic_light(frame, is_green)
        
        cv2.line(frame, (0, self.stop_line_y), (self.target_width, self.stop_line_y), (0, 0, 255), 2)
        
        cv2.putText(frame, f"Lane {lane_idx+1}", (self.target_width-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Wait: {self.lane_wait_times[lane_idx]:.1f}s", (self.target_width-120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(frame, f"Vehicles: {len(vehicles)}", (self.target_width-120, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        self.check_violations(frame, vehicles, lane_idx, is_green)
        
        return frame, vehicles

    def simulate_upstream_data(self):
        current_hour = time.localtime().tm_hour
        
        if 9 <= current_hour <= 10 or 17 <= current_hour <= 19:
            self.upstream_flow_data = {0: 1.5, 1: 0.8, 2: 0.8, 3: 1.2}
        else:
            self.upstream_flow_data = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}

    def run(self):
        while True:
            frames = []
            all_vehicles = []
            
            for i, cap in enumerate(self.caps):
                if cap is not None:
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    processed_frame, vehicles = self.process_frame(frame if ret else None, i)
                else:
                    processed_frame, vehicles = self.process_frame(None, i)
                
                frames.append(processed_frame)
                all_vehicles.append(vehicles)
            
            self.simulate_upstream_data()
            self.update_traffic_state(all_vehicles)
            
            if len(frames) >= 2:
                top_row = np.hstack(frames[:2])
                if len(frames) >= 4:
                    bottom_row = np.hstack(frames[2:4])
                    combined = np.vstack([top_row, bottom_row])
                else:
                    combined = top_row
            else:
                combined = frames[0] if frames else np.zeros((300, 400, 3), dtype=np.uint8)
            
            info_height = 80
            info_panel = np.zeros((info_height, combined.shape[1], 3), dtype=np.uint8)
            
            if self.traffic_state == "GREEN":
                state_info = f"GREEN: Lane {self.current_green_lane + 1}"
                state_color = (0, 255, 0)
            elif self.traffic_state == "YELLOW":
                state_info = f"YELLOW: Lane {self.current_green_lane + 1} -> Lane {self.next_green_lane + 1}"
                state_color = (0, 255, 255)
            else:
                state_info = "SWITCHING"
                state_color = (255, 255, 255)
            
            cv2.putText(info_panel, state_info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
            cv2.putText(info_panel, f"System Time: {time.strftime('%H:%M:%S')}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(info_panel, "Emergency Override: " + ("ACTIVE" if self.emergency_override else "NORMAL"), 
                       (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0) if self.emergency_override else (255, 255, 255), 1)
            
            final_display = np.vstack([combined, info_panel])
            cv2.imshow("Smart Traffic Management System", final_display)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        for cap in self.caps:
            if cap is not None:
                cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_sources = ["lane1.mp4", "lane2.mp4"]
    try:
        system = SmartTrafficSystem(video_sources)
        system.run()
    except Exception as e:
        print(f"Error: {e}")