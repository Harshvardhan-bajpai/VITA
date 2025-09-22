from flask import Flask, render_template, Response, jsonify
import trafficsolver
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(trafficsolver.gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analysis')
def analysis():
    """Get the latest traffic analysis as HTML"""
    steps_list = trafficsolver.get_steps()
    return "<br>".join(steps_list)

@app.route('/api/analysis')
def api_analysis():
    """Get the latest traffic analysis as JSON"""
    analysis = trafficsolver.get_steps()
    
    # Parse analysis into structured data
    response = {
        'timestamp': trafficsolver.frame_count,
        'analysis': analysis,
        'metrics': {
            'total_vehicles': len([line for line in analysis if 'Vehicle' in line]),
            'priority_vehicles': len(trafficsolver.priority_vehicles),
            'space_pockets': len(trafficsolver.space_pockets),
            'congestion_level': 'HIGH' if any('HIGH CONGESTION' in line for line in analysis) else 
                               'MEDIUM' if any('MEDIUM CONGESTION' in line for line in analysis) else 'LOW'
        }
    }
    
    return jsonify(response)

@app.route('/api/priority_vehicles')
def api_priority_vehicles():
    """Get detailed information about priority vehicles"""
    priority_data = []
    
    for priority_info in trafficsolver.priority_vehicles:
        vehicle = priority_info['vehicle']
        priority_data.append({
            'id': vehicle['id'],
            'type': vehicle['class'],
            'confidence': vehicle['confidence'],
            'position': vehicle['center'],
            'area': vehicle['area'],
            'priority_score': priority_info['priority_score'],
            'reasons': priority_info['reasons']
        })
    
    return jsonify({
        'count': len(priority_data),
        'vehicles': priority_data
    })

@app.route('/api/space_pockets')
def api_space_pockets():
    """Get detailed information about available space pockets"""
    space_data = []
    
    for pocket in trafficsolver.space_pockets:
        space_data.append({
            'id': pocket['id'],
            'center': pocket['center'],
            'area': pocket['area'],
            'bbox': pocket['bbox']
        })
    
    return jsonify({
        'count': len(space_data),
        'pockets': space_data
    })

@app.route('/api/heat_map_stats')
def api_heat_map_stats():
    """Get heat map statistics"""
    heat_map = trafficsolver.congestion_heat_map
    
    if heat_map is not None:
        stats = {
            'max_congestion': float(heat_map.max()),
            'avg_congestion': float(heat_map.mean()),
            'min_congestion': float(heat_map.min()),
            'total_congestion': float(heat_map.sum())
        }
    else:
        stats = {
            'max_congestion': 0.0,
            'avg_congestion': 0.0,
            'min_congestion': 0.0,
            'total_congestion': 0.0
        }
    
    return jsonify(stats)

# Legacy routes (keeping for compatibility)
@app.route('/steps')
def steps():
    """Legacy route - redirects to analysis"""
    return analysis()

@app.route('/switch_priority')
def switch_priority():
    """Legacy route - no longer functional in new system"""
    return ("Legacy function - not applicable in new heat map system", 200)

@app.route('/flag_priority')
def flag_priority():
    """Legacy route - no longer functional in new system"""
    return ("Legacy function - not applicable in new heat map system", 200)

if __name__ == "__main__":
    # Initialize YOLO model at startup
    print("Initializing YOLOv8n model...")
    trafficsolver.initialize_yolo()
    print("Starting Flask application...")
    
    app.run(host="0.0.0.0", port=5000, debug=True)