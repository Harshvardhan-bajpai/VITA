# VITA - Vehicle Intelligent Traffic Assistant
VITA is an innovative, AI-driven traffic management system designed to bring fluidity and intelligence to urban roadways. It addresses the challenges of urban congestion by replacing static, inefficient traffic control with a dynamic, dual-component system that manages everything from individual intersections to large-scale traffic jams.

# The Challenge: Navigating Urban Congestion
Pervasive traffic jams are a common blight in metropolitan areas, often caused by the static and inefficient control of traffic at critical road junctions. This leads to significant productivity loss and has become a normalized part of daily life. VITA aims to solve this by making traffic management adaptive and intelligent.

# Solution: A Two-Tiered System
VITA employs a two-tiered approach to traffic management: a tactical system for junction control and a strategic system for resolving major congestion.

1. Smart Junction Control (The Tactical System)
This system provides automated, intelligent management of traffic signals at intersections using real-time camera feeds.
Real-time Lane Monitoring: VITA uses dedicated lane cameras at each junction to constantly observe vehicle flow and density. Each lane is monitored independently to gauge traffic volume. This is implemented in 
lanecam.py by processing multiple video sources.
Adaptive Signal Control: The AI dynamically adjusts signal timings based on live traffic conditions, not fixed schedules. It determines optimal signal changes to maximize throughput and switches lanes instantly when traffic is minimal. The 
lanecam.py script achieves this through a priority algorithm that considers vehicle count, type, and wait times.
Red Light Violation Detection: To enhance road safety, VITA automatically detects and records vehicles violating red lights, capturing visual proof with precise timestamps for law enforcement. This feature is handled by the 
check_violations function in lanecam.py.

2. The Jam Solver Module (The Strategic System)
For complex scenarios like accidents or major events, VITA provides extended capabilities to manage and resolve severe congestion.
Aerial Intelligence: A dedicated drone can be deployed to hover above an affected junction, providing a live aerial feed to traffic control officers for a comprehensive assessment. This is simulated in the project by the trafficsolver.py script processing a high-angle video feed.
AI-Powered Analysis & Planning: The aerial feed is processed by VITA's AI, which identifies bottlenecks and vehicle density patterns. It then generates a precise, step-by-step clearance plan tailored to the specific jam scenario. This is implemented in trafficsolver.py through its congestion and space pocket analysis functions.
Web-Based Dashboard for Officer Guidance: Officers receive clear, real-time instructions and a live video feed through a web-based dashboard, allowing them to efficiently execute the clearance strategy. This dashboard is served by app.py and defined in index.html.
Priority Vehicle Handling
VITA is designed to act swiftly in emergencies to minimize delays for critical vehicles.
Officers can flag emergency vehicles (ambulance, fire, police) through the system.
The AI immediately calculates and facilitates the fastest possible clearance path for the flagged vehicle.
The lanecam.py system can do this automatically by detecting sirens and emergency lights, triggering an immediate green light for that lane.

# Technology Stack
Backend: Python, Flask
Computer Vision: OpenCV, Ultralytics YOLOv8
Data Processing & Scientific Computing: NumPy, SciPy
Audio Processing: Sounddevice (for siren detection)
Frontend: HTML, CSS, JavaScript

# Project Structure
The project is organized into two main components:

VITA-Smart-Traffic-Management/
├── jam_solver_webapp/        # The strategic system dashboard
│   ├── app.py
│   ├── trafficsolver.py
│   └── templates/
│       └── index.html
│
├── smart_junction_system/    # The tactical intersection controller
│   ├── lanecam.py
│   └── traffic_violations/
│
└── README.md

# Setup and Usage

Smart Junction System
This is a standalone script that simulates a multi-camera traffic intersection.

1. Navigate to the smart junction system directory
   cd smart_junction_system

3. Install the required libraries
   pip install -r requirements.txt

5. Run the script
   python lanecam.py

Jam Solver Web App
This runs a web server that displays the drone feed analysis.

1. Navigate to the jam solver directory
   cd jam_solver_webapp

3. Install the required libraries
   pip install -r requirements.txt

5. Run the Flask application
   python app.py
