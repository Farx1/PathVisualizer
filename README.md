# Pathfinding Algorithm Visualizer

## Description
This Pathfinding Algorithm Visualizer is an interactive Python application that demonstrates how various pathfinding algorithms work in a grid-based environment. It provides a visual representation of how different algorithms search for a path between two points, allowing users to better understand their behavior and efficiency.

## Features
- Visualize multiple pathfinding algorithms:
  - A* (A-star)
  - Dijkstra's Algorithm
  - Breadth-First Search (BFS)
  - Depth-First Search (DFS)
  - Greedy Best-First Search
- Interactive grid where users can:
  - Set start and end points
  - Create obstacles
  - Adjust grid size
- Real-time visualization of the algorithm's progress
- Option to toggle between real-time and instant visualization
- Adjustable brush size for creating obstacles
- Information panel displaying:
  - Current algorithm
  - Execution time
  - Path length
  - Current brush size
- Color-coded visualization:
  - Green: Start node
  - Red: End node
  - Black: Obstacles
  - Blue: Path
  - Cyan: Open set
  - Orange: Closed set

## Requirements
- Python 3.x
- Pygame
- Tkinter (usually comes pre-installed with Python)

## Installation
1. Ensure you have Python installed on your system.
2. Install Pygame if you haven't already:
   ```
   pip install pygame
   ```
3. Download or clone this repository to your local machine.

## Usage
1. Run the script:
   ```
   python pathfinding_visualizer.py
   ```
2. Use the left mouse button to set the start point (first click), end point (second click), and obstacles.
3. Use the right mouse button to erase nodes.
4. Press the spacebar to start the visualization once you've set the start and end points.
5. Use the following keys to control the visualizer:
   - '**C**': Clear the grid
   - '**T**': Toggle between different algorithms
   - '**V**': Toggle between real-time and instant visualization
   - **Up Arrow**: Increase brush size
   - **Down Arrow**: Decrease brush size
## Acknowledgements
This project was inspired by a video tutorial from Tech With Tim on pathfinding algorithm visualization. I encourage you to check out his excellent content and GitHub repositories:

- Tech With Tim YouTube: https://www.youtube.com/@TechWithTim
- Tech With Tim GitHub: https://github.com/techwithtim

The visualizer has been expanded with additional features and improvements, but the core concept and initial implementation draw from Tim's educational content on pathfinding algorithms.

This project aims to further the understanding of pathfinding algorithms and their applications in computer science and artificial intelligence.