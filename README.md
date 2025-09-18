# AI-ML-project-
The project is about grid environment and dynamic obstacles 
# Autonomous Delivery Agent – CSA2001 (Fundamentals of AI & ML)

## 📌 Project Overview
This project implements an **autonomous delivery agent** that navigates a **2D grid city** to deliver packages.  
The agent must avoid static and dynamic obstacles, handle varying terrain costs, and optimize delivery efficiency (time & fuel).  

We implemented and compared the following algorithms:
- **Uninformed Search**: BFS / Uniform-Cost Search  
- **Informed Search**: A* with admissible heuristics  
- **Local Search with Replanning**: Hill-Climbing with random restarts / simulated annealing  

The agent also supports **dynamic replanning** when obstacles appear or move during execution.

---

## ⚙️ Features
- Models:
  - Static obstacles (walls, blocked cells)  
  - Dynamic obstacles (moving vehicles)  
  - Different terrain costs (fuel/time)  
- Pathfinding algorithms:
  - BFS (Breadth-First Search)  
  - Uniform-Cost Search  
  - A* Search (Manhattan heuristic)  
  - Hill-Climbing / Random Restart (for replanning)  
- Handles **dynamic environments** by replanning when obstacles appear  
- CLI-based interface to test algorithms on different maps  
- Experimental analysis (path cost, nodes expanded, time)  
