# GPS Navigation System for Madrid Streets

This project implements a GPS navigation system for streets in Madrid using graph theory. It allows users to input two street names and get detailed directions from one point to another, either based on the shortest distance or the fastest time.

## Disclaimer

**This is an old project.** The code quality may not be optimal and could benefit from improvements. Please use it as a reference or learning tool.

## Overview

The system is built using Python and leverages graph theory to represent Madrid's street map. The streets' intersections are modeled as vertices, and the roads connecting them are edges. The system uses a custom Python library to perform graph operations, such as computing the shortest path between two given street names.

## Features

- **Graph Construction:** The system constructs a graph from two datasets containing intersection and street information.
- **Shortest Path Calculation:** Uses Dijkstra’s algorithm to calculate the shortest path between two points, either by distance or time.
- **User Interaction:** Allows users to input origin and destination street names, choose whether to prioritize the shortest or fastest path, and receive detailed directions.
  
## Requirements

- Python 3.x
- NetworkX library for graph-related operations
- Pandas (for data handling)
  
Install the required libraries using:

```bash
pip install networkx pandas

