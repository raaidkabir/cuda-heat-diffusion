# Heat Diffusion Simulation (CUDA)

## Overview
This CUDA C++ program simulates heat diffusion and visualizes the results. It uses parallel computing capabilities of CUDA for efficient processing.

## Components
- **UpdateTemperature Kernel**: Computes temperature changes using the diffusion equation.
- **TemperatureToColor Kernel**: Converts temperatures to RGB colors for visualization.
- **Main Function**: Initializes arrays, runs simulation loop, and saves frames as PNG images.
- **Memory Management**: Manages memory on host and device.

## Requirements
- CUDA Toolkit
- Compatible NVIDIA GPU

## How to Run
1. Compile the program using nvcc:
`nvcc heat.cu -o heat_simulation`
2. Run the executable:
`./heat_simulation`

## Output
The program outputs a series of PNG images visualizing the heat diffusion process.
