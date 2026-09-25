# Screen Tracker V2 scaffold
# Project Overview

This project is an educational computer vision prototype developed to explore real-time object detection, motion tracking, and prediction algorithms. The primary goal of this project is to gain practical experience in software development, artificial intelligence (AI), and computer vision concepts through hands-on experimentation in a controlled testing environment.

## System Architecture
<img width="1311" height="550" alt="System architecture" src="https://github.com/user-attachments/assets/e0d2bc6f-a293-45ef-8f05-e67f8d62b980" />


## Platform

- **Programming Language:** Python
- **Development Environment:** Visual Studio Code (VS Code)
- **Primary Library:** OpenCV
- **Version Control:** Git (if applicable)

## Project Purpose

The software processes video frames to detect motion, identify Regions of Interest (ROIs), track moving objects, and make short-term predictions about their future positions. These features are designed to demonstrate fundamental computer vision and AI concepts while improving programming and algorithm development skills.

## AI and Prediction

The project incorporates AI-inspired prediction techniques to estimate the future movement of detected objects based on previous observations. These prediction algorithms are intended for experimentation and learning rather than perfect accuracy, allowing different approaches to be tested and evaluated.

## Project Structure

The codebase is organized into separate modules, each responsible for a specific task, including:

- Motion detection
- ROI (Region of Interest) detection
- Object tracking
- Movement prediction
- Debugging and visualization

This modular structure improves readability, maintainability, and future expansion of the project.

## Testing Environment

All development and testing are performed in a controlled environment using recorded gameplay and experimental test scenarios. Testing focuses on evaluating detection accuracy, prediction performance, and reducing false detections while improving overall system reliability.

## Inspiration

This project is inspired by real-world computer vision applications, including technologies used in Tesla's driver-assistance and autonomous driving systems. Tesla vehicles use cameras, computer vision, and predictive algorithms to detect surrounding objects and estimate their future movement. Although this project operates on a much smaller scale and serves a different purpose, it applies similar foundational concepts for educational learning.

## Educational Purpose

This project was created solely for educational and research purposes. Its objective is to improve programming, computer vision, and artificial intelligence skills through experimentation and software development. It is not intended for commercial use or to promote unfair gameplay or violations of any software's Terms of Service.

---
## Testing Results


### Player Tracking Yellow Box
https://github.com/user-attachments/assets/09c242c9-3fb3-4cc5-9476-4829c8127b59





---

## Experimental Testing and Modifications


### Goal
Minimize background noise and significantly improve the precision of tracking and locking onto the projectile (cannonball).
### Methods & Results
1. Enhancing Identification Features
- Method: Shifted the image tracking feature from Grayscale to Hue Feature to perform motion tracking.
- Result: Effectively eliminated significant flooring noise, making the projectile and moving objects much more distinct. This adjustment succeeded because most background noise exhibits minimal movement, whereas the target object features highly saturated and vivid coloring.
2. Optimizing the Tracking Algorithm
- Problem Statement: Although the modified features from Method 1 clearly captured the target within the Region of Interest (ROI), the initial tracking algorithm still failed to actively track and follow the object.
- Attempt 2.1: Tweaking Lucas-Kanade Optical Flow Parameters (cv2.calcOpticalFlowPyrLK)
  - Action: Expanded the search window and added an extra pyramid level (winSize=(31, 31), maxLevel=4) to accommodate fast motion of up to 500 px/sec (approximately 17 px of displacement between 30 FPS frames).
  - Result: Unsuccessful. The algorithm was still unable to maintain a stable track.
- Attempt 2.2: Implementing a New Tracking Algorithm (Final Solution)
  - Action: Replaced the optical flow method with a Hue-based Absolute Frame Differencing algorithm.
  - Technical Core: This approach leverages a dynamic Region of Interest (ROI) combined with morphological image processing to accurately estimate the moving center of the object.
  - Result: Successful. This algorithm resolved the tracking failures and successfully locked onto the high-speed target.

---

## Disclaimer

This project was developed solely for educational and research purposes to explore computer vision and object tracking techniques. Any use of this software that violates a game's Terms of Service or results in account restrictions, suspensions, or bans (including actions taken by Supercell) is the sole responsibility of the user. The developers assume no responsibility or liability for any consequences arising from the use or misuse of this software.

Disclaimer: This project was developed solely for educational and research purposes to explore computer vision and object tracking techniques. It is not intended to encourage or support cheating or unfair gameplay. Any use of this software that violates a game’s Terms of Service or results in account restrictions, suspensions, or bans (including actions taken by Supercell) is the sole responsibility of the user. The developers assume no responsibility or liability for any consequences arising from the use or misuse of this software.
