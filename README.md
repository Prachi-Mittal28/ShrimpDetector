# 🦐 ShrimpDetector (Posture Punisher)

> A real-time computer vision daemon that forces you to fix your posture by minimizing your active windows and dimming your screen when you slouch.

![Demo GIF Placeholder](link-to-your-gif-here.gif) 
*(Add a 5-second GIF here showing you slouching, the warning popping up, and the screen dimming)*

## 🛑 The Problem
Software engineers spend 8 hours a day slowly folding into the shape of a shrimp. Traditional posture apps just send a passive desktop notification that is incredibly easy to ignore. I needed something that actually disrupted my workflow and forced me to sit up straight.

## 💡 The Solution
ShrimpDetector uses a webcam, OpenCV, and Google's MediaPipe to track my body's alignment in real-time. 

If my posture drops below an acceptable baseline for more than 10 seconds, the app executes **"The Punishment"**:
1. It hooks into the OS window manager and instantly **minimizes my active window**.
2. It sends hardware instructions to **dim my monitor brightness to 10%**.
The screen only restores its brightness when I physically sit back up.

## ⚙️ How It Works (The Math)
I didn't want to use heavy Machine Learning models to classify "good" vs "bad" posture, so I used geometric distance instead:

1. **Calibration:** On startup, the app takes 5 seconds to map the user's perfect posture. 
2. **Landmark Tracking:** It extracts the Y-axis coordinates of the user's ears (Landmarks 7 & 8) and shoulders (Landmarks 11 & 12).
3. **The Algorithm:** It calculates the vertical distance between the ear and the shoulder. When a user slouches, their head drops forward and down, decreasing this vertical distance. 
4. **Trigger:** If the current distance drops below 75% of the calibrated baseline, the timer starts. 

## 🛠️ Tech Stack
* **Python 3**
* **OpenCV (`cv2`)**: Webcam frame capture and visual overlays.
* **MediaPipe**: Real-time pose landmark extraction.
* **Screen-Brightness-Control**: OS-level hardware brightness manipulation.
* **PyGetWindow**: OS-level window manager integration.

## 🚀 How to Run Locally

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ShrimpDetector.git
cd ShrimpDetector
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

Press `q` while the webcam window is focused to quit the application.
