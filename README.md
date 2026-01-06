# Larvae Detection API

A switchable Computer Vision API for counting larvae in water samples. It supports two detection backends:
1.  **Traditional CV:** Optical Flow + Frame Differencing (CPU optimized).
2.  **Deep Learning:** YOLOv8 + ByteTrack (GPU/High-Accuracy).


## Setup

### 1. Environment
Create a virtual environment to keep dependencies isolated.

```bash
# Create venv
python -m venv venv

# Activate venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

```

### 2. Install Dependencies

Install the required libraries for the API and Computer Vision.

```bash
pip install -r requirements.txt

```

*(Ensure your `requirements.txt` contains: `fastapi`, `uvicorn`, `python-multipart`, `opencv-python-headless`, `numpy`, `ultralytics`, `supervision`)*

---

## Running the API

Start the server using Uvicorn.

```bash
uvicorn app.main:app --reload

```

* **API URL:** `http://localhost:8000`
* **Swagger UI (Docs):** [http://localhost:8000/docs](https://www.google.com/search?q=http://localhost:8000/docs)

### How to Test

1. Go to the **Swagger UI** link above.
2. Open the **POST /analyze_video** endpoint.
3. Set `mode` to `traditional`.
4. Upload a sample video (MP4).
5. Click **Execute**.

---

## Code Explanation: Traditional Strategy

The `TraditionalDetectorStrategy` (`app/core/strategies/trad_strategy.py`) uses classical computer vision to detect motion without neural networks. This makes it extremely fast and capable of running on low-end hardware.

### Core Logic Pipeline

#### 1. Gamma Correction & Preprocessing

**Goal:** Fix lighting and contrast issues.

* **Gamma Correction:** We apply a non-linear brightness adjustment (`invGamma = 2.0`). This brightens the dark, muddy water to reveal hidden larvae without blowing out the shiny reflections (glare).
* **Grayscale & Blur:** Converts the video to black-and-white and applies a Gaussian Blur to remove sensor noise (grain).

#### 2. Video Stabilization (Global Motion Compensation)

**Goal:** Ignore camera shake.

* If the user holds the phone with shaky hands, the whole image moves. A simple motion detector would think the *entire water* is moving.
* **Optical Flow (Lucas-Kanade):** The code finds "corners" (distinct points) in the previous frame and tracks where they moved in the current frame.
* **Affine Warp:** It calculates the camera's movement (rotation + translation) and mathematically "un-rotates" the current frame to perfectly align with the previous one.

#### 3. Frame Striding & Differencing

**Goal:** Detect "Wriggling" behavior.

* **Striding:** Instead of comparing Frame T vs. T-1 (where motion is tiny), we compare **Frame T vs. T-5**. This makes small wriggles look like large, detectable jumps.
* **3-Frame Difference:** We compare `(Prev vs Curr)` AND `(Curr vs Next)`. A pixel is only marked as "moving" if it changes in *both* comparisons. This eliminates ghosting artifacts.

#### 4. Auto-Gain & Thresholding

**Goal:** See the invisible.

* **Normalization:** We stretch the difference values. If the larvae only change the pixel color by 2% (invisible to humans), the code stretches that range to 0-100%, forcing the motion to stand out.
* **Thresholding:** Any pixel with a change value > 30 is marked as "White" (Motion). Everything else is "Black" (Background).

#### 5. Tracking & Counting (The Registry)

**Goal:** Count *Unique* individuals.

* **Centroid Tracking:** We convert the white blobs into (x, y) coordinates.
* **Distance Matching:** If a blob in Frame 10 is close to a blob in Frame 5, we assume it's the same larva (ID #1).
* **Persistence Check:** An ID is only added to the **Final Registry** if it is seen for at least 5 frames. This prevents random water glints from being counted as larvae.