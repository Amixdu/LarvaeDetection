import cv2
import numpy as np
import os
import shutil
from typing import List, Tuple
from app.core.interfaces import BaseLarvaDetector, FrameAnalysis, DetectionResult

class TraditionalDetectorStrategy(BaseLarvaDetector):
    def __init__(self):
        # Tuning Parameters
        self.MIN_AREA = 10
        self.MAX_AREA = 600
        self.TRACK_DIST = 30
        self.CONFIRM_FRAMES = 5
        
        # State Management
        self.frame_buffer = []
        self.next_id = 0
        self.active_tracks = {}
        self.confirmed_ids = set()
        
        # --- DEBUG SETUP ---
        self.debug_dir = "debug_frames"
        # Reset debug folder on every run so we don't fill the disk
        if os.path.exists(self.debug_dir):
            shutil.rmtree(self.debug_dir)
        os.makedirs(self.debug_dir, exist_ok=True)

    def _stabilize(self, prev_gray, curr_gray):
        # (Same implementation as before)
        p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
        if p0 is None or len(p0) < 5: return curr_gray

        p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
        if p1 is None: return curr_gray

        good_new = p1[status == 1]
        good_old = p0[status == 1]
        if len(good_new) < 3: return curr_gray
        
        m, _ = cv2.estimateAffinePartial2D(good_old, good_new)
        if m is None: return curr_gray
            
        rows, cols = curr_gray.shape
        m_inv = cv2.invertAffineTransform(m)
        stabilized = cv2.warpAffine(curr_gray, m_inv, (cols, rows))
        
        return stabilized

    def _update_tracker(self, detections: List[Tuple[int, int, int, int]]):
        # (Same implementation as before)
        current_centroids = []
        for (x, y, w, h) in detections:
            cx, cy = x + w // 2, y + h // 2
            current_centroids.append({'pos': (cx, cy), 'box': (x, y, w, h)})

        used_tracks = set()
        matched_results = [] 

        for detection in current_centroids:
            cx, cy = detection['pos']
            best_id = None
            min_dist = self.TRACK_DIST

            for track_id, track_data in self.active_tracks.items():
                if track_id in used_tracks: continue
                lx, ly = track_data['last_pos']
                dist = np.sqrt((cx - lx)**2 + (cy - ly)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_id = track_id

            if best_id is not None:
                self.active_tracks[best_id]['last_pos'] = (cx, cy)
                self.active_tracks[best_id]['seen_count'] += 1
                self.active_tracks[best_id]['missing_count'] = 0
                used_tracks.add(best_id)
                matched_results.append((best_id, detection['box']))
                if self.active_tracks[best_id]['seen_count'] > self.CONFIRM_FRAMES:
                    self.confirmed_ids.add(best_id)
            else:
                new_id = self.next_id
                self.active_tracks[new_id] = {
                    'last_pos': (cx, cy),
                    'seen_count': 1,
                    'missing_count': 0
                }
                self.next_id += 1
                matched_results.append((new_id, detection['box']))

        ids_to_remove = []
        for track_id in self.active_tracks:
            if track_id not in used_tracks:
                self.active_tracks[track_id]['missing_count'] += 1
                if self.active_tracks[track_id]['missing_count'] > 10:
                    ids_to_remove.append(track_id)
        for tid in ids_to_remove: del self.active_tracks[tid]

        return matched_results

    def detect(self, frame: np.ndarray, frame_index: int) -> FrameAnalysis:
        # Keep a clean copy for visualization before we mess with the image
        debug_frame = frame.copy()

        # Gamma Correction
        invGamma = 1.0 / 0.5 
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        frame = cv2.LUT(frame, table)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        self.frame_buffer.append(gray)
        BUFFER_SIZE = 7
        if len(self.frame_buffer) > BUFFER_SIZE:
            self.frame_buffer.pop(0)

        raw_detections = [] 
        debug_mask = np.zeros_like(gray) # Default black mask if no processing

        if len(self.frame_buffer) == BUFFER_SIZE:
            prev = self.frame_buffer[0]
            curr = self.frame_buffer[BUFFER_SIZE // 2]
            next_fr = self.frame_buffer[-1]

            stab_prev = self._stabilize(curr, prev)
            stab_next = self._stabilize(curr, next_fr)

            diff1 = cv2.absdiff(stab_prev, curr)
            diff2 = cv2.absdiff(curr, stab_next)
            motion_mask = cv2.bitwise_and(diff1, diff2)

            motion_mask = cv2.normalize(motion_mask, None, 0, 255, cv2.NORM_MINMAX)
            _, thresh = cv2.threshold(motion_mask, 30, 255, cv2.THRESH_BINARY)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            thresh = cv2.dilate(thresh, kernel, iterations=2)
            
            # Save for debug output
            debug_mask = thresh

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 10 < area < self.MAX_AREA:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect = float(h)/w if h > w else float(w)/h
                    if aspect > 1.1:
                        raw_detections.append((x, y, w, h))

        tracked_objects = self._update_tracker(raw_detections)
        
        results = []
        for obj_id, (x, y, w, h) in tracked_objects:
            # Draw boxes on the debug frame
            cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_frame, f"ID:{obj_id}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            results.append(DetectionResult(
                id=obj_id,
                box=(x, y, w, h),
                confidence=1.0,
                class_id=0
            ))

        # Save every 30th frame (approx 1 second interval)
        if frame_index % 30 == 0:
            # Convert mask to 3-channel so we can stack it next to the color frame
            mask_bgr = cv2.cvtColor(debug_mask, cv2.COLOR_GRAY2BGR)
            
            # Stack images side-by-side (Left: Real, Right: Brain)
            combined = np.hstack((debug_frame, mask_bgr))
            
            filename = os.path.join(self.debug_dir, f"frame_{frame_index:04d}.jpg")
            cv2.imwrite(filename, combined)

        return FrameAnalysis(
            frame_index=frame_index,
            larvae_count=len(self.active_tracks),
            detections=results
        )

    def get_total_unique_count(self) -> int:
        return len(self.confirmed_ids)