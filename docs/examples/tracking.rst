Object Tracking
===============

Beispiel für Multi-Object Tracking mit APTT.

Basic Tracking
--------------

.. code-block:: python

   from aptt.modules.tracking import TrackingModule
   from aptt.tracker.tracker import KalmanFilter, LSTMTracker
   from aptt.lightning_base.trainer import BaseTrainer
   
   # Tracking Modell erstellen
   model = TrackingModule(
       detector=yolo_model,  # Ihr trainiertes Detection-Modell
       tracker_type="kalman",  # oder "lstm", "particle"
       num_classes=80,
       lr=1e-4
   )
   
   # Training
   trainer = BaseTrainer(
       log_dir="logs/tracking",
       max_epochs=50
   )
   
   trainer.fit(model, datamodule=tracking_datamodule)

Kalman Filter Tracking
-----------------------

.. code-block:: python

   from aptt.tracker.tracker import KalmanFilter
   import numpy as np
   
   # Kalman Filter initialisieren
   tracker = KalmanFilter(
       state_dim=4,  # [x, y, vx, vy]
       measurement_dim=2  # [x, y]
   )
   
   # Tracking Loop
   tracks = {}
   next_track_id = 0
   
   for frame in video_frames:
       # Detektionen vom Detektor
       detections = detector(frame)
       boxes = detections["boxes"]
       scores = detections["scores"]
       
       # Tracking update
       for box, score in zip(boxes, scores):
           if score > 0.5:
               cx = (box[0] + box[2]) / 2
               cy = (box[1] + box[3]) / 2
               measurement = np.array([cx, cy])
               
               # Finde nächsten Track
               min_dist = float('inf')
               matched_id = None
               
               for track_id, track in tracks.items():
                   predicted = track.predict()
                   dist = np.linalg.norm(predicted[:2] - measurement)
                   
                   if dist < min_dist and dist < 50:  # Threshold
                       min_dist = dist
                       matched_id = track_id
               
               if matched_id is not None:
                   # Update existierenden Track
                   tracks[matched_id].update(measurement)
               else:
                   # Neuer Track
                   new_tracker = KalmanFilter(state_dim=4, measurement_dim=2)
                   new_tracker.update(measurement)
                   tracks[next_track_id] = new_tracker
                   next_track_id += 1

LSTM Tracker
------------

.. code-block:: python

   from aptt.tracker.tracker import LSTMTracker
   import torch
   
   # LSTM Tracker
   lstm_tracker = LSTMTracker(
       input_dim=4,  # [x, y, w, h]
       hidden_dim=128,
       num_layers=2
   )
   
   # Training
   optimizer = torch.optim.Adam(lstm_tracker.parameters(), lr=1e-3)
   
   for epoch in range(100):
       for sequence, target in train_loader:
           # sequence: [batch, seq_len, 4]
           # target: [batch, seq_len, 4]
           
           optimizer.zero_grad()
           prediction = lstm_tracker(sequence)
           loss = F.mse_loss(prediction, target)
           loss.backward()
           optimizer.step()

DeepSORT-style Tracking
-----------------------

.. code-block:: python

   from aptt.tracker.reid_encoder import ReIDEncoder
   from aptt.tracker.tracker import Track
   import torch
   
   class DeepSORT:
       def __init__(self, reid_model, max_age=30, min_hits=3):
           self.reid_model = reid_model
           self.max_age = max_age
           self.min_hits = min_hits
           self.tracks = []
           self.next_id = 0
       
       def update(self, detections, frame):
           """
           detections: Liste von [x1, y1, x2, y2, score]
           frame: aktuelles Bild
           """
           # Extract features für ReID
           features = []
           for det in detections:
               x1, y1, x2, y2, score = det
               crop = frame[int(y1):int(y2), int(x1):int(x2)]
               feature = self.reid_model.extract(crop)
               features.append(feature)
           
           features = torch.stack(features)
           
           # Matching mit existierenden Tracks
           matches, unmatched_dets, unmatched_tracks = self._match(
               detections, features
           )
           
           # Update matched tracks
           for det_idx, track_idx in matches:
               self.tracks[track_idx].update(
                   detections[det_idx],
                   features[det_idx]
               )
           
           # Neue Tracks für unmatched detections
           for det_idx in unmatched_dets:
               new_track = Track(
                   track_id=self.next_id,
                   detection=detections[det_idx],
                   feature=features[det_idx]
               )
               self.tracks.append(new_track)
               self.next_id += 1
           
           # Entferne alte Tracks
           self.tracks = [
               t for t in self.tracks 
               if t.time_since_update < self.max_age
           ]
           
           return self.tracks
       
       def _match(self, detections, features):
           # IoU + Feature distance matching
           from scipy.optimize import linear_sum_assignment
           
           if len(self.tracks) == 0:
               return [], list(range(len(detections))), []
           
           # Cost matrix
           cost_matrix = np.zeros((len(detections), len(self.tracks)))
           
           for i, (det, feat) in enumerate(zip(detections, features)):
               for j, track in enumerate(self.tracks):
                   # IoU distance
                   iou = self._iou(det[:4], track.last_detection[:4])
                   
                   # Feature distance
                   feat_dist = torch.nn.functional.cosine_similarity(
                       feat.unsqueeze(0),
                       track.last_feature.unsqueeze(0)
                   ).item()
                   
                   # Kombinierte cost
                   cost_matrix[i, j] = 0.5 * (1 - iou) + 0.5 * (1 - feat_dist)
           
           # Hungarian algorithm
           row_ind, col_ind = linear_sum_assignment(cost_matrix)
           
           matches = []
           unmatched_dets = []
           unmatched_tracks = list(range(len(self.tracks)))
           
           for r, c in zip(row_ind, col_ind):
               if cost_matrix[r, c] < 0.3:  # Threshold
                   matches.append((r, c))
                   unmatched_tracks.remove(c)
               else:
                   unmatched_dets.append(r)
           
           return matches, unmatched_dets, unmatched_tracks
       
       @staticmethod
       def _iou(box1, box2):
           # IoU Berechnung
           x1 = max(box1[0], box2[0])
           y1 = max(box1[1], box2[1])
           x2 = min(box1[2], box2[2])
           y2 = min(box1[3], box2[3])
           
           intersection = max(0, x2 - x1) * max(0, y2 - y1)
           area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
           area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
           union = area1 + area2 - intersection
           
           return intersection / union if union > 0 else 0

Video Tracking
--------------

.. code-block:: python

   import cv2
   from aptt.tracker.inference import run_tracking
   
   # Video laden
   cap = cv2.VideoCapture("input_video.mp4")
   
   # Output writer
   fourcc = cv2.VideoWriter_fourcc(*'mp4v')
   fps = int(cap.get(cv2.CAP_PROP_FPS))
   width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   out = cv2.VideoWriter("tracked_video.mp4", fourcc, fps, (width, height))
   
   # Tracker initialisieren
   tracker = DeepSORT(reid_model, max_age=30, min_hits=3)
   
   # Farben für verschiedene IDs
   colors = {}
   
   while True:
       ret, frame = cap.read()
       if not ret:
           break
       
       # Detektionen
       detections = detector(frame)
       
       # Tracking update
       tracks = tracker.update(detections, frame)
       
       # Visualisierung
       for track in tracks:
           if track.hits >= tracker.min_hits:
               x1, y1, x2, y2 = track.last_detection[:4].astype(int)
               track_id = track.track_id
               
               # Farbe für diese ID
               if track_id not in colors:
                   colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
               
               color = colors[track_id]
               
               # Box zeichnen
               cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
               
               # ID Label
               label = f"ID: {track_id}"
               cv2.putText(frame, label, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
               
               # Trajectory
               if hasattr(track, 'trajectory'):
                   points = np.array(track.trajectory, dtype=np.int32)
                   cv2.polylines(frame, [points], False, color, 2)
       
       out.write(frame)
       
       # Anzeige
       cv2.imshow('Tracking', frame)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   
   cap.release()
   out.release()
   cv2.destroyAllWindows()

Metrics
-------

.. code-block:: python

   import motmetrics as mm
   
   def evaluate_tracking(gt_tracks, pred_tracks):
       """
       Evaluiert Tracking mit MOT Metrics
       """
       acc = mm.MOTAccumulator(auto_id=True)
       
       for frame_id in range(len(gt_tracks)):
           gt = gt_tracks[frame_id]  # [(id, x, y, w, h), ...]
           pred = pred_tracks[frame_id]
           
           # Distance matrix
           distances = mm.distances.iou_matrix(
               [g[1:] for g in gt],
               [p[1:] for p in pred],
               max_iou=0.5
           )
           
           acc.update(
               [g[0] for g in gt],  # GT IDs
               [p[0] for p in pred],  # Pred IDs
               distances
           )
       
       # Metriken berechnen
       mh = mm.metrics.create()
       summary = mh.compute(
           acc,
           metrics=['num_frames', 'mota', 'motp', 'num_switches',
                   'num_false_positives', 'num_misses'],
           name='tracking'
       )
       
       print(summary)
       return summary
