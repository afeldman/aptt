import motmetrics as mm
from aptt.lightning_base.module import BaseModule
from aptt.model.tracking import Tracker
from aptt.tracker.tracker_manager import TrackerManager

class TrackingModule(BaseModule):
    def __init__(self, 
                 detection_model,
                 tracker_type='kalman', 
                 hidden_dim=128, 
                 rnn_type='GRU',  
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trackingmodel = Tracker(detection_model=detection_model, 
                                     hidden_dim=hidden_dim,
                                     rnn_type=rnn_type)

        self.tracker = TrackerManager(filter_type=tracker_type, device=self.device)
        self.accumulator = mm.MOTAccumulator(auto_id=True)

        # Frame-weises Zwischenspeichern der Ergebnisse
        self._gt_by_frame = {}
        self._pred_by_frame = {}

    def validation_step(self, batch, batch_idx):
        """
        Batch sollte enthalten:
        - images: [B, C, H, W]
        - targets: Dict mit 'boxes' und 'ids' (pro Frame)
        """
        images, targets = batch
        batch_size = images.shape[0]

        #all_preds = []

        for i in range(batch_size):
            #img = images[i]
            gt_boxes = targets[i]['boxes']
            gt_ids = targets[i]['ids']
            frame_id = targets[i].get('frame_id', batch_idx * batch_size + i)

            # MOT: Detektion → (z. B. YOLO o. CenterNet, hier Dummy)
            pred_boxes = gt_boxes  # TODO: echte detections
            pred_features = [None for _ in pred_boxes]  # TODO: ReID-Features wenn gewünscht

            self.tracker.update(pred_boxes, pred_features)
            tracks = self.tracker.get_active_tracks()

            pred_ids = [t.id for t in tracks]
            pred_boxes = [t.boxes[-1] for t in tracks]

            # Speichern
            self._gt_by_frame[frame_id] = (gt_ids, gt_boxes)
            self._pred_by_frame[frame_id] = (pred_ids, pred_boxes)

            # Evaluation mit motmetrics
            dist = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
            self.accumulator.update(gt_ids, pred_ids, dist)

        return {}  # Kein loss

    def on_validation_epoch_end(self):
        # Trackingmetriken berechnen
        mh = mm.metrics.create()
        summary = mh.compute(
            self.accumulator,
            metrics=['mota', 'motp', 'idf1', 'num_switches'],
            name='MOT'
        )

        mot_results = summary.loc['MOT'].to_dict()
        for metric, value in mot_results.items():
            self.log(f"val/{metric}", value, prog_bar=True, logger=True)

        # Reset für nächsten Lauf
        self.accumulator = mm.MOTAccumulator(auto_id=True)
        super().on_validation_epoch_end()
