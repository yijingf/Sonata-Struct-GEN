from fractions import Fraction

import sys
sys.path.append("..")

from utils.align import t_to_bar_pos
# Function to calculate overlap proportion


def calculate_overlap(final_segments, pred_segments):
    start_B, end_B = final_segments
    start_A, end_A, _ = pred_segments
    overlap = max(0, min(end_B, end_A) - max(start_B, start_A))
    return overlap / (end_B - start_B) if end_B > start_B else 0


# Assign most likely label
def assign_labels(pred_segments, final_segments):
    result = []
    for interval in final_segments:
        max_overlap = 0
        best_label = None
        for interval_pred in pred_segments:
            overlap = calculate_overlap(interval, interval_pred)
            if overlap > max_overlap:
                max_overlap = overlap
                best_label = interval_pred[2]
        result.append((interval, best_label))
    return result


def get_struct_label(sect_phrase, seg_bound, ts_cpt):

    remain_label = ['i', 'A', 'B', 'X', 'b', 'o']
    beat_per_bar = Fraction(ts_cpt[0]['time_signature']) * 4

    pred_labels = []
    for interval, label in zip(*seg_bound):
        pred_labels.append(list(interval) + [label])

    label_map = {}
    struct_labels = []

    for sect in sect_phrase:
        segments = sect_phrase[sect]
        # Assign labels to B based on A
        labeled_segments = assign_labels(pred_labels, segments)

        # convert label_id to label (iABbXo), ""
        for i, (interval, label_id) in enumerate(labeled_segments):
            if label_id not in label_map:
                label = ''
                if sect == "expose":
                    if i == 0:
                        label = 'i'
                    elif i == len(labeled_segments) - 1:
                        label = 'o'
                elif sect == "intro":
                    label = 'X'
                elif sect == "dev":
                    if "b" in remain_label:
                        label = "b"

                if label:
                    remain_label.remove(label)
                else:
                    label = remain_label.pop(0)
                label_map[label_id] = label
            else:
                label = label_map[label_id]

            st_bar, st_pos = t_to_bar_pos(interval[0], ts_cpt)
            ed_bar, ed_pos = t_to_bar_pos(interval[1], ts_cpt)
            st = st_bar + st_pos / beat_per_bar
            ed = ed_bar + ed_pos / beat_per_bar
            struct_labels.append(f"{label}{round(ed-st)}")

    return "".join(struct_labels)
