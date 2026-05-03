def nms(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression.
    """
    # Write code here
    def iou(boxA, boxB):
        xa1, ya1, xa2, ya2 = boxA
        xb1, yb1, xb2, yb2 = boxB

        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)
        union_area = area_a + area_b - inter_area

        return inter_area / union_area if union_area > 0 else 0

    indices = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []

    while indices:
        current = indices.pop(0)
        keep.append(current)
        indices = [
            i for i in indices
            if iou(boxes[current], boxes[i]) < iou_threshold
        ]

    return keep
    