# =============================================================================
# 시각화 함수 정의
# =============================================================================

COLORS = [
    [0.850, 0.325, 0.098],  # fully-ripe (빨강)
    [0.929, 0.694, 0.125],  # semi-ripe (노랑)
    [0.466, 0.674, 0.188],  # unripe (초록)
]

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes, id2label):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        cl = p.argmax()
        color = COLORS[cl % len(COLORS)]
        ax.add_patch(plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            fill=False, color=color, linewidth=3
        ))
        text = f'{id2label[int(cl.item())]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin - 5, text, fontsize=12, color='white',
                bbox=dict(facecolor=color, alpha=0.8, pad=2))
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_predictions(image, outputs, id2label, threshold=0.7):
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)
    plot_results(image, probas[keep], bboxes_scaled, id2label)

