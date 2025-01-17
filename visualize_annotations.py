import os
import json
import cv2
from IPython.display import Image, display

def visualize_annotations(image_path, annotation_file):
    """
    Visualize annotations directly from JSON file with correct scaling.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Load annotations
    with open(annotation_file, "r") as f:
        annotations = json.load(f)

    for annotation in annotations:
        if annotation["image_name"] == os.path.basename(image_path):
            for obj in annotation.get("annotations", []):
                bbox = obj["bbox"]  # Already scaled to original size
                score = obj["score"]
                class_name = obj["class_name"]

                # Draw bounding boxes and labels on the image
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} ({score:.2f})"
                cv2.putText(
                    image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

    # Save and display the image
    output_path = f"{os.path.splitext(image_path)[0]}_visualized.png"
    cv2.imwrite(output_path, image)
    display(Image(filename=output_path))
    print(f"Annotated image saved to: {output_path}")


# Example usage
image_path = "/content/YOLOX_outputs/yolo_voc_custom_s/vis_res/2025_01_15_20_34_46/3-phases-separator.png"
annotation_file = "/content/YOLOX_outputs/yolo_voc_custom_s/vis_res/2025_01_15_20_34_46/annotations.json"
visualize_annotations(image_path, annotation_file)
