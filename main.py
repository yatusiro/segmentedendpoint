import image_processing
import find_endpoint
import cv2  
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
# path = "MR1 BP_1_20250516_135217.jpg"

folder_path = "segment/boat"

jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))


print(jpg_files)
for path in jpg_files:
    print(path)
    img = Image.open(path)
    processed_img = image_processing.detect_and_highlight_wires_test(img)
    annotated, pts = find_endpoint.find_endpoints(processed_img)
    outputfloder = "endpoint"
    filename = os.path.basename(path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(outputfloder, f"{name}_processed{ext}")
    annotated.save(output_path)


















# img = Image.open(path)

# processed_img = image_processing.detect_and_highlight_wires_test(img)
# annotated, pts = find_endpoint.find_endpoints(processed_img)

# plt.imshow(annotated)
# plt.axis('off')
# plt.title("Detected Endpoints")
# # endpointフォルダーに保存
# plt.savefig("endpoint/annotated.jpg", bbox_inches='tight', pad_inches=0)


