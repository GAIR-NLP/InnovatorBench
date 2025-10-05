from tools import segment_and_mark, detection, depth, crop_image, zoom_in_image_by_bbox, sliding_window_detection, overlay_images
from PIL import Image
image_path = "image1.jpg"
image = Image.open(image_path)
image_path = "image2.jpg"
image1 = Image.open(image_path)



x, y, width, height = 0, 0.2, 0.4,0.8
# Use object detection
output_image, bboxes = detection(image=image, objects=["bird"])
# print(output_image, bboxes)
output_image.annotated_image.save("test.jpg")
# Use image segmentation
output_image, processed_boxes = segment_and_mark(image=image)
output_image.annotated_image.save("test1.jpg")
# Use depth estimation
output_image = depth(image=image)
output_image.save("test2.jpg")
# Use image cropping
cropped_img = crop_image(image=image, x=x, y=y, width=width, height=height)
cropped_img.save("test3.jpg")
# Use region zooming
zoomed_image = zoom_in_image_by_bbox(image=image, box=(x, y, width, height))
zoomed_image.save("test4.jpg")
# Use sliding window detection
possible_patches, possible_boxes = sliding_window_detection(image=image, objects=["bird","tree"])
possible_patches[0].annotated_image.save("test5.jpg")
# Use image overlay
overlay_image = overlay_images(background_img=image, overlay_img=image1)
overlay_image.save("test6.jpg")