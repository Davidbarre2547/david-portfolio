import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

def image_loader(image_path):
  # Load an Image
  image = cv2.imread(str(image_path))

  # Analyze the Image for emotion
  analysis = DeepFace.analyze(img_path=image_path, actions=["emotion"])
  # Add emotion label to the Image
  for face in analysis:
    emotion_label = max(face['emotion'], key=face['emotion'].get)
    # Get bounding box coordinates, handling potential extra values
    region_values = list(face['region'].values()) # Convert values to a list
    x, y, w, h = region_values[:4]

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

  # Convert BGR to RGB for matplotlib
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Display the Image
  plt.figure(figsize=(10, 8))
  plt.imshow(image_rgb)
  plt.axis("off")
  plt.title("Emotion Detection")
  plt.show()
     

import pathlib
     

target = pathlib.Path("/content/")
     

for image in target.glob("*"):
  if image.is_file() and image.suffix.lower() in ('.jpg', '.jpeg', '.png'): # Check if it's a file and has a common image extension
    image_loader(image)