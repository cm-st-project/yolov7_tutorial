import torch

model = torch.hub.load("WongKinYiu/yolov7", "custom", 'models/yolov7.pt')
# Sample Image URL
BASE_URL = 'https://github.com/ultralytics/yolov5/raw/master/data/images/'
image_url = BASE_URL + 'zidane.jpg'
# A batch of images (only one entry here)
imgs = [image_url]

# Inference
results = model(imgs)
# Display the results
# results.show()

# Save the results
# results.save()
print(results.xyxy[0])

# Supported classes
print(model.names)