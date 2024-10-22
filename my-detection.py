import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
image_path = "/home/nvidia/jetson-inference/examples/img_831.jpeg"
display = jetson.utils.videoOutput("display://0")  # Use display output

while display.IsStreaming():
  img = jetson.utils.loadImage(image_path)
  if img is None:#capture timeout
     continue
  detections = net.Detect(img)

  for detection in detections:
      print(f"-- ClassID: {detection.ClassID}")
      print(f"-- Confidence: {detection.Confidence:.6f}")
      print(f"-- Left: {detection.Left:.5f}")
      print(f"-- Top: {detection.Top:.5f}")
      print(f"-- Right: {detection.Right:.5f}")
      print(f"-- Bottom: {detection.Bottom:.5f}")
      print(f"-- Width: {detection.Width:.5f}")
      print(f"-- Height: {detection.Height:.5f}")
      print(f"-- Area: {detection.Area:.5f}")
      print(f"-- Center: ({detection.Center[0]:.3f}, {detection.Center[1]:.3f})")
      print()

  display.Render(img)
  display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
