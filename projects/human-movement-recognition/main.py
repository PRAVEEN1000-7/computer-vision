# DOWNLOAD THE MODEL USING PROVIDED LINK : https://github.com/visiongeeklabs/human-activity-detection/releases/download/v0.1.0/frozen_inference_graph.pb
import numpy as np
import time
import cv2
import tensorflow.compat.v1 as tf

# Load labels
with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the frozen graph
frozen_graph_path = 'frozen_inference_graph.pb'
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    t1 = time.time()
    #fid = tf.gfile.GFile(frozen_graph_path, 'rb')
    fid = tf.io.gfile.GFile(frozen_graph_path, 'rb')

    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    t2 = time.time()
    print("Model loading time: ", t2 - t1)

    sess = tf.Session(graph=detection_graph)
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
if not cap.isOpened():
    print("Could not open webcam")
    exit()

# Generate random colors for bounding boxes
colors = np.random.uniform(0, 255, size=(80, 3))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Expand dimensions since the model expects images to have shape [1, None, None, 3]
    frame_exp = np.expand_dims(frame, axis=0)

    # Run inference
    t1 = time.time()
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: frame_exp})
    t2 = time.time()
    print("Inference time: ", t2 - t1)

    # Post-process the results
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    threshold = 0.7
    for i in range(output_dict['num_detections']):
        if int(output_dict['detection_classes'][i]) not in [1, 3, 17, 37, 43, 45, 46, 47, 59, 65, 74, 77, 78, 79, 80]:
            if output_dict['detection_scores'][i] > threshold:
                bbox = output_dict['detection_boxes'][i]
                height, width, _ = frame.shape
                bbox[0] *= height
                bbox[1] *= width
                bbox[2] *= height
                bbox[3] *= width

                idx = int(output_dict['detection_classes'][i]) - 1
                cv2.rectangle(frame, (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])), colors[idx], 2)
                cv2.putText(frame, labels[idx], (int(bbox[1]), int(bbox[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
sess.close()
