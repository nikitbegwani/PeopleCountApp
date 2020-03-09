"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default="/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so",
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-c", "--color", required=False, type=str,
                        help="he color of the bounding boxes to draw; RED, GREEN or BLUE",
                        default='BLUE')
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client =  mqtt.Client()
    return client

def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Blue if an invalid color is given.
    '''
    colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors['BLUE']

def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    
    people_count = 0
    for box in result[0][0]:
        conf = box[2]
        if conf >= args.prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), args.color, 1)
            people_count = people_count + 1
    return frame, people_count


def infer_on_stream(args, client = None):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    
    # Initialise the class
    infer_network = Network()
    image_flag = False
    
    current_request_id = 0
    last_count = 0
    total_count = 0
    start_time = 0    
    
    args.color = convert_color(args.color)
    # Set Probability threshold for detections
    args.prob_threshold = float(args.prob_threshold) 

    ### TODO: Load the model through `infer_network` ###
    n, c, h, w = infer_network.load_model(args.model, args.device, 
                                          current_request_id, args.cpu_extension)[1]
    print(n,c,h,w)

    ### TODO: Handle the input stream ###
    
    # Webcam as a input
    if args.input == 'CAM':
        input_stream = 0
        
    # Image as input
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        image_flag = True
        
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    
    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    ### TODO: Loop until stream is over ###
    
    while cap.isOpened():
        # Read the next frame
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        ### Pre-process the frame as needed ###
        
        image = cv2.resize(frame, (w, h))        
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))

        ### TODO: Start asynchronous inference for specified request ###
        
        inference_start_time = time.time()
        infer_network.exec_net(current_request_id, image)
        
         ### Wait for the result ###
        if infer_network.wait(current_request_id) == 0:
            inference_time = time.time() - inference_start_time
            
            ### Get the results of the inference request ###
            result = infer_network.get_output(current_request_id)
            out_frame, people_count = draw_boxes(frame, result, args, width, height)
            log.info("people_count is {} after draw_boxes...".format(people_count))
            
            ### Calculate and send relevant information on ###
            
            inference_time_msg = "Time taken to infer: {:.3f}ms"\
                               .format(inference_time * 1000)
            
            cv2.putText(out_frame, inference_time_msg, (25, 25),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, args.color, 1)
            
            ### current_count, total_count and duration to the MQTT server ###
            
            ### If some new people enter the room
            ### We add the new comers to total_count
            ### Topic "person": keys of "count" and "total" ###
            if people_count > last_count:
                start_time = time.time()
                total_count = total_count + people_count - last_count
                client.publish("person", json.dumps({"total": total_count}))

            # If a person leaves the room
            # We calculate his/her duration spent
            ### Topic "person/duration": key of "duration" ###
            # Avoiding any kind of flicker by putting time difference to be atleast 1sec
            if (people_count < last_count) and int(time.time() - start_time) >=1:
                duration = int(time.time() - start_time)
                # Push info
                client.publish("person/duration",
                               json.dumps({"duration": duration}))

            client.publish("person", json.dumps({"count": people_count}))
            last_count = people_count

            if key_pressed == 27:
                break

        # Send frame to the ffmpeg server
        sys.stdout.buffer.write(out_frame)  
        sys.stdout.flush()

        if image_flag:
            cv2.imwrite('output_image.jpg', frame)
    
    ### Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    ### Disconnect from MQTT
    client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)

if __name__ == '__main__':
    main()
    exit(0)
