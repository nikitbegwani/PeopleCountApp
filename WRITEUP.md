# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The models which are imported might have custom layers whose exact replica is not present in openvino. The process behind converting custom layers involves registering the layers as Custom, then use Caffe to calculate the output shape of the layer. We need Caffe on your system to do this option.

If not handled, custom layers are not able to infer and is listed under unsupported layers. To avoid this from happening, its important that we convert the custom layers

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The size of the model pre- and post-conversion was... same when I provided just the input_model and input_proto as parameter.

The inference time of the model pre- and post-conversion was... ~2s and ~ 44ms

## Assess Model Use Cases

This framework can be used for alot of use cases and why it will be useful :-
1. Counting number of people entering or leaving a store to determine the peak hours.
2. Alarming the security in case someone is trying to trespass an area.
3. Counting crowds in forbidden areas in a manufacturing unit to enforce safety rules and minimize health risks. 

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

1. If the lighting is not proper, the detection becomes difficult and hence provides incorrect results. 
2. Decrease in model accuracy can lead to miscounting which will lead to wrong statistics. 
3. If the focal length/ image size is very skewed, the detection becomes difficult as the image of human in frame might not be captured nicely.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [MobileNet-SSD]
  - [https://github.com/chuanqi305/MobileNet-SSD]
  - I converted the model to an Intermediate Representation with the following arguments...
  --input_model MobileNetSSD_deploy10695.caffemodel --input_proto MobileNetSSD_deploy.prototxt
  ""
  - The model was insufficient for the app because...
  It had unsupported layers 
  Unsupported layers found: ['conv17_2_mbox_priorbox', 'conv16_2_mbox_priorbox', 'conv15_2_mbox_priorbox', 'conv14_2_mbox_priorbox', 'conv13_mbox_priorbox', 'conv11_mbox_priorbox', 'detection_out']
  - I tried to improve the model for the app by...
  
- Model 2: [VGG_q6 Faster_rcnn]
  - [https://docs.openvinotoolkit.org/2018_R5/_samples_object_detection_demo_README.html]
  - I converted the model to an Intermediate Representation with the following arguments...
  
  --input_model VGG16_faster_rcnn_final.caffemodel --input_proto faster_rcnn.prototxt
  
  - The model was insufficient for the app because...
  The prototext file was corrupted and threw `318:5 : Message type "mo_caffe.DropoutParameter" has no field named "scale_train".`
 
  
  - I tried to improve the model for the app by...
  
    Tried fixing it by changing the prototxt url to "https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt"
    
   Model Conversion is not ending. Its taking too long to convert.
Later on model got converted but the size was huge.

- Model 3: [MobileNet-SSD V2]
  - [https://github.com/C-Aniruddh/realtime_object_recognition]
  - I converted the model to an Intermediate Representation with the following arguments...
  Same error as Model 1
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
