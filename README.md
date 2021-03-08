First Kaggle Competition for VinBigData

Goal:
- 

- Rebuild the ResNet 50 model ( backbone for the detectron2 model ) [ Currently on this step ]
- Transfer learning to rebuilt ResNet model
- Train new ResNet model on Check X-rays for detecting abnormal x-rays
- Export weights
- Load weights into detectron2 res-50-fpn model
- Use weights as backbone for the actual bounding box classifier
- Ensemble the two models, custom resnet 50 model and detectron to classify if an abnormality is present then find the bounding box for said abnormality respectively

Hypothesis:
- 

By training the backbone to detect abnormalities we can get two benefits

1.) Utilize unlabeled or Healthy labeled X-rays to flesh out the backbones weights (more training examples)

2.) Ensemble models so that during inference, the model can decide if any abnormality is present instead of always predicting multiple abnormalities as seen with detectron models


Nice to haves (things in the backlog*)
- 

- Better Documentation 
- Better Logging
- Visualizations of training/validation trade offs
- ResNet / RetinaNet implementations that scale
- CUDA implementation with env files
