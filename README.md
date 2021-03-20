First Kaggle Competition for VinBigData

[COMPLETED] PreTraining Goals:
- 

- [x] Rebuild the ResNet 50 model ( backbone for the detectron2 model )
    - [x] Find 256px Grayscale pretrained resnet-50 weights, someone has to have it
    - [x] Visualize training to make sure its working correctly 
      - [x] Add hooks to training tasks for visualizations
- [x] Train ResNet model 
- [x] Load weights into detectron2 res-50-fpn model and train (might use my model over detectrons)
    - [x] Create a method of mapping weights between models 
    - [x] Fix Mix Up 
        - [x] Overlapping mixup check
        - [x] Increase likelyhood of interesting mixups
    - [x] Better Loss Logging for Retina
    - [x] Create custom retina classification head (instead of duck patch)
    - [x] Fix Batch Augmentation
- [x] Create ResNet50 FPN model w/ classifiers per class (weights from detectron)
- [x] Ensemble the two models, custom resnet 50 model and detectron to classify if an abnormality is present then find the bounding box for said abnormality respectively


Training Goals
- 

####_Current Kaggle Score 0.052_

- [ ] Explore ways to enhance the model [ Currently Working on This ]


Hypothesis:
- 

By training the backbone to detect abnormalities we can get two benefits

1.) Utilize unlabeled or Healthy labeled X-rays to flesh out the backbones weights (more training examples)

2.) Ensemble models so that during inference, the model can decide if any abnormality is present instead of always predicting multiple abnormalities as seen with detectron models


Nice to haves (things in the backlog*)
- 

- [ ] Better Documentation 
- [ ] Better Logging
- [x] Visualizations of training/validation trade offs
- [x] ResNet / RetinaNet implementations that scale
- [x] CUDA implementation with env files
- [ ] Finish the custom Resnet model because its cool


Submission text
-

My first kaggle competition submission!

These predictions are from 2 retina based models.  One being a binary classifier with a custom classification head, used to determine if the image is healthy or abnormal.  THe second model is for when the image is abnormal, using a traditional retina model, classifies the abnormalities and their locations.  Trained locally on 2 980tis for 6hours total