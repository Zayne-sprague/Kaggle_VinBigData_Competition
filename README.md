First Kaggle Competition for VinBigData

Goal:
- 

- [x] Rebuild the ResNet 50 model ( backbone for the detectron2 model )
    - [x] Find 256px Grayscale pretrained resnet-50 weights, someone has to have it
    - [x] Visualize training to make sure its working correctly 
      - [x] Add hooks to training tasks for visualizations
- [x] Train ResNet model 
- [x] Load weights into detectron2 res-50-fpn model and train (might use my model over detectrons)
    - [ ] Create a method of mapping weights between models [ currently working on this ]
    - [ ] Fix Mix Up 
        - [ ] Refactor out for loop
        - [ ] Overlapping mixup check
        - [ ] Increase likelyhood of interesting mixups
    - [ ] Optimize memory performance (torch lightening?)
    - [ ] Better Loss Logging for Retina
    - [ ] Create custom retina classification head (instead of duck patch)
    - [ ] Fix Validation for Retina to be faster/cleaner in logs
        - [ ] Create custom mAP metrics (better understand it, but also verifies implementation)
    - [ ] Fix Batch Augmentation
- [ ] Create ResNet50 FPN model w/ classifiers per class (weights from detectron)
- [ ] Ensemble the two models, custom resnet 50 model and detectron to classify if an abnormality is present then find the bounding box for said abnormality respectively

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
