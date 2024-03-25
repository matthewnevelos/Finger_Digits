# a5-model-improvment

## a5 Goal
The goal in this lab is to improve the finger digits classifier  
OR  
to understand better why the model makes predictions in a certain way.

You may choose at least one of the approaches below
### 1. Improve your data
- Acquire more/other training and validation data with a2 tools. 
- More specifically, make sure only your hand is in the field of view and you have a selection of backgrounds and lightning conditions
- Retrain the model using possibly the learning rate finder and freeze/unfreeze capabilities `fine_tune()` provides (see fastbook Ch 5). 
- Re-evaluate production performance similar to a4.
### 2. Share and grab data
- Copy your training and validation data to a folder `digits_zzz` (`zzz` your choice of number) on [UofC onedrive](https://uofc-my.sharepoint.com/:f:/g/personal/yves_pauchard_ucalgary_ca/Eswi4KbJsmlMl8M4G7lM5ioBM_TIpwilEV9x86wsBcaTRQ?e=dvJXH5)  
- In turn, grab any data present in the onedrive folder.
- Combine and retrain a model with this larger dataset using possibly the learning rate finder and freeze/unfreeze capabilities `fine_tune()` provides (see fastbook Ch 5).
- Re-evaluate production performance similar to a4.
- Note that the data in this folder is available under a [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. If you upload your images you agree to these conditions.
### 3. Different architecture
- Build your own architecture using fastbook Ch 13 and Ch 14 ideas OR
- Use [pre-trained timm models](https://timm.fast.ai/): EfficientNet or
  MobileNet, or other. If you pass a `str` to `vision_learner` for `arch`, then
  a timm model will be created. For example, `learn = vision_learner(dls,
  'convnext_tiny')`. More information:
    - Install timm with `pip install timm` in your `endg411` conda environment.
    - [timm modesl performance
      comparison](https://www.kaggle.com/code/jhoward/which-image-models-are-best/)
    - [More info on timm library](https://huggingface.co/docs/timm/index)
- Retrain a model using possibly the learning rate finder and freeze/unfreeze capabilities `fine_tune()` provides (see fastbook Ch 5).
- Re-evaluate production performance similar to a4.
### 4. Interpret the CNN:
- Use class activation map (CAM) described in fastbook Ch 18 to interpret which areas in the image the model uses for prediction.
- Does the CAM change when you add more varied data OR train differently (look at [RandomErasing](https://docs.fast.ai/vision.augment.html#randomerasing)) ?
- Present your findings in a structured manner with summary and conclusion.

## What to hand in
- A notebook `a5-model-improvement.ipynb`:
    - *Introduction* Describe the approach chosen, reason for choosing it and how you implemented it.
    - *Model Code* model training and results
    - *Summary and Conclusion* Summarize your results, re-state a3/a4 results and comment on changes/improvements.
    - *Reflection*
- Keep code clean, comment/document and remove any unnecessary cells in the notebook.

During development, save progress with git and use descriptive commit messages.

Hand in: git push all files, verify on github, submit url on D2L.

**Important:** Do **not** commit image data to github. Images do **not** need to be handed in.
