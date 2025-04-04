# Segmentation and Pixel Classification 

*Lab authors: Damian Dalle Nogare and Florian Jug* . 

<small>This file last updated 2025-03-31.</small>

---

## **Segmentation** 

### Learning Objectives

- Train a pixel classifier to segment images
- Learn to use ilastik {cite}`Berg2019-no`
- Experiment with how features and data affect performance

**Lab Data** in [this folder](https://tinyurl.com/QI2025AnalysisLabData) (Machine_Learning)

Remember to **unzip** the data folder after downloading.

---

### Training a Random Forest Pixel Classifier in ilastik

```{margin} Want to learn more about working with ilastik?
Check out the [documentation and tutorials list](https://www.ilastik.org/documentation/), or check them out on [the image.sc forum!](https://forum.image.sc/tag/ilastik)
```

- Today we will be following along with the excellent [ilastik pixel classifier tutorial](https://bit.ly/3x11EZf).  
  Let us know if you hit a snag!

- You can use the data in Machine_Learning/Pixel_Classifier/ (link to
  Lab Data can be found on top of this page!). Load both the 'Train' and the 'Test' files. Note that you’ll need to
  change the axes of the 'Train' file to “tyx” because these images are not a z-stack (right click on your image and go to Edit Properties).

```{admonition} Question
:class: important

There is a train and a test stack. Why is that?
 ```

- Try to segment the images by providing training examples for four
  classes: 

  - Cell bodies: <img src="images/ml_seg_class/image1.png" height="100px" />

  - Synapses: <img src="images/ml_seg_class/image2.png" height="100px" />

  - Granule cell layer: <img src="images/ml_seg_class/image3.png" height="100px" />

  - ...and “*Other stuff*”

```{tip}
You can change the  contrast of your reference image by right clicking 'raw input' in the layer list on the lower-left of the screen and choosing 'Adjust Thresholds'
```

- Can you unambiguously define/draw labels for each of the classes?  
  - If not, would that be a problem? Why or why not?

- Pay attention to how the segmentation changes as you add training
  data. Can you understand why it changes as it does? If not ask!

- We learned about this in the lecture - what exactly is this “random
  forest pixel classifier” doing behind the scenes? If it’s not clear,
  ask!

- Using what you learned about filtering, what can you say about the
  features we might need for this segmentation task? What are the
  minimal set that might do? Try that out. If your segmentation needs
  improvement, what features might you need to add?

- When you think you’re done with training, try the model on the test
  image. How does it perform? What does this tell you about your data,
  the model, and how you trained it?

- Start a new pixel classification project with the same data so you can
  play around with how the training data affects the resulting model and
  classification.

  - What happens if you give the model a ton of examples of one class
    and very few examples of the others? Why is that?

  - What happens if you only annotate the brightest cell bodies you can
    find? Why?

- To export your pixel predictions, got to the 'Prediction Export' tab, then 'Choose Export Image Settings' and change the format of the output file to `.tif`.

- Now run your final classifier on the 'Test' data (go back to 'Input Data' and add it if you haven't done it before) and export the resulting predictions by clicking on `Export` next to the name of the Test image .

```{admonition} **Class Challenge** (Optional)  
:class: important

Let's make this (even more) interesting! Try to come up with a 'good enough' classifier with as few total annotations as possible. What strategy/reasoning did you use to get there? Compare your results with your labmates and see who manages to win the coveted **"Least pixels for good classification" Crown** :crown: this year!

If you want to submit your classification model for the competition, save your project as `YOUR_NAME_2025.ilp` and upload it together with an example of your classification to this [Google Drive folder](https://drive.google.com/drive/folders/12V2UAylQlCgq39ScCftEHK-3CxCFQd8I?usp=drive_link). 
```
---

## **Cell Classification** 

### Learning Objectives

- Segment cells
- Extract features from cells
- Feature-based cell classification

**Lab Data** in [this folder](https://tinyurl.com/QI2025AnalysisLabData) (under Machine_Learning/Cell_Classifier)

### Segment Nuclei for Feature Extraction

- Start a new project in ilastik using the workflow “Pixel
  Classification + Object Classification”

- Use the file in Machine_Learning/Cell_Classifier/. The first step will
  be to train a pixel classifier to segment the nuclei, just like you
  did in the previous exercise.

- Why do we need to segment the cells before we classify them? Are there
  other approaches that might let us classify cells without segmenting
  them? What are the pros and cons (ask us to discuss if it’s not
  clear!)

- Once you have a good pixel classifier, you can move on to the
  thresholding step to convert your probabilities into a mask.

### Classifying Mitotic and Interphase Cells

- Now, you can follow [this tutorial](https://bit.ly/35BXgVx), starting at
  the “From segmentation to objects” step.

- Can you find a minimal set of features that will classify these cells
  into dividing and non-dividing? Does adding more features help? Is
  there any reason we might not want to add *all* the features?

- If you only label cells in the first few frames of the time-series,
  are your classifications accurate at the end? What does this tell you
  about your model? Do you think the model would work well on a
  different image time series? If not, what might improve its
  performance on new data?

---

```{tip}
Did you know that there are ilastik image reading and model running plugins for both [Fiji](https://www.ilastik.org/documentation/fiji_export/plugin) and [CellProfiler](https://plugins.cellprofiler.org/supported_plugins.html)? They can be helpful for complex workflows.

Once you've trained an ilastik classifier, you can also export the images in ilastik's  [batch processor](https://www.ilastik.org/documentation/basics/batch) for your records and/or to interface with downstream programs. See [this tutorial](https://tutorials.cellprofiler.org/#pixel-based-classification) for an example of an ilastik-to-CellProfiler workflow.
```

---

##  **Bonus Exercises - Play with Labkit (pixel classification in Fiji)**

```{margin} Want to learn more about Labkit?
Check out the [documentation](https://imagej.net/plugins/labkit/), or check them out on [the image.sc forum!](https://forum.image.sc/tag/Labkit)
```

- You find Labkit on an update site called “Labkit”. Ask us if you have
  trouble installing it. Try it with the same training data and see how it performs!
  (We will also be using this software in lab 6!)

---

## **Bonus Exercises - Challenging Segmentations**

- Pick the hardest image you’ve seen in the course (maybe the neuron images from Lab 2?) and try to segment it
  using the ilastik RF pixel classifier. Can you crack it? Can you
  understand why or why not?

---

## **Bonus Exercises - Breaking it**

- Can you find an image segmentation or object classification problem
  that you *can’t* solve with ilastik? If you don’t have time to
  actually try it, can you think about what sort of problem might be
  unsolvable by any pixel classification approach?
