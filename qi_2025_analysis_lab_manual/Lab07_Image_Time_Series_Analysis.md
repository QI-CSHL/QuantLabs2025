# Image Time Series Analysis

*Lab authors: Hunter Elliott, Damian Dalle Nogare and Florian Jug* . 

<small>This file last updated 2025-04-10.</small>

<!-- Maybe add some UlTrac and/or TracX -->

---

## **Part 1: Photobleaching**

### Learning Objectives

- Apply and scrutinize photobleach correction
- Double-normalized FRAP analysis

**Lab Data** in [this folder](https://tinyurl.com/QI2025AnalysisLabData) (Time_Series)

Remember to **unzip** the data folder after downloading.

---

### Bleach Correction

In this portion of the lab, you will compensate for photobleaching by
fitting an exponential model to the decaying fluorescence intensity, and
then correcting for this decay.

Load the data:

- Go to folder `Time_Series > BleachCorrection`

- Load `TRITC` into Fiji by dragging the folder into the Fiji window
  (click yes when asked to open as a stack, but leave the checkboxes
  unchecked).

Apply Fiji's built-in photobleach correction plugin

- `Image > Adjust > Bleach Correction`

- Correction Method: Exponential Fit. Press OK.

- This plugin outputs a corrected image as well the fitted exponential
  decay. Since you have not selected an ROI it is using the average
  intensity in the entire image to perform the correction. Have a look
  at the corrected image: does it look like what you would expect?

- You can select a region and measure its intensity over time by drawing
  an ROI and then going to `Image > Stacks > Plot Z Axis Profile` (since
  there’s no Z axis this will actually plot a profile over time). Make a
  measurement like this of the center of a nucleus as well as the
  background. What do you see?

- Now, re-load the original dataset and select an ROI of a nucleus. Be
  sure it stays on the nucleus during the entire time series! Now
  perform the photobleach correction again. What do you see now? Why is
  this? What is the “right” way to photobleach correct these images?

```{tip}
You can easily compare two plot in Fiji by plotting them together on the same window. In the plot window, go to `Data > Add from Plot...` and select the data that you want to add to your current plot.
```

- Make a note of the decay rate (parameter 'b') from the exponential
  fit.

- Repeat for the `FITC` folder. Which image is bleaching faster?

- How well does the correction work on the FITC channel? Do we need to
  select an ROI here? Why or why not?

- We now have approximately the same average intensity at the end of the
  FITC time series. Is the SNR the same at the beginning and end? Does
  the photobleach correction affect the SNR?

### FRAP

In this lab we will be using a FRAP plugin for Fiji. Unfortunately it does not work on the most current version of Fji (*Welcome to the world of research software, you should be used to this by now!*). Inside the data folder for this lab, we have included a compatible version of Fiji with the plugin already installed in the folder `Time_Series/Fiji_for_FRAP`. 

- Load the data: `FRAP/33108 SU295 try1.tif` (Or use your own if you
  prefer)

- Create an ROI over the square bleached region. Add it to the ROI
  Manager (press ‘t’).

- Plot the intensity in this region over time 
  (`Image > Stacks > Plot Z-axis Profile`)

- Does the intensity recover to a stable value?

- Could we use a photobleaching correction like in the previous
  exercise? Why or why not?

Now we will run the FRAP plugin. You can find this under the plugin menu, under `FRAP profiler v2`. This is going to apply a similar analysis that you did above: photobleach correction, and then exponential fitting, but it will do it using a double-normalization approach. 

- First, perform a *uniform* background subtraction on your images by
  measuring the mean intensity in a region of the background and then
  going to `Process > Math > Subtract`. Why do we want to do a uniform
  subtraction rather than e.g. a rolling ball?

- You will need to have two ROIs in the ROI Manager. One for the square
  bleached region, and a second ROI for the whole cell (including the
  frapped region). If you can’t include the whole cell, just include a
  large region of the same cell which includes the frapped region. When
  might this non-whole-cell ROI be OK? When might it be problematic?

- Launch the FRAP Profiler plugin (`Plugins > FRAP Profiler v2`).

- Enter the correct time interval, so that your recovery rate is in the
  correct units.

- Run first with a single exponential. Compare the fitted curve to the
  raw data in the figure that is created (in the procFRAP and normFRAP
  windows). (Ideally we would compare this quantitatively but the plugin
  doesn’t support this :( )

- Re-run the plugin with double exponential. Is the fit better? A double
  exponential models a two step process, e.g., fast but weak association
  of the molecule of interest, followed by slower stronger association.

- What is the % mobile?


## **Part 2: Tracking**

### Learning Objectives

- (Semi-)automated Tracking with Mastodon in Fiji
- Using Trackmate{cite}`Tinevez2017-fb` from within Mastodon
- Tracking with ilastik{cite}`Berg2019-no`

---

### Overview

Tracking is an unsolved problem. It can be easy, but it usually is not.
If you need to reach flawless results, you have to curate manually.

This being said, manual tracking is very time consuming and
boring/annoying. Hence, we will be exploring semi-automatic tracking
regimes and also see how automated results can then be loaded for manual
curation.

More specifically, we will get to know ***Mastodon*** and learn more
about ***ilastik***. Mastodon is a Fiji plugin designed to deal with
truly huge time-lapses, which makes it an exquisitely valuable choice
for many tracking projects.

We know ilastik already from other exercises, but today we will explore
some of ilastik’s tracking features. The automated tracking in ilastik is
quite involved. Some key ideas we discussed in the lecture.

```{note}
This lab is intended to be interactive. If you find
something cool… don’t hold back… share it with others. These tools are
rather complex and we will certainly not be able to see all their
features.
```

### Get the data we need

Please start to download the `Time_Series/Tracking` folder from the [Data Lab Share](https://tinyurl.com/QI2025AnalysisLabData). It might take some time so better start early :\)

### (Semi-)automated Tracking in Mastodon


```{important}
Collect screenshots or remember what you learned here.
Be prepared to show the coolest thing you found out tomorrow morning
(you absolutely will be volunteered to share your findings ;)).
```

```{margin} Want to learn more about working with Mastodon?
Check out the [documentation](https://mastodon.readthedocs.io/en/latest/index.html), or check them out on [the image.sc forum!](https://forum.image.sc/tag/Mastodon)
```

#### Step 1: Start a Mastodon project from a single tiff File

1.  In Fiji go to: `Help > Update > Manage update sites`

    - Check boxes next to “Mastodon”.

    - `Close > Apply changes` and restart Fiji

2.  Open `drosophilus_floriansus.tif` in Fiji.

3.  In Fiji, start Mastodon by `Plugins > Trtacking > Mastodon > Mastodon launcher`

4.  In Mastodon, start a new project by clicking “new Mastodon project”.

    - Click on “use an image open in ImageJ” and select the Drosophila image
      you just opened in Fiji.

> <img src="images/time_series/image2.png" style="width:3.91146in;height:2.20594in" />

- Click on “create” to create the new Mastodon project. A new window
  will open.

> <img src="images/time_series/image1.png" style="width:2.49479in;height:2.49479in" />

- In this new window, you can now open BDVs (BigDataViewers ;\)) and TrackSchemes by clicking
  the respective buttons.

5.  Please try to navigate the data and play with a few BDV shortcuts.
    Navigating 3D space can be confusing and a few shortcuts are really,
    really valuable. Here a quick reminder to what we’ve seen before:

    - If you left click and drag, you can reslice in arbitrary
      directions.  
      Note the overview sketch on the top left… very helpful!

    - Right click and drag is moving the volume around.

    - Shift + X/Y/Z is rotating between the three exes parallel
      directions. Also here these rotations happen around the current
      mouse location.

    - The mouse wheel is shifting the visible image plane in and out.

    - Shift + ctrl + mouse wheel (on windows) is zooming in and out.

    - Spend some additional time to internalize [some more shortcuts](https://imagej.net/BigDataViewer#Basic_Navigation)

#### Step 2: Manually track some cells

1.  Check the **Mastodon cheat sheet** (`Time_Series/Tracking/Mastodon_cheatsheet.pdf`)

2.  Figure out how to:

```{tip}
Many of these tools work only if you are hovering you mouse pointer on the selection of interest.
```

  1.  Manually add and link cell detections (**a**):
      - Hover your mouse on a cell and press **a** to add a new detection
      - Hover your mouse on an existing detection and press & hold **a** to go to the next frame. Move your mouse to the new location of the cell and release **a** to add a new detection. The new detection wil be linked to the previous one.
      - Repeat the process to create a track
      - If you relese **a** on a previously existing detection, it will link it to your track.

  2.  Change the detection size to better fit the data (**q/e**).

  3.  Move individual detections around (**hold space + drag**)

  4.  Look at the detections you created using the TrackSceme view


```{margin}
See the three lock symbols on the top of some of Mastodon's screens?
All views that have the same 'lock' activated will be linked!
Try 'linking' the BDV and Trackscheme or adding a new BDV window to visualize your sample across the Z axis and 'locking' it too. Do you find it useful?
```

  5.  Delete detections/links (**d**)

  6.  Figure out what the ‘context’ in a TrackSceme controls (you will
      need to have tracked some lineages for this).

  7.  Did you spot the undo function yet? (**Ctrl + z**)

  8. Ask yourself how long it would take to manually track every
      single nucleus in this really quite small dataset. Once this
      sunk in, pity everyone who has to do that for their project.

#### Step 3: Semi-automatic tracking to speed things up

1.  Check the **instructions** file (`Time_Series/Tracking/Mastodon_Manual_SemiAutoTracking.pdf`) for the steps to set up and try the Semi-automatic tracking. 

```{warning}
These instructions are slightly outdated, screenshots will likely look a bit different than the Mastodon you are using today… 
```

2.  Now speed up your tracking work and try to find an efficient way to
    navigate Mastodon. (Remember, tomorrow we will volunteer people to
    show and compare found ways to track fast and efficiently…)

#### Step 4: Automatic tracking using the TrackMate plugin


```{margin} Want to learn more about working with TrackMate?
Check out the [documentation](https://imagej.net/plugins/trackmate/) and https://imagej.net/plugins/trackmate/#documentation-and-tutorials, or check them out on [the image.sc forum!](https://forum.image.sc/tag/TrackMate)
```

1.  Use (in Mastodon, **not** Fiji): `Plugins > Tracking > Detection…`
    and detect cells in some part of the volume (use the ROI-box to select a
    part of the volume in the BDV).

2.  Choose a `Detector` (there might be a familiar name in there!) and play around with the `Estimated diameter` of the objects and the `Quality threshold` to automatically detect your cells. Hit `Preview` to visualize the result.

3.  Then use: `Plugins > Tracking > Linking…` and link those detections
    to each other. This will not lead to perfect results. Don’t worry
    about that!

3.  Play with options, check if you can improve results somehow. Check with your labmates what they have tried, what worked for them!

### Tracking in ilastik

Please go to
[the tracking documentation for ilastik](https://ilastik.github.io/documentation/tracking/tracking)
and follow the instructions.

```{margin} Want to learn more about working with ilastik?
Check out the [documentation and tutorials list](https://www.ilastik.org/documentation/), or check them out on [the image.sc forum!](https://forum.image.sc/tag/ilastik)
```

You can download plenty of [example datasets](https://ilastik.github.io/download.html) from the ilastik website, although it can be a bit slow. 

If so, you can find the same examples in the [Lab Data Share](https://tinyurl.com/QI2025AnalysisLabData) under `Time_Series/Tracking/ilastik_tracking_projects` (note that each one is a folder containing several files):

1.  2D+t tracking example for ilastik

2.  3D+t tracking example for ilastik

The 2D example will be way faster to work with, but please choose any of
the two. After the download is completed, unzip the file and open the project file
“***conservationTracking.ilp***” in ilastik (Project \> Open project).

```{tip}
Please collect some screenshots or remember what you liked most. Be
prepared to show the coolest thing you found out tomorrow morning. Some
of you will be volunteered to share their findings… ;)
```

Some things to try after you start feeling comfortable with ilastik.
There is much more interesting stuff to explore. Here some inspiration:

```{margin}
Does this data look familiar? Maybe from the object classification exercise in Ilastik from the ML lab?
If you're curious, you can check all the previous steps that have already been done in the Tracking pipeline, like thresholding, identifying divisions, categorizing double and triple nuclei.  
```

1.  For the 2D+t example…

    1.  Can you spot some cells in the last time-point that are actually
        mergers of more than 2 cells? Can you change this somehow and
        make ilastik also detect mergers of more then 2 objects?

    2.  Please play with the individual steps and observe the many image
        layers. Which one might be particularly useful? When?

2.  For the 3D+t example…

    1.  How long did it take to track automatically? Is there a way to
        speed this up? (Ask the online manual…)

If you want to export the result of your tracking, you can do so in the `Tracking Result Export` tab. 
- Choose 'Tracking Result' as your source.
- Go to `Choose Image Export Settings...` and choose a suitable format. The default `.h5` format cannot be easily read in Fiji, maybe try `.tif sequence` (because this is time lapse data). Also choose the output directory and how the output images should be named.

```{tip}
Ilastik has some useful *terms* that make naming files easy:
- `{dataset_dir}` replaces the full path to where your input data is stored
- `{nickname}` replaces the name of the file being processed (excluding extension)
- `{result_type}` replaces the type of result you're exporting (in this case it would be 'Tracking_result') 
- `{slice_index}` is the index for the frame (or slice if this was 3D) being exported (only for image sequence exports)

You can learn more about the export setting for Ilastik from their [documentation](https://www.ilastik.org/documentation/basics/export#settings).
```

### Tracking with btrack in napari

In napari, there is a quite capable plugin for object tracking called `btrack`.
Let us install it and then use it.


```{margin} Want to learn more about working with btrack?
Check out the [documentation](https://btrack.readthedocs.io/en/latest/user_guide/index.html)!
```

**Installation of btrack and napari from scratch**
- Open a new Miniforge Prompt and install all we need by executing...
  - `conda create -y -n btrack -c conda-forge python=3.11`
  - `conda activate btrack`
  - `conda install -y pip`
  - `pip install btrack[napari]`
  - `conda install -y -c conda-forge napari pyqt`

**Using btrack to track objects we segmented**
- Next, we need a dataset to track. Actually, we need not only a dataset, we need
  data that is segmented. Since we learned about segmentation before, here we will
  simply use one of the btrack example datasets. 
  You can find a file called `segmented_nuclei.tif` it in the folder `btrack_masks` in the [Lab Data share](https://tinyurl.com/QI2025AnalysisLabData).
- Start btrack by starting napari (execute `napari` from within the *btrack* conda 
  environment we installed above).
- Start btrack via `Plugins > Track (btrack)`.
- Drag and drop the masks file into napari.

  ```{warning}
  Napari does not recognize this tiff file as labels. You must manually convert 
  to labels by right clicking on the layer and select `Convert to labels`.
  ```

- In the btrack side-bar, select the segmentation masks in the freshly coverted labels 
  layer and click on `Track`.
- Since this is one of the canned examples of btrack, the result is naturally quite good.
  If you have any labels for any other dataset - try it!
- The real work starts now: after objects are tracked, how would we continue analyzing 
  these tracks?

```{tip}
Tracking in napari is a thing. Kind of. 
If you want to learn more about it, maybe [this](https://focalplane.biologists.com/2023/06/01/tracking-in-napari/) is a good place to start.
```