# Uniformizing Techniques to Process CT scans with 3D CNNs for Tuberculosis Prediction

This code is part of the supplementary materials for our paper titled *Uniformizing Techniques to Process CT scans with 3D CNNs for Tuberculosis Prediction* accepted for publication in MICCAI 2020 workshop on Predictive Intelligence in Medicine.


## Abstract

A common approach to medical image analysis on volumetric data uses deep 2D convolutional neural networks (CNNs). This is largely attributed to the challenges imposed by the nature of the 3D data: variable volume size, GPU exhaustion during optimization. However, dealing with the individual slices independently in 2D CNNs deliberately discards the depth information which results in poor performance for the intended task. Therefore, it is important to develop methods that not only overcome the heavy memory and computation requirements but also leverage the 3D information. To this end, we evaluate a set of volume uniformizing methods to address the aforementioned issues. The first method involves sampling information evenly from a subset of the volume. Another method exploits the full geometry of the 3D volume by interpolating over the z-axis. We demonstrate performance improvements using controlled ablation studies as well as put this approach to the test on the ImageCLEF Tuberculosis Severity Assessment 2019 benchmark and report performance on the test set of 73% AUC and binary classification accuracy of 67.5% beating all methods which leveraged only image information (without using clinical meta-data) and achieve 5-th position overall, post-challenge. All codes are made available at https://github.com/hasibzunair/uniformizing-3D.


### Dependencies

*    Ubuntu 14.04
*    Windows 8
*    Python 3.6
*    Tensorflow: 2.0.0
*    Keras: 2.3.1

### Environment setup

You can create the appropriate conda environment by running

`conda env create -f environment.yml`


### Directory Structure & Usage

* Run notebook in order
* `others`: Contains helper codes to preprocess and visualize samples in dataset.


### This is an extension of previous work

More details at this [link](https://github.com/hasibzunair/tuberculosis-severity)


```
Zunair,  H.,  Rahman,  A.,  Mohammed,  N.:   Estimating  Severity  from  CT  Scans
of  Tuberculosis  Patients  using  3D  Convolutional  Nets  and  Slice  Selection.   In:
CLEF2019  Working  Notes.  Volume  2380  of  CEUR  Workshop  Proceedings.,
Lugano, Switzerland, CEUR-WS.org
<http://ceur-ws.org/Vol-2380>(September 9-12 2019) 
```
Previous paper published in CEUR-WS. Paper can be found at [CLEF Working Notes 2019](http://www.dei.unipd.it/~ferro/CLEF-WN-Drafts/CLEF2019/paper_77.pdf) under the section ImageCLEF - Multimedia Retrieval in CLEF.


### License

Your driver's license.


