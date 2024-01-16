# nnU-Net with CADD-UNet

We propose a novel network, CADD-UNet, optimized for small object segmentation, based on the nnUNet pipeline that is tailored for medical image segmentation. The usage guidelines for the proposed network are outlined below:


<Train for one class (Ureter)>

1. Setting Dataset Path
Set the path from the below script.
[paths.py]

2. Dataset conversion (ex, Task 300)
```bash
dataset_conversion/Task_300_Ureter.py
```

3. Data Preprocessing
```bash
experimnet_planning/nnUNet_plan_and_preprocess.py -t 300
experimnet_planning/nnUNet_plan_and_preprocess.py -t 300 -pl3d ExperimentPlanner_Double_Dense_3DUNet_v21
```

4. Train
```bash
/home/sukmin/nnUNet_CADD/run/run_training_DP.py 
-network 3d_fullres -network_trainer HasTrainer_DP_CADD -task Task300_Ureter -fold 4 -gpu 1 -p nnUNetPlans_Double_DenseUNet_v2.1_2
```

5. Inference
```bash
/home/sukmin/nnUNet_CADD/inference/predict_simple.py 
-i {Input Path} -o {Output Path}  -t Task273_Urinary -tr HasTrainer_DP_CADD_3kid -m 3d_fullres -chk model_best -f 4 -p nnUNetPlans_Double_DenseUNet_v2.1_2
```



<Train for multi-class (Urinary system)>

1. Setting Dataset Path
Set the path from the below script.
[paths.py]

2. Dataset conversion (ex, Task 310)
```bash
dataset_conversion/Task_310_Urinary.py
```
3. Data Preprocessing
```bash
experimnet_planning/nnUNet_plan_and_preprocess.py -t 310
experimnet_planning/nnUNet_plan_and_preprocess.py -t 310 -pl3d ExperimentPlanner_Double_Dense_3DUNet_v21
```

4. Train
```bash
/home/sukmin/nnUNet_CADD/run/run_training_DP.py 
-network 3d_fullres -network_trainer HasTrainer_DP_CADD_3kid -task Task_310_Urinary -fold 4 -gpu 1 -p nnUNetPlans_Double_DenseUNet_v2.1_2
```

5. Inference
```bash
/home/sukmin/nnUNet_CADD/inference/predict_simple.py 
-i {Input Path} -o {Output Path} -t Task_310_Urinary -tr HasTrainer_DP_CADD_3kid -m 3d_fullres -chk model_best -f 4 -p nnUNetPlans_Double_DenseUNet_v2.1_2
```






# About nnU-Net and installation

In 3D biomedical image segmentation, dataset properties like imaging modality, image sizes, voxel spacings, class 
ratios etc vary drastically.
For example, images in the [Liver and Liver Tumor Segmentation Challenge dataset](https://competitions.codalab.org/competitions/17094) 
are computed tomography (CT) scans, about 512x512x512 voxels large, have isotropic voxel spacings and their 
intensity values are quantitative (Hounsfield Units).
The [Automated Cardiac Diagnosis Challenge dataset](https://acdc.creatis.insa-lyon.fr/) on the other hand shows cardiac 
structures in cine MRI with a typical image shape of 10x320x320 voxels, highly anisotropic voxel spacings and 
qualitative intensity values. In addition, the ACDC dataset suffers from slice misalignments and a heterogeneity of 
out-of-plane spacings which can cause severe interpolation artifacts if not handled properly. 

In current research practice, segmentation pipelines are designed manually and with one specific dataset in mind. 
Hereby, many pipeline settings depend directly or indirectly on the properties of the dataset 
and display a complex co-dependence: image size, for example, affects the patch size, which in 
turn affects the required receptive field of the network, a factor that itself influences several other 
hyperparameters in the pipeline. As a result, pipelines that were developed on one (type of) dataset are inherently 
incomaptible with other datasets in the domain.

**nnU-Net is the first segmentation method that is designed to deal with the dataset diversity found in the domain. It 
condenses and automates the keys decisions for designing a successful segmentation pipeline for any given dataset.**

nnU-Net makes the following contributions to the field:

1. **Standardized baseline:** nnU-Net is the first standardized deep learning benchmark in biomedical segmentation.
Without manual effort, researchers can compare their algorithms against nnU-Net on an arbitrary number of datasets 
to provide meaningful evidence for proposed improvements. 
2. **Out-of-the-box segmentation method:** nnU-Net is the first plug-and-play tool for state-of-the-art biomedical 
segmentation. Inexperienced users can use nnU-Net out of the box for their custom 3D segmentation problem without 
need for manual intervention. 
3. **Framework:** nnU-Net is a framework for fast and effective development of segmentation methods. Due to its modular 
structure, new architectures and methods can easily be integrated into nnU-Net. Researchers can then benefit from its
generic nature to roll out and evaluate their modifications on an arbitrary number of datasets in a 
standardized environment.  

For more information about nnU-Net, please read the following paper:


    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method 
    for deep learning-based biomedical image segmentation. Nature Methods, 1-9.

Please also cite this paper if you are using nnU-Net for your research!


# Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  * [How to run nnU-Net on a new dataset](#how-to-run-nnu-net-on-a-new-dataset)
    + [Dataset conversion](#dataset-conversion)
    + [Experiment planning and preprocessing](#experiment-planning-and-preprocessing)


# Installation
nnU-Net has been tested on Linux (Ubuntu 16, 18 and 20; centOS, RHEL). We do not provide support for other operating 
systems.

nnU-Net requires a GPU! For inference, the GPU should have 4 GB of VRAM. For training nnU-Net models the GPU should have at 
least 10 GB (popular non-datacenter options are the RTX 2080ti, RTX 3080 or RTX 3090). Due to the use of automated mixed 
precision, fastest training times are achieved with the Volta architecture (Titan V, V100 GPUs) when installing pytorch 
the easy way. Since pytorch comes with cuDNN 7.6.5 and tensor core acceleration on Turing GPUs is not supported for 3D 
convolutions in this version, you will not get the best training speeds on Turing GPUs. You can remedy that by compiling pytorch from source 
(see [here](https://github.com/pytorch/pytorch#from-source)) using cuDNN 8.0.2 or newer. This will unlock Turing GPUs 
(RTX 2080ti, RTX 6000) for automated mixed precision training with 3D convolutions and make the training blistering 
fast as well. Note that future versions of pytorch may include cuDNN 8.0.2 or newer by default and 
compiling from source will not be necessary.
We don't know the speed of Ampere GPUs with vanilla vs self-compiled pytorch yet - this section will be updated as 
soon as we know.

For training, we recommend a strong CPU to go along with the GPU. At least 6 CPU cores (12 threads) are recommended. CPU 
requirements are mostly related to data augmentation and scale with the number of input channels. They are thus higher 
for datasets like BraTS which use 4 image modalities and lower for datasets like LiTS which only uses CT images.

We very strongly recommend you install nnU-Net in a virtual environment. 
[Here is a quick how-to for Ubuntu.](https://linoxide.com/linux-how-to/setup-python-virtual-environment-ubuntu/)
If you choose to compile pytorch from source, you will need to use conda instead of pip. In that case, please set the 
environment variable OMP_NUM_THREADS=1 (preferably in your bashrc using `export OMP_NUM_THREADS=1`). This is important!

Python 2 is deprecated and not supported. Please make sure you are using Python 3.

1) Install [PyTorch](https://pytorch.org/get-started/locally/). You need at least version 1.6
2) Install nnU-Net depending on your use case:
    1) For use as **standardized baseline**, **out-of-the-box segmentation algorithm** or for running **inference with pretrained models**:
      
        ```pip install nnunet```
    
    2) For use as integrative **framework** (this will create a copy of the nnU-Net code on your computer so that you can modify it as needed):
          ```bash
          git clone https://github.com/MIC-DKFZ/nnUNet.git
          cd nnUNet
          pip install -e .
          ```
3) nnU-Net needs to know where you intend to save raw data, preprocessed data and trained models. For this you need to 
set a few of environment variables. Please follow the instructions [here](documentation/setting_up_paths.md).
4) (OPTIONAL) Install [hiddenlayer](https://github.com/waleedka/hiddenlayer). hiddenlayer enables nnU-net to generate 
plots of the network topologies it generates (see [Model training](#model-training)). To install hiddenlayer, 
run the following commands:
    ```bash
    pip install --upgrade git+https://github.com/nanohanno/hiddenlayer.git@bugfix/get_trace_graph#egg=hiddenlayer
    ```

Installing nnU-Net will add several new commands to your terminal. These commands are used to run the entire nnU-Net 
pipeline. You can execute them from any location on your system. All nnU-Net commands have the prefix `nnUNet_` for 
easy identification.

Note that these commands simply execute python scripts. If you installed nnU-Net in a virtual environment, this 
environment must be activated when executing the commands.

All nnU-Net commands have a `-h` option which gives information on how to use them.

A typical installation of nnU-Net can be completed in less than 5 minutes. If pytorch needs to be compiled from source 
(which is what we currently recommend when using Turing GPUs), this can extend to more than an hour.

# Usage
To familiarize yourself with nnU-Net we recommend you have a look at the [Examples](#Examples) before you start with 
your own dataset.

## How to run nnU-Net on a new dataset
Given some dataset, nnU-Net fully automatically configures an entire segmentation pipeline that matches its properties. 
nnU-Net covers the entire pipeline, from preprocessing to model configuration, model training, postprocessing 
all the way to ensembling. After running nnU-Net, the trained model(s) can be applied to the test cases for inference. 

### Dataset conversion
nnU-Net expects datasets in a structured format. This format closely (but not entirely) follows the data structure of 
the [Medical Segmentation Decthlon](http://medicaldecathlon.com/). Please read 
[this](documentation/dataset_conversion.md) for information on how to convert datasets to be compatible with nnU-Net.

### Experiment planning and preprocessing
As a first step, nnU-Net extracts a dataset fingerprint (a set of dataset-specific properties such as 
image sizes, voxel spacings, intensity information etc). This information is used to create three U-Net configurations: 
a 2D U-Net, a 3D U-Net that operated on full resolution images as well as a 3D U-Net cascade where the first U-Net 
creates a coarse segmentation map in downsampled images which is then refined by the second U-Net.

Provided that the requested raw dataset is located in the correct folder (`nnUNet_raw_data_base/nnUNet_raw_data/TaskXXX_MYTASK`, 
also see [here](documentation/dataset_conversion.md)), you can run this step with the following command:

```bash
nnUNet_plan_and_preprocess -t XXX --verify_dataset_integrity
```

`XXX` is the integer identifier associated with your Task name `TaskXXX_MYTASK`. You can pass several task IDs at once.

Running `nnUNet_plan_and_preprocess` will populate your folder with preprocessed data. You will find the output in 
nnUNet_preprocessed/TaskXXX_MYTASK. `nnUNet_plan_and_preprocess` creates subfolders with preprocessed data for the 2D 
U-Net as well as all applicable 3D U-Nets. It will also create 'plans' files (with the ending.pkl) for the 2D and 
3D configurations. These files contain the generated segmentation pipeline configuration and will be read by the 
nnUNetTrainer (see below). Note that the preprocessed data folder only contains the training cases. 
The test images are not preprocessed (they are not looked at at all!). Their preprocessing happens on the fly during 
inference.

`--verify_dataset_integrity` should be run at least for the first time the command is run on a given dataset. This will execute some
 checks on the dataset to ensure that it is compatible with nnU-Net. If this check has passed once, it can be 
omitted in future runs. If you adhere to the dataset conversion guide (see above) then this should pass without issues :-)

Note that `nnUNet_plan_and_preprocess` accepts several additional input arguments. Running `-h` will list all of them 
along with a description. If you run out of RAM during preprocessing, you may want to adapt the number of processes 
used with the `-tl` and `-tf` options.

After `nnUNet_plan_and_preprocess` is completed, the U-Net configurations have been created and a preprocessed copy 
of the data will be located at nnUNet_preprocessed/TaskXXX_MYTASK.

Extraction of the dataset fingerprint can take from a couple of seconds to several minutes depending on the properties 
of the segmentation task. Pipeline configuration given the extracted finger print is nearly instantaneous (couple 
of seconds). Preprocessing depends on image size and how powerful the CPU is. It can take between seconds and several 
tens of minutes.
