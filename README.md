# HyperReenact: One-Shot Reenactment via Jointly Learning to Refine and Retarget Faces

Authors official PyTorch implementation of the **[HyperReenact: One-Shot Reenactment via Jointly Learning to Refine and Retarget Faces (ICCV 2023)](https://arxiv.org/abs/2307.10797)**. If you use this code for your research, please [**cite**](#citation) our paper.

<p align="center">
<img src="images/architecture.png" style="width: 750px"/>
</p>

>**HyperReenact: One-Shot Reenactment via Jointly Learning to Refine and Retarget Faces**<br>
> Stella Bounareli, Christos Tzelepis, Vasileios Argyriou, Ioannis Patras, Georgios Tzimiropoulos<br>
>
> **Abstract**: In this paper, we present our method for neural face reenactment, called HyperReenact, that aims to generate realistic 
            talking head images of a source identity, driven by a target facial pose. Existing state-of-the-art face reenactment methods train 
            controllable generative models that learn to synthesize realistic facial images, yet producing reenacted faces that are prone to significant 
            visual artifacts, especially under the challenging condition of extreme head pose changes, or requiring expensive few-shot fine-tuning to better preserve 
            the source identity characteristics. We propose to address these limitations by leveraging the photorealistic generation ability and the disentangled properties of a pretrained StyleGAN2 generator, by first inverting the real images into its latent space and then using a hypernetwork to perform: (i) refinement of the source identity characteristics and (ii) facial pose re-targeting, eliminating this way the dependence on external editing methods that typically produce artifacts. Our method operates under the one-shot setting (i.e., using a single source frame) and allows for cross-subject reenactment, 
            without requiring any subject-specific fine-tuning. We compare our method both quantitatively and qualitatively against several state-of-the-art 
            techniques on the standard benchmarks of VoxCeleb1 and VoxCeleb2, demonstrating the superiority of our approach in producing artifact-free images, 
            exhibiting remarkable robustness even under extreme head pose changes. 

<a href="https://arxiv.org/abs/2307.10797"><img src="https://img.shields.io/badge/arXiv-2307.10797-b31b1b.svg" height=22.5></a>
<a href="https://stelabou.github.io/hyperreenact.github.io/"><img src="https://img.shields.io/badge/Page-Demo-darkgreen.svg" height=22.5></a>


## Neural Face Reenactment Results

<p align="center">
<img src="images/self.gif" style="height: 150px"/>
<img src="images/cross.gif" style="height: 150px"/>
</p>



# Installation

* Python 3.5+ 
* Linux
* NVIDIA GPU + CUDA CuDNN
* Pytorch (>=1.5)

We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/).  

```
conda create -n hyperreenact_env python=3.8
conda activate hyperreenact_env
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```

# Pretrained Models

We provide a StyleGAN2 model trained using [StyleGAN2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) and an [e4e](https://github.com/omertov/encoder4editing) inversion model trained on [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) dataset. We also provide our HyperNetwork trained on VoxCeleb dataset.


| Path | Description
| :--- | :----------
|[StyleGAN2-VoxCeleb1](https://drive.google.com/file/d/1cBwIFwq6cYIA5iR8tEvj6BIL7Ji7azIH/view?usp=sharing)  | StyleGAN2 trained on VoxCeleb1 dataset.
|[e4e-VoxCeleb1](https://drive.google.com/file/d/1TRATaREBi4VCMITUZV0ZO2XFU3YZKGlQ/view?usp=share_link)  | e4e trained on VoxCeleb1 dataset.
|[HyperReenact-net](https://drive.google.com/file/d/1BUp6S3Wf2SeM3a-b_7mkKZyaXAxKhLrI/view?usp=sharing)  | hypernetwork trained on VoxCeleb1 dataset.

# Auxiliary Models

We provide additional auxiliary models needed during training/inference.

| Path | Description
| :--- | :----------
|[face-detector](https://drive.google.com/file/d/1IWqJUTAZCelAZrUzfU38zK_ZM25fK32S/view?usp=share_link)  | Pretrained face detector taken from [face-alignment](https://github.com/1adrianb/face-alignment).
|[ArcFace Model](https://drive.google.com/file/d/1F3wrQALEOd1Vku8ArJ_Gn4T6U3IX7Pz7/view?usp=sharing)  | Pretrained ArcFace model taken from [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) used as our Appearance encoder
|[DECA model](https://drive.google.com/file/d/1BHVJAEXscaXMj_p2rOsHYF_vaRRRHQbA/view?usp=sharing)  | Pretrained model taken from [DECA](https://github.com/YadiraF/DECA) used as our Pose encoder. Extract data.tar.gz under  `./pretrained_models`.

# Inference 

Please download all models and save them under `./pretrained_models` path.

Given as input a source frame (.png or .jpg) and a target video (.png or .jpg, .mp4 or a directory with images), reenact the source face. 
```
python run_inference.py --source_path ./inference_examples/source.png \
							--target_path ./inference_examples/target_video_1.mp4 \
							--output_path ./results --save_video
```

## Citation

[1] Stella Bounareli, Christos Tzelepis, Argyriou Vasileios, Ioannis Patras, and Georgios Tzimiropoulos. HyperReenact: One-Shot Reenactment via Jointly Learning to Refine and Retarget Faces. IEEE International Conference on Computer Vision (ICCV), 2023.

Bibtex entry:

```bibtex
@InProceedings{bounareli2023hyperreenact,
    author    = {Bounareli, Stella and Tzelepis, Christos and Argyriou, Vasileios and Patras, Ioannis and Tzimiropoulos, Georgios},
    title     = {HyperReenact: One-Shot Reenactment via Jointly Learning to Refine and Retarget Faces},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2023},
}
```



## Acknowledgment

This research was supported by the EU's Horizon 2020 programme H2020-951911 [AI4Media](https://www.ai4media.eu/) project.


