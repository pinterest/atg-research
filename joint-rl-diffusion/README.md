<div align="center">

<!-- TITLE -->
# **Large-scale Reinforcement Learning for Diffusion Models**

<img src="https://github.com/pinterest/atg-research/blob/main/docs/rl-diffusion/static/rl_vis_progression/tasks_progression_teaser.png" width=60% height=60%>

[![arXiv](https://img.shields.io/badge/cs.LG-arXiv:2401.12244-b31b1b.svg)](https://arxiv.org/abs/2401.12244)
[![Website](https://img.shields.io/badge/🌎-Website-blue.svg)](https://pinterest.github.io/atg-research/rl-diffusion/)
</div>

This is the official implementation of our paper [Large-scale Reinforcement Learning for Diffusion Models](https://arxiv.org/abs/2401.12244) by Yinan Zhang, Eric Tzeng, Yilun Du, and Dmitry Kislyuk.


<!-- DESCRIPTION -->
## Abstract
Text-to-image diffusion models are a class of deep generative models that have demonstrated an impressive capacity for high-quality image generation. However, these models are susceptible to implicit biases that arise from web-scale text-image training pairs and may inaccurately model aspects of images we care about. This can result in suboptimal samples, model bias, and images that do not align with human ethics and preferences. In this paper, we present an effective scalable algorithm to improve diffusion models using Reinforcement Learning (RL) across a diverse set of reward functions, such as human preference, compositionality, and fairness over millions of images. We illustrate how our approach substantially outperforms existing methods for aligning diffusion models with human preferences. We further illustrate how this substantially improves pretrained Stable Diffusion (SD) models, generating samples that are preferred by humans 80.3% of the time over those from the base SD model while simultaneously improving both the composition and diversity of generated samples.

## Code

### Coming Soon
- Model Weights
- Other Reward functions


### Training Code


#### Human Preference Reward model
The hyperparameters and the training setting for human preference fine-tuning can be found in [configs/imagereward_train_configs.yaml](https://github.com/pinterest/atg-research/blob/main/joint-rl-diffusion/configs/imagereward_train_configs.yaml). Run the following command to train SD2 model with ImageReward: 


```bash
accelerate launch --multi_gpu ./train.py --config_name image_reward
```


### Acknowledgement

We use the opensourced [ImageReward](https://github.com/THUDM/ImageReward/tree/main) model for optimizing human preferences. The teaser video on our website is built using the animation script provided by the authors of [DDPO](https://github.com/jannerm/ddpo).

We would like to thank Kevin Black, DDPO and ImageReward teams for opensourcing their code.


## Citation

If you find this work useful in your research, please cite:

```bibtex
@misc{zhang2024largescale,
      title={Large-scale Reinforcement Learning for Diffusion Models}, 
      author={Yinan Zhang and Eric Tzeng and Yilun Du and Dmitry Kislyuk},
      year={2024},
      eprint={2401.12244},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



