# Large-scale Reinforcement Learning for Diffusion Models

Training code for the paper [Large-scale Reinforcement Learning for Diffusion Models](https://pinternal/joint-rl-diffusion.github.io/).



Run the following command to finetune diffusion model with pretrained ImageReward model:
```bash
accelerate launch --multi_gpu ./train.py --config_name image_reward
```
