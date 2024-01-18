Run the following command to finetune diffusion model with pretrained ImageReward model:
```bash
accelerate launch --multi_gpu trainer/ppytorch/synthesis/rl_release_code/train.py --config_name image_reward
```
