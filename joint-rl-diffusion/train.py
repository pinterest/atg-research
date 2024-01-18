import os
import tempfile
import time
import json
import hashlib
import torch
import torch.nn.functional as F
import random
import pandas as pd
import accelerate
import yaml
import argparse
from munch import munchify
from urllib.request import urlretrieve
from PIL import Image
from accelerate import Accelerator
from packaging import version
from torchvision import transforms
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
from transformers import CLIPTextModel
from transformers import CLIPTokenizer
import diffusers
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers import AutoencoderKL
from diffusers import DDPMScheduler
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.training_utils import EMAModel
from diffusers.loaders import LoraLoaderMixin
from pipeline_with_prob import StableDiffusionPipelineCustom
from ddim_with_prob import DDIMSchedulerCustom
import ImageReward.ImageReward as RM

class DiffusionDB(torch.utils.data.dataset.Dataset):
    # val_num refers to how many prompts are left for validation purpose and thus not used for training
    def __init__(self, split="train", val_num=2000):
        # Download the parquet table
        table_url = f'https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet'
        urlretrieve(table_url, 'metadata.parquet')
        # Read the table using Pandas
        metadata_df = pd.read_parquet('./metadata.parquet')

        all_db_prompts = list(metadata_df["prompt"])
        random.seed(1928)
        random.shuffle(all_db_prompts)
        # if the seed is fixed, the split of train/val is fixed as well
        if split == "train":
            self.db_prompts = all_db_prompts[:-val_num]
        else:
            assert split == "val"
            self.db_prompts = all_db_prompts[-val_num:]

    def __getitem__(self, index):
        return {"text": self.db_prompts[index]}

    def __len__(self):
        return len(self.db_prompts)


def create_optimizer(model, config):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
        eps=config.adam_epsilon,
    )


def create_model(config):
    model = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name, subfolder="unet",
    )
    # Enable xformers efficient attention.
    if is_xformers_available() and config.enable_xformers_memory_efficient_attention:
        model.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")
    # Enable gradient checkpointing.
    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    # Enable attention slicing.
    if config.enable_attention_slicing:
        model.set_attention_slice("max")
    return model


def load_pipeline_s3(config):
    pipeline = StableDiffusionPipelineCustom.from_pretrained(config.pretrained_model_name, safety_checker=None, requires_safety_checker=False)
    return pipeline


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="image_reward", help="which training config to use")
    args = parser.parse_args()
    # mapping from config name to config path
    config_mapping = {"image_reward": "/machine-learning/trainer/ppytorch/synthesis/rl_release_code/configs/imagereward_train_configs.yaml"}
    # config_mapping = {"image_reward":  "./configs/imagereward_train_configs.yaml"}
    with open(config_mapping[args.config_name]) as file:
        config_dict= yaml.safe_load(file)
        config = munchify(config_dict)
    return config


def create_lr_scheduler(optimizer, config):
    if config.lr_scheduler.lower() == "constant":
        def lr_lambda(current_step: int):
            start_factor = config.lr_warmup_start_ratio or 0
            if current_step < config.lr_warmup_steps:
                return (1-start_factor) * (float(current_step) / float(max(1.0, config.lr_warmup_steps))) + start_factor
            return 1.0
        return LambdaLR(optimizer, lr_lambda)
    else:
        assert config.lr_scheduler.lower() == "cosineannealing"
        start_factor_custom = config.lr_warmup_start_ratio or 1e-12
        warmup = lr_scheduler.LinearLR(optimizer, start_factor=start_factor_custom, end_factor=1.0, total_iters=config.lr_warmup_steps)
        cosine_decay = lr_scheduler.CosineAnnealingLR(optimizer, T_max=1_000_000-config.lr_warmup_steps)
        scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine_decay], milestones=[config.lr_warmup_steps])
    return scheduler


def create_pipeline(config, accelerator, unet, ema_unet, vae, text_encoder, noise_scheduler):
    if config.use_ema:
        unet_config = UNet2DConditionModel.load_config(config.pretrained_model_name, subfolder="unet")
        unet_copy = UNet2DConditionModel.from_config(unet_config)
        ema_unet.copy_to(unet_copy.parameters())
    save_unet = unet_copy if config.use_ema else accelerator.unwrap_model(unet)
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained_model_name,
        text_encoder=text_encoder,
        vae=vae,
        unet=save_unet,
        # unet=self.accelerator.unwrap_model(self.ema_unet.averaged_model),
        scheduler=noise_scheduler,
    )

    return pipeline


def export_pipeline(config, accelerator, unet, ema_unet, vae, text_encoder, noise_scheduler, current_step):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = create_pipeline(config, accelerator, unet, ema_unet, vae, text_encoder, noise_scheduler)
        save_path = os.path.join(config.snapshot_dir, f"iteration_{current_step}")
        pipeline.save_pretrained(save_path)
        accelerator.print(f"Pipeline saved to {save_path}")


def save_training_state(accelerator, current_step, config):
    """
    Save a snapshot if necessary
    """
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(config.snapshot_dir, f"iter{current_step:08d}")
        accelerator.save_state(save_path)
        accelerator.print(f"Saved to {save_path}")


def load_image_reward_model(accelerator, config):
    reward_model = RM.load("ImageReward-v1.0").to(accelerator.device)
    if config.use_bfloat16:
        reward_model = reward_model.to(dtype=torch.bfloat16)
    reward_model.requires_grad_(False)
    return reward_model


def make_uncond_text(text_tokenizer, batch_size):
    uncond_prompt = text_tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=text_tokenizer.model_max_length,
        return_tensors="pt",
    )
    return uncond_prompt.input_ids


def tokenize_batch(batch, config, tokenizer):
    batch_text = batch['text']
    if config.skin_tone_diversity_sp_tuning and len(batch['text']) != config.batch_size:
        batch_text = random.choices(batch['text'], k=config.batch_size)
    assert len(batch_text) == config.batch_size

    prompt_tokens = tokenizer(
        batch_text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids

    batch["tokens"] = prompt_tokens
    batch["text"] = batch_text
    return batch


def sync_model_weights(pipeline, accelerator, unet):
    pipeline.unet.load_state_dict(accelerator.unwrap_model(unet).state_dict())


def calculate_mse_loss(data_batch, text_encoder, vae, accelerator, unet, noise_scheduler, config, encoder_hidden_states=None):
    # here the input data_batch can be data from pretraining dataset
    if not encoder_hidden_states:
        with torch.no_grad():
            encoder_hidden_states = text_encoder(data_batch["tokens"].to(accelerator.device))[0]

    if config.use_bfloat16:
        latents = vae.encode(
            data_batch["image"].to(accelerator.device, dtype=torch.bfloat16)
        ).latent_dist.sample()
    else:
        latents = vae.encode(data_batch["image"].to(accelerator.device)).latent_dist.sample()
    latents = latents * 0.18215

    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    # Sample a random timestep for each image
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Get the target for loss depending on the prediction type
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    # Predict the noise residual and compute loss
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    loss_mse = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    return loss_mse


def train():
    config = load_config()
    unet = create_model(config)
    pipeline = load_pipeline_s3(config)
    text_encoder, vae, tokenizer = pipeline.text_encoder, pipeline.vae, pipeline.tokenizer
    noise_scheduler = pipeline.scheduler

    if config.image_reward_tuning:
        train_dataloader = torch.utils.data.DataLoader(
            dataset=DiffusionDB(),
            batch_size=config.batch_size,
            shuffle=True)

    optimizer = create_optimizer(unet, config)
    lr_scheduler = create_lr_scheduler(optimizer, config)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )
    text_encoder = text_encoder.to(accelerator.device)
    text_encoder.requires_grad_(False)
    vae = vae.to(accelerator.device)
    vae.requires_grad_(False)
    pipeline.to(accelerator.device)
    if config.use_bfloat16:
        text_encoder = text_encoder.to(dtype=torch.bfloat16)
        vae = vae.to(dtype=torch.bfloat16)
    else:
        text_encoder = text_encoder.to(dtype=unet.dtype)
        vae = vae.to(dtype=unet.dtype)

    if config.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)
        ema_unet.to(accelerator.device)
        if config.use_bfloat16:
            ema_unet = ema_unet.to(dtype=torch.bfloat16)
        accelerator.register_for_checkpointing(ema_unet)
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if config.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                for model in models:
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if config.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model
        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    train_iter = iter(train_dataloader)

    unet.train()
    if not config.use_bfloat16:
        pipeline.unet.to(dtype=accelerator.unwrap_model(unet).dtype)

    if config.image_reward_tuning:
        image_reward_model = load_image_reward_model(accelerator, config)

    pipeline.unet.requires_grad_(False)
    pipeline.safety_checker = None
    pipeline.scheduler = DDIMSchedulerCustom.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(num_inference_steps=config.n_inference_steps,
                                          device=accelerator.device)

    # local_log_file_name = save_folder.strip("/").split("/")[-1] + ".txt"
    current_step = 0
    while config.max_iterations == 0 or current_step < config.max_iterations:
        with accelerator.accumulate(unet):
            start_time = time.time()
            try:
                batch = next(train_iter)
            except StopIteration:
                # StopIteration is thrown if dataset ends; reinitialize data loader
                train_iter = iter(train_dataloader)
                batch = next(train_iter)
            batch = tokenize_batch(batch, config, tokenizer)

            data_load_time = time.time() - start_time
            start_time = time.time()

            # Get the text embedding for conditioning
            with torch.no_grad():
                encoder_hidden_states = text_encoder(batch["tokens"].to(accelerator.device))[0]

            if config.image_reward_tuning or config.spatial_unidet_tuning or config.skin_tone_diversity_sp_tuning:
                sync_model_weights(pipeline, accelerator, unet)

                if config.use_pretrain_dataset:
                    pretrain_mse_loss = calculate_mse_loss(pretrain_batch, text_encoder, vae, accelerator, unet, noise_scheduler, config)
                    avg_loss_mse = accelerator.gather(
                        pretrain_mse_loss.repeat(pretrain_batch["image"].shape[0])).mean()
                    accelerator.print(f"step {current_step}\t",
                                      f"MSE loss:{round(avg_loss_mse.item(), 5)}\t")
                    accelerator.backward(pretrain_mse_loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                with torch.no_grad():
                    uncond_prompt_ids = make_uncond_text(tokenizer, config.batch_size)
                    uncond_prompt_embeds = text_encoder(uncond_prompt_ids.to(accelerator.device))[0]

                    final_latents, latents_lst, next_latents_lst, ts_lst, log_probs_lst, pred_img_t_ori = pipeline(
                        height=config.resolution,
                        width=config.resolution,
                        num_inference_steps=config.n_inference_steps,
                        guidance_scale=config.guidance_scale,
                        eta=config.eta,
                        prompt_embeds=encoder_hidden_states,
                        negative_prompt_embeds=uncond_prompt_embeds,
                    )
                    # final_latents: torch.Size([bs, 4, 64, 64]); latents_lst: torch.Size([bs, 50, 4, 64, 64]);
                    # next_latents_lst: torch.Size([bs, 50, 4, 64, 64])
                    # ts_lst: torch.Size([bs, 50])
                    # log_probs_lst: torch.Size([bs, 50])

                    if config.image_reward_tuning:
                        _, reward_scores = image_reward_model.inference_rank(batch["text"], pred_img_t_ori)
                    reward_scores_none = False
                    if set(reward_scores) == {None}:
                        reward_scores_none = True
                        reward_scores = [np.NaN] * config.batch_size
                    if isinstance(reward_scores, list):
                        reward_scores = torch.Tensor(reward_scores).to(accelerator.device)

                    all_rewards = accelerator.gather(reward_scores)
                    all_rewards_valid = all_rewards[~torch.isnan(all_rewards)]
                    avg_reward_step = all_rewards_valid.mean().item() / config.gradient_accumulation_steps
                    if not reward_scores_none:
                        # add a small offset 1e-7 to avoid denominator being 0
                        advantages = (reward_scores - torch.mean(all_rewards_valid)) / (
                                torch.std(all_rewards_valid) + 1e-7)
                        advantages = torch.clamp(advantages, -config.ADV_CLIP_MAX, config.ADV_CLIP_MAX)


                ddpo_ratio_vals_lst, ddpo_loss_vals_lst, mse_loss_vals_lst = [], [], []

                loop_train_steps = config.n_train_steps

                for i in random.sample(range(config.n_inference_steps), k=loop_train_steps):  # latents_lst.shape: torch.Size([6, 10, 4, 64, 64])
                    latents_i, next_latents_i, t_i = latents_lst[:, i], next_latents_lst[:, i], ts_lst[:, i]
                    # pre-processing of model input
                    if config.train_cfg:
                        prompt_embeds_input, latent_model_input, t_input = torch.cat(
                            [uncond_prompt_embeds, encoder_hidden_states]), torch.cat([latents_i] * 2), torch.cat([t_i] * 2)

                        if config.combine_DDPO_MSE or reward_scores_none:
                            latents_from_gt_image = vae.encode(
                                batch["image"].to(accelerator.device)).latent_dist.sample()
                            if batch["image"].shape[0] != config.batch_size:
                                latents_from_gt_image = torch.broadcast_to(
                                    latents_from_gt_image, [config.batch_size, *latents_from_gt_image.shape[1:]])
                            latents_from_gt_image = latents_from_gt_image * 0.18215

                            # Sample noise that we'll add to the latents
                            noise = torch.randn_like(latents_from_gt_image)
                            # Sample a random timestep for each image
                            timesteps_gt_latents = torch.randint(0, pipeline.scheduler.config.num_train_timesteps,
                                                                 (batch_size,), device=t_input.device)
                            timesteps_gt_latents = timesteps_gt_latents.long()
                            # Add noise to the latents according to the noise magnitude at each timestep (this is the forward diffusion process)
                            noisy_latents = pipeline.scheduler.add_noise(latents_from_gt_image, noise,
                                                                              timesteps_gt_latents)

                            # Get the target for loss depending on the prediction type
                            if pipeline.scheduler.config.prediction_type == "epsilon":
                                target = noise
                            elif pipeline.scheduler.config.prediction_type == "v_prediction":
                                target = pipeline.scheduler.get_velocity(latents_from_gt_image, noise, timesteps_gt_latents)
                            else:
                                raise ValueError(
                                    f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                            latent_model_input, t_input, prompt_embeds_input = \
                                torch.cat([latent_model_input, noisy_latents]), torch.cat(
                                    [t_input, timesteps_gt_latents]), torch.cat(
                                    [prompt_embeds_input, encoder_hidden_states])

                            # Predict the noise residual and compute loss
                            noise_pred_unet = unet(latent_model_input, t_input, prompt_embeds_input).sample
                            noise_pred_uncond, noise_pred_text, noise_pred_text_gt_img = noise_pred_unet.chunk(3)
                            noise_pred_ddpo = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        else:
                            noise_pred_unet = unet(latent_model_input, t_input, prompt_embeds_input).sample
                            noise_pred_uncond, noise_pred_text = noise_pred_unet.chunk(2)
                            noise_pred_ddpo = noise_pred_uncond + config.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        noise_pred_ddpo = unet(latents_i, t_i, encoder_hidden_states).sample

                    # DDPO losses
                    _, _, log_prob = pipeline.scheduler.step(noise_pred_ddpo, t_i[0], latents_i, eta=config.eta,
                                                                  prev_sample=next_latents_i, return_dict=False)
                    ratio_ddpo = torch.exp(log_prob - log_probs_lst[:, i])
                    ddpo_ratio_vals_lst.append(torch.mean(ratio_ddpo.detach()))
                    if not reward_scores_none:
                        unclipped_loss_ddpo = -advantages * ratio_ddpo
                        clipped_loss_ddpo = -advantages * torch.clamp(ratio_ddpo, 1.0 - config.clip_range, 1.0 + config.clip_range)
                        loss_ddpo = torch.sum(torch.max(unclipped_loss_ddpo, clipped_loss_ddpo))
                        ddpo_loss_vals_lst.append(loss_ddpo.detach())

                    if config.combine_DDPO_MSE or reward_scores_none:
                        loss_mse_reconstruction = F.mse_loss(noise_pred_text_gt_img.float(), target.float(),
                                                             reduction="mean")
                        mse_loss_vals_lst.append(loss_mse_reconstruction.detach())
                        total_loss = loss_mse_reconstruction
                        if not reward_scores_none:
                            total_loss += config.ddpo_loss_coef * loss_ddpo
                    else:
                        total_loss = config.ddpo_loss_scale * loss_ddpo

                    # Backpropagate
                    accelerator.backward(total_loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                mean_ddpo_ratio = sum(ddpo_ratio_vals_lst) / len(ddpo_ratio_vals_lst)
                avg_gathered_ddpo_ratio = accelerator.gather(mean_ddpo_ratio.repeat(config.batch_size)).mean()
                avg_gathered_ddpo_ratio_step = avg_gathered_ddpo_ratio.item() / config.gradient_accumulation_steps
                iteration_time = time.time() - start_time

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            if config.use_ema:
                ema_unet.step(unet.parameters())
            if (current_step % config.save_interval == 0 or current_step == config.max_iterations) and current_step != 0:
                accelerator.wait_for_everyone()
                #  Create the pipeline using the trained modules and save it.
                if accelerator.is_main_process:
                    # save_training_state(accelerator, current_step, config)
                    unet_save = accelerator.unwrap_model(unet)
                    if config.use_ema:
                        ema_unet.copy_to(unet_save.parameters())
                    pipeline_save = StableDiffusionPipeline.from_pretrained(
                        config.pretrained_model_name,
                        text_encoder=text_encoder,
                        vae=vae,
                        unet=unet_save,
                        scheduler=noise_scheduler,
                    )
                    save_path = os.path.join(config.snapshot_dir, f"iteration_{current_step}")
                    pipeline_save.save_pretrained(save_path)
                    accelerator.print(f"Pipeline saved to {save_path}")

            current_step += 1
            if config.image_reward_tuning or config.spatial_unidet_tuning or config.skin_tone_diversity_sp_tuning:
                log_str = (
                    f"step {current_step}\t"
                    f"data/training(seconds): {round(data_load_time, 3)}/{round(iteration_time, 3)}\t"
                    f"mean_reward: {round(avg_reward_step, 6)}\t"
                    f"mean_ddpo_ratio: {round(avg_gathered_ddpo_ratio_step, 6)}\t"
                )
                accelerator.print(f"{log_str}")


if __name__ == "__main__":
    train()
