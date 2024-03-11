<div align="center">

<!-- TITLE -->
# **OmniSearchSage: Multi-Task Multi-Entity Embeddings for Pinterest Search**

[//]: # (<img src="https://github.com/pinterest/atg-research/blob/main/docs/rl-diffusion/static/rl_vis_progression/tasks_progression_teaser.png" width=60% height=60%>)

[//]: # ([![arXiv]&#40;https://img.shields.io/badge/cs.LG-arXiv:2401.12244-b31b1b.svg&#41;]&#40;https://arxiv.org/abs/2401.12244&#41;)

[//]: # ([![Website]&#40;https://img.shields.io/badge/🌎-Website-blue.svg&#41;]&#40;https://pinterest.github.io/atg-research/rl-diffusion/&#41;)
</div>

This is the official implementation of our paper OmniSearchSage: Multi-Task Multi-Entity Embeddings for Pinterest Search by Prabhat Agarwal, Minhazul Islam Sk, Nikil Pancha, 
Kurchi Subhra Hazra, Jiajing Xu and Charles Rosenberg.


<!-- DESCRIPTION -->
## Abstract
In this paper, we present OmniSearchSage, a versatile and scalable system for understanding search queries, pins, and products for Pinterest search.
We jointly learn a unified query embedding coupled with pin and product embeddings, leading to > 8% relevance, > 7% engagement, 
and > 5% ads CTR gain in Pinterest’s production search system. The main contributors to these gains are improved content understanding, 
better multi-task learning, and real-time serving. We enrich our entity representations using diverse text derived from image captions 
from a generative LLM, historical engagement, and user-curated boards. Our multitask learning setup
produces a single search query embedding in the same space as pin and product embeddings and compatible with pre-existing pin and
product embeddings. We show the value of each feature through ablation studies, and show the effectiveness of a unified model
compared to standalone counterparts. Finally, we share how these embeddings have been deployed across the Pinterest search stack,
from retrieval to ranking, scaling to serve 300𝑘 requests per second at low latency.

[//]: # ()
[//]: # (## Citation)

[//]: # ()
[//]: # (If you find this work useful in your research, please cite:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@misc{zhang2024largescale,)

[//]: # (      title={Large-scale Reinforcement Learning for Diffusion Models},)

[//]: # (      author={Yinan Zhang and Eric Tzeng and Yilun Du and Dmitry Kislyuk},)

[//]: # (      year={2024},)

[//]: # (      eprint={2401.12244},)

[//]: # (      archivePrefix={arXiv},)

[//]: # (      primaryClass={cs.CV})

[//]: # (})

[//]: # (```)



