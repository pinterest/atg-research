cmd="pip install -r /interactrank/requirements.txt && \
cd /interactrank/ && \
python3 interactrank/common/launcher/launcher.py \
        --mode=local \
        --resource_config.gpus_per_node=1 \
        --app_config.num_workers=3 \
        --config_bundle=interactrank.configs.bundle.SearchLwEngagementTabularTrainerConfigBundle "

docker run --gpus all -it --rm \
    -v "$PWD":/interactrank \
    -e PYTHONPATH=":/interactrank" \
    --ipc=host \
    nvcr.io/nvidia/pytorch:23.05-py3 \
    bash -c "export PYTHONPATH=/interactrank && $cmd"
