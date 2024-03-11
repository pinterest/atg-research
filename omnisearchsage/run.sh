cmd="pip install -r /omnisearchsage/requirements.txt && \
cd /omnisearchsage/extension/third_party/icu && \
git apply ../udata.patch && \
cd /omnisearchsage/extension && \
pip install . && \
rm -rf /omnisearchsage/extension/build && \
cd /omnisearchsage && pytest && \
cd /omnisearchsage && \
python omnisearchsage/launcher/launcher.py \
        --mode=local \
        --resource_config.gpus_per_node=2 \
        --app_config.num_workers=3 \
        --app_config.batch_size=8 \
        --config_bundle=omnisearchsage.configs.configs.OmniSearchSageTrainingConfigBundle \
        --app_config.eval_index_size=1k \
        --app_config.eval_every_n_iter=200 \
        --app_config.iterations=10"

docker run --gpus all -it --rm \
    -v "$PWD":/omnisearchsage \
    -e PYTHONPATH=":/omnisearchsage" \
    --ipc=host \
    nvcr.io/nvidia/pytorch:23.05-py3 \
    bash -c "$cmd"
