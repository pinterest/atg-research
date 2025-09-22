
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from interactrank.configs.base_configs import ConfigBundle, ResourceConfig, TrainerConfig
from interactrank.configs.base_configs import ENGAGEMENT_TASK, TASKS, DEFAULT_TABULAR_LONG_CLICK_WEIGHT_LWS


MEM_GB_PER_GPU = 130
CPUS_PER_GPU = 11
NUM_GPUS_PER_NODE = 8
def get_utc_date_str_n_days_ago(n=2) -> str:
    """
    This function gets the UTC time in datestr format n days ago
    with default n=2, to make sure data is produced by workflow.
    """
    return (datetime.utcnow().date() - timedelta(days=n)).strftime("%Y-%m-%d")

@dataclass
class SearchMultiPassAppLwConfig:
    """
    Search MultiPass Lw DNN model custom configs.

    :task: The task to train on, either 'engagement' or 'relevance'
    :batch_size: The pre-batch size of the dataset (it is the per-gpu batch size)
    :lr: Learning rate to use for training
    :iterations: Number of iterations to train on (this is per-gpu)
    :eval_only: Only run evaluation
    :max_num_eval_files: Number of files to use for evaluation
    :max_num_final_eval_files: Only used for training. Number of files to use for evalution in the last eval run
    :eval_every_n_iter: Runs eval_func every n iterations (used in BasicSolver)
    :should_eval: Runs eval
    :enable_cross_features: Enable cross features when training model
    :enable_train_static_rank: Enable training static rank
    """

    task: str = ENGAGEMENT_TASK
    neg_ratio: int = 4
    engagement_head_config: str = "SEARCH_PROD_HEAD_CONFIG"
    batch_size: int = 1000
    lr: float = 6e-4
    iterations: int = 100
    eval_only: bool = False
    max_num_eval_files: int = -1
    max_num_final_eval_files: int = -1
   # eval_additional_labels_str: str = "pin_id, user_id,request_id",
    # "request_id,img_sig,search_query,user_id,pin_id"
    # Metamodel weights: 'engagement', 'repin', 'longclick', 'longclick_5s', 'save_to_device'
    metamodel_weights: str = "1,0,0"

    # Feature Management (value is comma-separated list of tags)
    add_feature_tags: str = ''
    remove_feature_tags: str = ''

    # For tabular ML
    max_num_eval_iterations: int = -1
    max_num_final_eval_iterations: int = -1
    eval_every_n_iter: int = 500
    summarize_every_n_iter: int = 1000
    should_eval: bool = True
    learned_retrieval: str = "false"
    enable_cross_features: bool = True
    enable_train_static_rank: bool = False
    use_in_batch_negatives: bool = True
    in_batch_negative_weight: float = 0.01
    logged_relevance_loss_weight: float = 0.0
    in_batch_neg_mask_on_queries: bool = False
    correct_sample_probability: bool = True
    # For TabularML, update long click weight to override the default
    long_click_weight: float = DEFAULT_TABULAR_LONG_CLICK_WEIGHT_LWS
    # filter bad actor user ids in tabular dataloader
    filter_bad_actors: bool = True
    # enable user sequence in LW DNN model
    enable_user_sequence: bool = True
    # enable the recall metric compute for LW DNN Model.
    enable_recall_computation: bool = True
    # enable replacing average of navboost features
    enable_compute_average_navboost: bool = False
    # enable ingesting item side features
    enable_item_side_features: bool = True
    # calculate eval only with searchsage version, set this to searchsage version for eval, and set
    # pin and query feature names in map SEARCHSAGE_VERSION_TO_FEATURE in two_tower_generic/constants.py
    searchsage_eval_only_with_version = None
    # enable unified embedding v4
    enable_visual_uve4_pinnersage_v3e: bool = True
    upweight_lower_negatives: bool = False
    use_focal_loss: bool = True
    use_skip_connections: bool = False
    upweight_tprc_factor: float = 1.0
    # enable Parallel Mask Net layers
    enable_pmn: bool = True
    # eval on unsampled data
    use_unsampled_eval_data: bool = True
    # enable hits@k metric computation
    enable_estimator_rewards: bool = False
    # enable relevance metrics
    enable_relevance_metrics: bool = False
    # skip dense normalization layers
    skip_dense_norm: bool = False
    # skip mlp summarization layers
    skip_mlp_summarize_layer: bool = False

    # label sampling
    enable_label_sampling: bool = True
    impression_sampling_rate: float = 0.5
    closeup_sampling_rate: float = 1.0

    def __post_init__(self) -> None:
        assert self.__class__ == SearchMultiPassAppLwConfig, "Inheriting SearchMultiPassAppLwConfig is not allowed."
        assert self.task in TASKS
        # eval_additional_labels = self.eval_additional_labels_str.split(",")
        # assert "request_id" in eval_additional_labels, "eval_additional_labels_str must contain request_id."
        # # modify flags related to Learned Retrieval if learned_retrieval is set to organic or shopping


GPU_RESOURCE_CONFIG = ResourceConfig(
    num_nodes=1,
    gpus_per_node=8,
    cpus_per_node=11,
    mem_gb_per_node=MEM_GB_PER_GPU,
)

@dataclass
class SearchLwEngagementTabularTrainerConfigBundle(ConfigBundle):
    """
    ConfigBundle for SearchLwTrainer.
    python3.8 interactrank/common/launcher/launcher.py \
    --mode=local \
    --config_bundle=interactrank.configs.bundle.SearchLwEngagementTabularTrainerConfigBundle
    """

    trainer_config: TrainerConfig = TrainerConfig(
        trainer_class="interactrank.train.SearchLwsTrainer",
        nccl_timeout_mins=24 * 60,
        namespace="lw_dnn_training",
    )
    app_config: SearchMultiPassAppLwConfig = SearchMultiPassAppLwConfig(
        batch_size=500,
        iterations=40,
        max_num_eval_files=-1,
        max_num_final_eval_files=-1,
        eval_every_n_iter=10,
        # Run in-trainer eval and final eval for the following number of iterations per GPU
        max_num_eval_iterations=1,
        max_num_final_eval_iterations=2,
    )
    resource_config: ResourceConfig = GPU_RESOURCE_CONFIG
