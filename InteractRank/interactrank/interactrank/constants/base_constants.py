"""
Constants file for all search related constants
"""
from interactrank.common.types import EntityType

ENABLE_CONTINUOUS_LAYER = True
CANDIDATE_FEATURE_GROUP_NAMES = set([f"{EntityType.SIGNATURE}"])
CANDIDATE_SIG_ITEM_FEATURE_GROUP_NAMES = set([f"{EntityType.ITEM}", f"{EntityType.SIGNATURE}"])
CONTEXT_FEATURE_GROUP_NAMES = set([f"{EntityType.SEARCH_QUERY}"])
CROSS_FEATURE_GROUP_NAMES = set([f"{EntityType.CROSS}"])
DEFAULT_TIMESTAMP_FEAT_NAME = (
    f"{EntityType.SEARCH_QUERY.value}/timestamp"
)

# constants used to configure data set paths name in DatasetConfig
TRAIN_DATASET = "train_dataset"
TRAIN_FEATURE_STATS = "train_feature_stats"
EVAL_DATASET = "eval_dataset"
EVAL_DATASET_UNSAMPLED = "eval_dataset_unsampled"
CALIBRATION_TRAIN_DATASET = "calibration_train_dataset"
CALIBRATION_EVAL_DATASET = "calibration_eval_dataset"
REL_EVAL_DATASET = "relevance_eval_dataset"
CORPUS_NAME = 'corpus'

MULTI_HEAD_WEIGHTS_FEATURE = f"{EntityType.SEARCH_QUERY.value}/lws_multihead_weights.name.id"

CROSS_FEATURES = [
    '4/query_item_1_year_historical_click_country',
    '4/query_item_2_year_historical_click_gender',
    '4/query_item_token_boost',
    '4/query_item_1_year_fresh_historical_repin_country',
    '4/query_item_1_year_fresh_historical_repin_gender',
]


# key entities in the model
LABEL_TYPE = "label_type"
WEIGHT_FIELD = "weight"
LABEL_FIELD = "binary_label"
REPIN_LABEL_FIELD = "repin_binary_label"
LC_LABEL_FIELD = "lc_binary_label"
QUERY_ID_FIELD = "query_id"
REQUEST_ID_FIELD = "request_id"
USER_ID_FIELD = "user_id"
ENTITY_ID_FIELD = "entity_id"
IMG_SIG_FIELD = "img_sig"
PIN_ID_FIELD = "pin_id"
QUERY_STR_FIELD = "search_query"
# fill these from dataframe?
QUERY_EMBEDDING_FIELD = "query_embedding"
PIN_EMBEDDING_FIELD = "pin_embedding"

LABEL_COLUMNS = {LABEL_FIELD, WEIGHT_FIELD, LABEL_TYPE, "user_id"}

# PR-AUC and ROC-AUC related fields
ROC_AUC_FIELD = "ROC_AUC"
QUERY_SEGMENTED_ROC_AUC_FIELD = "QUERY_SEGMENTED_ROC_AUC"
SHOPPING_PR_AUC_FIELD = "SHOPPING_PR_AUC"
SHOPPING_ROC_AUC_FIELD = "SHOPPING_ROC_AUC"
FRESH_PR_AUC_FIELD = "FRESH_PR_AUC"
FRESH_ROC_AUC_FIELD = "FRESH_ROC_AUC"
PR_AUC_FIELD = "PR_AUC"
PREDICTIONS = "predictions"
ROC_KEY = "ROC"
PR_KEY = "PR"
REQ_LEVEL_ROC_AUC = "req_level_ROC_AUC"
HITS_AT_3 = "hits_at_3"

# hidden sizes for fully connected layer
DEFAULT_HIDDEN_SIZES_STR = "512, 256"
DEFAULT_HIDDEN_SIZES = [int(x) for x in DEFAULT_HIDDEN_SIZES_STR.split(",")]
CONTEXT_TOWER_NAN_FILL_FEATURES = []
CANDIDATE_TOWER_NAN_FILL_FEATURES = []

DEFAULT_USE_LATENT_CROSS = "false"
DEFAULT_NUM_LATENT_CROSS = 0

DEFAULT_EMBEDDING_DIM = 64
SHARED_EMBEDDING_VOC_MIN_COUNT = None

DEFAULT_CANDIDATE_TOWER_TORCH_SCRIPT_PREFIX = "snap_final_candidate_model"
DEFAULT_CONTEXT_TOWER_TORCH_SCRIPT_PREFIX = "snap_final_context_model"
DOT_PRODUCT_FEATURE_FIELD = "84/two_tower_dot_product"

DEFAULT_DEVICE_LIST = ("cpu",)

OUTPUT_NAMES = ["embedding"]

# 5 eval workers caused CPU OOM. Reducing to 4 for now.
DEFAULT_NUM_EVAL_WORKER = 4
DEFAULT_NUM_WORKER = 12

# DEFAULT_NUM_EVAL_WORKER * num of GPUs
DEFAULT_MAX_NUM_FILES_FOR_EVAL = 40

# default number of steps
DEFAULT_NUM_STEPS = 60000

# number of steps to run before saving snapshot
DEFAULT_SNAPSHOT_N_ITER = 25000

# num of batches for evaluation every DEFAULT_EVAL_N_ITER steps
DEFAULT_MAX_NUM_BATCHES_FOR_EVAL = 250  # 1000
DEFAULT_MAX_NUM_BATCHES_FOR_FINAL_RELEVANCE_EVAL = 10
DEFAULT_MAX_NUM_BATCHES_FOR_RELEVANCE_EVAL = 2
MULTIHEAD_LONG_CLICK_WEIGHT = 20.0
