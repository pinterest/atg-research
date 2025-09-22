import enum

from interactrank.common.types import EntityType
from interactrank.common.types import FeatureType

TENSOR_FEATURES = {
    EntityType.SIGNATURE: [
        "gs_v5",
        "ue_v4",
        "searchsage_item",
        "sig_count",
        "item_category"
    ],
    EntityType.SEARCH_QUERY: ["user_seq_action_type",
                              "user_seq_gs_v5",
                              "user_seq_timestamp",
                              "timestamp",
                              "searchsage_query",
                              "itemsage_seq",
                              "category_count"
                              ],
    EntityType.CROSS: [
        "item_click_90day",
        "item_save_90day",
    ],
}

TENSOR_FEATURE_TO_EMB_SIZE = {
    "gs_v5": 256,
    "item_is_v1": 256,
    "item_is_v2": 256,
    "item_is_v2_alpha": 256,
    "ms_v2_alpha": 256,
    "ue_v4": 1024,
}

ITEM_FEATURES_IMAGE_LEVEL = [
    "gs_v5",
    "ue_v4",
]

PROD_TENSOR_FEATURES = {EntityType.SIGNATURE: ["gs_v4"]}
IMAGE_SIGNATURE_FEATURE = "image_sig"
SEARCH_QUERY_FEATURE = "search_query"
USER_ID_FEATURE = "user_id"
REQUEST_ID_FEATURE = "request_id"
ITEM_ID_FEATURE = "item_id"
IS_REPIN_FEATURE = "is_repin"
IS_LONGCLICK_FEATURE = "is_longclick"

FEATURE_NAME_TO_TYPE_MAPPING = {
    "item_save_90day": FeatureType.NUMERIC,
    "item_category_hash": FeatureType.NUMERIC,
    "item_click_90day": FeatureType.NUMERIC,
    "gs_v5": FeatureType.DENSE_NUMERIC,
    "ue_v4": FeatureType.DENSE_NUMERIC,
    "user_seq_action_type": FeatureType.DENSE_NUMERIC,
    "user_seq_gs_v5": FeatureType.DENSE_NUMERIC,
    "user_seq_timestamp": FeatureType.DENSE_NUMERIC,
    "timestamp": FeatureType.DENSE_NUMERIC,
    "searchsage_query": FeatureType.DENSE_NUMERIC,
    "searchsage_item": FeatureType.DENSE_NUMERIC,
    "itemsage_seq": FeatureType.DENSE_NUMERIC,
    "sig_count": FeatureType.NUMERIC,
    "category_count": FeatureType.NUMERIC,
    "item_category": FeatureType.CATEGORICAL,
}


REPLAY_INPUT = 'replay_input'
ENCODED_REPLAY_INPUT = REPLAY_INPUT.encode('utf-8')

USER_ID_KEY = "user_id"
REQUEST_ID_KEY = "request_id"
TOP_LEVEL_TRAFFIC_SOURCE_KEY = "top_level_traffic_source"
TOP_LEVEL_TRAFFIC_SOURCE_DEPTH_KEY = "top_level_traffic_source_depth"
ITEM_ID_KEY = "item_id"
USER_STATE_KEY = "user_state"
PIN_TYPE_KEY = "pin_content_type"
SLOT_IDX_KEY = "slot_index"
QUERY_PIN_ID_KEY = "query_pin_id"
QUERY_SIGNATURE_KEY = "query_signature"
SAMPLING_PROB_KEY = "sampling_probability"
RANDOMIZED_POSITION_KEY = "randomized_position"
SCORES_KEY = "scores"
LOGGED_SCORES_KEY = "logged_scores"
REWARD_VECTOR_KEY = "rewards"
USER_COUNTRY_KEY = "user_country"
DOWNSTREAM_CLICKTHROUGH_KEY = "ds_clickthrough"
DOWNSTREAM_CLOSEUP_KEY = "ds_closeup"
DOWNSTREAM_LONGCLICK_KEY = "ds_long_click"
DOWNSTREAM_REPIN_KEY = "ds_repin"
PIN_AGE_DAYS_KEY = "pin_age_days"
IS_FRESH_PIN_KEY = "is_fresh"
CAND_PIN_AGE_DAYS_KEY = "cand_pin_age_days"
IS_CAND_FRESH_KEY = "is_cand_fresh"
QUERY_SEGMENT_KEY = "query_segment"
NAVBOOST_XPERF2_EXP_GRIDCLICK_KEY = "navboost_gridclicks"
NAVBOOST_COVERAGE_KEY = "navboost_coverage"
CAND_PIN_AGE_DAYS_KEY = "cand_pin_age_days"
IS_CAND_FRESH_PIN_KEY = "is_cand_fresh"
