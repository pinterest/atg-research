import enum


class EntityType(enum.Enum):
    SIGNATURE = 1
    SEARCH_QUERY = 2
    ITEM = 3
    CROSS = 4
    LABEL = 5

    _VALUES_TO_NAMES = {
        1: "SIGNATURE",
        2: "SEARCH_QUERY",
        3: "ITEM",
        4: "CROSS",
        5: "LABEL",
    }

    _NAMES_TO_VALUES = {
        "SIGNATURE": 1,
        "SEARCH_QUERY": 2,
        "ITEM": 3,
        "CROSS": 4,
        "LABEL": 5,
    }


class UtilityKey(enum.Enum):
    ENGAGEMENT = 1
    REPIN = 2
    LONG_CLICK = 3
    SAVE_TO_DEVICE = 4

    _VALUES_TO_NAMES = {
        1: "ENGAGEMENT",
        2: "REPIN",
        3: "LONG_CLICK",
        4: "SAVE_TO_DEVICE",
    }
    _NAMES_TO_VALUES = {
        "ENGAGEMENT": 1,
        "REPIN": 2,
        "LONG_CLICK": 3,
        "SAVE_TO_DEVICE": 4,
    }


class SearchHeadNames(enum.Enum):
    ENGAGEMENT = "engagement"
    REPIN = "repin"
    LONG_CLICK = "longclick"



def get_label_name_for_training(feature_definition: str) -> str:
    """
    Args:
        feature_definition: UFR feature definition
    Returns: name of this label feature as used in training data after parsing the TabularML dataset
    """
    return f'{EntityType.LABEL.value}/{feature_definition}'


class SearchLabels(enum.Enum):
    ENGAGEMENT = get_label_name_for_training("engagement")
    REPIN = get_label_name_for_training("is_repin")
    LONG_CLICK = get_label_name_for_training("is_longlick")
    LONG_CLICK_5S = get_label_name_for_training("is_longlick_5s")
    SAVE_TO_DEVICE = get_label_name_for_training("save_to_device")
    CLICK = get_label_name_for_training("click")
    CLOSEUP = get_label_name_for_training("closeup")
    IMPRESSION = get_label_name_for_training("impression")
    SHORT_CLICK_5S = get_label_name_for_training("short_click_5sec")
    SHARE = get_label_name_for_training("share")
    IS_TRUSTWORTHY = get_label_name_for_training("is_trustworthy")
    LOGGED_RELEVANCE_SCORE = get_label_name_for_training("logged_relevance_score")
    REQUEST_ID = get_label_name_for_training("request_id")
    USER_ID = get_label_name_for_training("user_id")
    ITEM_ID = get_label_name_for_training("item_id")


class FeatureType(enum.Enum):
    SPARSE_NUMERIC = 1,
    CATEGORICAL = 2,
    MULTI_CATEGORICAL = 3,
    DENSE_NUMERIC = 4,
    NUMERIC = 5