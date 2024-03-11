from omnisearchsage.common.types import EntityType

TENSOR_FEATURES = {
    EntityType.SIGNATURE: ["gs_v5", "item_is_v2", "ue_v4"],
    EntityType.ITEM: [
        "gs_v5_feat",
        "gs_v5_mask",
        "item_is_v2",
        "ue_v4_feat",
        "ue_v4_mask",
    ],
    EntityType.SEARCH_QUERY: [],
}

CONTENT_ONLY_TENSOR_FEATURES = {
    EntityType.SIGNATURE: ["ue_v4"],
    EntityType.ITEM: ["ue_v4_feat", "ue_v4_mask"],
}


STRING_FEATURES = {
    EntityType.SIGNATURE: [
        "item_title",
        "item_description",
        "item_domains",
        "item_brand",
        "item_colors",
        "item_size_type",
        "item_links",
        "item_size",
        "item_patterns",
        "item_materials",
        "item_product_types",
        "item_gpt",
        "canonical_desc",
        "board_title",
        "navboost_v2",
        "domain",
        "image_caption",
        "rich_pin_title",
        "rich_pin_description",
    ],
    EntityType.ITEM: [
        "title",
        "description",
        "domains",
        "brand",
        "colors",
        "size_type",
        "links",
        "size",
        "patterns",
        "materials",
        "product_types",
        "gpt",
        "canonical_desc",
        "board_title",
        "navboost_v2",
        "domain",
        "image_caption_feat",
        "item_rich_pin_title",
        "item_rich_pin_description",
    ],
    EntityType.SEARCH_QUERY: ["query_text"],
}

TENSOR_FEATURE_TO_EMB_SIZE = {
    "gs_v5": 256,
    "item_is_v1": 256,
    "item_is_v2": 256,
    "item_is_v2_alpha": 256,
    "ms_v2_alpha": 256,
    "ue_v4": 1024,
    "linksage": 256,
}

ITEM_FEATURES_IMAGE_LEVEL = [
    "gs_v5",
    "ue_v4",
]
VISUAL_FEATURE_NAME = "ue_v4"

PROD_TENSOR_FEATURES = {EntityType.SIGNATURE: ["gs_v4"]}
PROD_STRING_FEATURES = {EntityType.SIGNATURE: []}
