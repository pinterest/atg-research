class EntityType:
    SIGNATURE = 1
    SEARCH_QUERY = 2
    ITEM = 3

    _VALUES_TO_NAMES = {
        1: "SIGNATURE",
        2: "SEARCH_QUERY",
        3: "ITEM",
    }

    _NAMES_TO_VALUES = {
        "SIGNATURE": 1,
        "SEARCH_QUERY": 2,
        "ITEM": 3,
    }
