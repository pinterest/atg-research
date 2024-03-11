from __future__ import annotations

import errno
import hashlib
import os


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def hash_str(str_input):
    return hashlib.sha256(str_input.encode("utf-8")).hexdigest()
