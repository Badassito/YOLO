"""Shared standard-library imports used by extracted implementation modules."""

from __future__ import annotations

import argparse
import atexit
import colorsys
import contextlib
import functools
import gc
import inspect
import importlib.metadata as importlib_metadata
import io
import json
import math
import mmap
import multiprocessing as mp
from multiprocessing import reduction as mp_reduction
import os
import queue
import re
import signal
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import weakref
import zlib
from collections import Counter, OrderedDict, deque
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import dataclass, field, replace as dataclasses_replace
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


__all__ = tuple(name for name in globals() if not name.startswith("__"))
