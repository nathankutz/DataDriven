import sys
import warnings
from pathlib import Path
from typing import Any
from typing import Sequence
from typing import Union

import h5py
import numpy as np
from pysindy import AxesArray


if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

reduction_methods = entry_points(group="ara.dim_reduction")

Pathlike = Union[str, Path]
STDataset = dict
DATA_DIR = Path("/home/ara/data")
st_loc = Path("01_matrix_HDF5")
NUMERICAL_PREFIX = "010"
DATAFILE_PREFIX = "simVectors" + NUMERICAL_PREFIX
vector_axes = {"ax_spatial": [0, 1], "ax_time": 2, "ax_coord": 3}


def _open(filename: Pathlike) -> STDataset:
    with h5py.File(filename) as h5file:
        return {
            "t1": np.array(h5file["t1"]),
            "t2": np.array(h5file["t2"]),
            "x": np.array(h5file["x"]),
            "y": np.array(h5file["y"]),
            "v1": AxesArray(
                np.stack((h5file["Vx1"], h5file["Vy1"], h5file["Vz1"]), axis=-1),
                axes=vector_axes,
            ),
            "v2": AxesArray(
                np.stack((h5file["Vx2"], h5file["Vy2"], h5file["Vz2"]), axis=-1),
                axes=vector_axes,
            ),
        }


def open(data_num: Union[int, str]) -> STDataset:
    """Open a dataset in the default data directory on doppio

    Args:
        data_num: the trial id, e.g. 104.  "simVectors010" is prepended
            to create the filename

    Returns:
        dictionary with keys to the spatiotemporal grid and vectors
    """
    filename = DATAFILE_PREFIX + str(data_num) + ".h5"
    return _open(DATA_DIR / st_loc / filename)


def _list_datasets():
    matches = Path(DATA_DIR / st_loc).glob(DATAFILE_PREFIX + "*.h5")
    return tuple(matches)


def list_datasets():
    """List available datasets in the default data directory on doppio

    Returns:
        tuple of datasets, by number, that can be passed to ara.open()
    """
    matches = _list_datasets()
    return tuple(match.name[13:-3] for match in matches)


def time_misalignment(t1: np.ndarray, t2: np.ndarray) -> float:
    """Calculate how misaligned the times of v1 and v2 are.

    Args:
        t1, t2: the times of v1 and v2 observations

    Returns:
        Maximum value of time delta between equivaluent t1 and t2
        indices, divided by the t1 or t2 gap on either side of the index
    """
    dts1 = t1[1:] - t1[:-1]
    dts2 = t2[1:] - t2[:-1]
    terr_rel = np.stack(
        (
            (t2 - t1)[1:] / dts1,
            (t2 - t1)[:-1] / dts1,
            (t2 - t1)[1:] / dts2,
            (t2 - t1)[:-1] / dts2,
        )
    )
    return np.max(np.abs(terr_rel))


def svd_time(arr: AxesArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the SVD of arr, flattening all axes but time"""
    flattened = np.reshape(arr, (-1, arr.n_time))
    return np.linalg.svd(flattened)


svd_time.names = ("u", "s", "v")


def save_dim_reduction(
    data_num: Union[int, str],
    reduction_type: str,
    suffix: str = "",
    names: Sequence[str] = None,
    **kwargs,
) -> Any:
    """Create and cache a dimensionality reduction, applied to the data.

    Args:
        data_num: number that identifies the original data (e.g. the last
            three digits of the spatiotemporal h5 files
        reduction_type: the name of a reduction method registered to the
            ara.dim_reduction entry point.
        suffix: a custom suffix to append to the reducer name in creating
            the file name
        names: the names for the output arrays from the reduction method,
             e.g. ["u", "s", "v"] for singular value decomposition.
        **kwargs: arguments passed to reduction method, including the data

    Returns:
        The output of the reduction method.
    """
    try:
        reducer = reduction_methods[reduction_type]
    except KeyError:
        raise ValueError(
            f"No reduction method named {reduction_type} is installed."
            "Reduction methods need to be installed as an entry point to"
            "the 'ara.dim_reduction' group"
        )
    out = reducer(**kwargs)
    filename = f"{reducer.__name__}{suffix}{NUMERICAL_PREFIX}{data_num}"
    try:
        save_kwargs = {name: arr for name, arr in zip(names, out, strict=True)}
        save_args = ()
    except ValueError:
        warnings.warn(
            "Names has different number of elements than output of reduction method."
            "Assigning generic names."
        )
        save_args = out
        save_kwargs = {}
    np.savez_compressed(DATA_DIR / st_loc / filename, *save_args, **save_kwargs)
    return out
