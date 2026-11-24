"""Use napari-ome-zarr's reader to open and validate the image and label layers."""
import json
import sys

import numpy as np
from napari_ome_zarr import napari_get_reader


def main(path, bounds, expected):
    reader = napari_get_reader(path)
    if reader is None:
        raise ValueError(f"napari-ome-zarr did not recognize {path}")
    layers = reader(path)
    images = [(data, metadata) for data, metadata, layer in layers if layer == "image"]
    labels = [(data, metadata) for data, metadata, layer in layers if layer == "labels"]
    if len(images) != 1 or len(labels) != 1:
        raise ValueError(f"expected one image and one label layer, got {len(images)} and {len(labels)}")
    image, image_metadata = images[0]
    label, label_metadata = labels[0]
    if [a.shape for a in image] != [a.shape for a in label]:
        raise ValueError("viewer returned mismatched image and label pyramids")
    for key in ("scale", "translate"):
        np.testing.assert_equal(image_metadata.get(key), label_metadata.get(key))
    selection = tuple(slice(a, b) for a, b in json.loads(bounds))
    # Force lazy layers to decode actual stored chunks.
    if not np.asarray(image[0][selection]).any():
        raise ValueError("viewer returned an empty image ROI")
    np.testing.assert_equal(int(np.asarray(label[0][selection]).sum()), int(expected))
    print(f"Opened {len(image)} image and label levels and decoded the full-resolution ROI")


if __name__ == "__main__":
    main(*sys.argv[1:])
