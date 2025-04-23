"""Label definition."""

from collections import namedtuple

# a label and all meta information
# Code inspired by Cityscapes https://github.com/mcordts/cityscapesScripts
Label = namedtuple(
    "Label",
    [
        "name",  # The identifier of this label
        # We use them to uniquely name a class
        "id",  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images An ID
        # of -1 means that this label does not have an ID and thus is ignored
        # when creating ground truth images (e.g. license plate). Do not modify
        # these IDs, since exactly these IDs are expected by the evaluation
        # server.
        "trainId",
        # Feel free to modify these IDs as suitable for your method. Then
        # create ground truth images with train IDs, using the tools provided
        # in the 'preparation' folder. However, make sure to validate or submit
        # results to our evaluation server using the regular IDs above! For
        # trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the
        # inverse mapping, we use the label that is defined first in the list
        # below. For example, mapping all void-type classes to the same ID in
        # training, might make sense for some approaches. Max value is 255!
        "category",  # The name of the category that this label belongs to
        "categoryId",
        # The ID of this category. Used to create ground truth images
        # on category level.
        "hasInstances",
        # Whether this label distinguishes between single instances or not
        "ignoreInEval",
        # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not
        "color",  # The color of this label
    ],
)

# Our extended list of label types. Our train id is compatible with IJmond_seg
labels = [
    #       name                     id    trainId   category catId
    #       hasInstances   ignoreInEval   color
    Label("smoke", 0, 0, "smoke", 0, False, False, (128, 128, 128)),
    Label("high-opacity-smoke", 1, 1, "smoke", 0, False, False, (255, 0, 0)),
    Label("low-opacity-smoke", 2, 2, "smoke", 0, False, False, (0, 255, 0)),
]
