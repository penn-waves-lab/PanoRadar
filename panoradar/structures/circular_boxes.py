import torch
from typing import List, Tuple

from detectron2.structures.boxes import Boxes

class CircularBoxes(Boxes):
    """
    Structure that stores a list of boxes as a Nx4 torch.Tensor, where N is the number of boxes
    This class supports circular boxes, i.e. boxes that cross over the width of the image from one end to the other


    Attributes:
        tensor (torch.Tensor) : float matrix of Nx4. Each row is (x1, y1, x2, y2). Here it is assumed that x1y1 is the bottom left, x2y2 is top right, note it is possible x2 < x1
        width (int) : Width of the image of the boxes, default is 512
    """

    def __init__(self, tensor: torch.Tensor, width=512):

        """
        Args:
            tensor : a Nx4 tensor where each row is (x1, y1, x2, y2)
            width : integer denoting the width of the image that boxes are for
        """

        super().__init__(tensor=tensor)
        self.tensor[:, 2] = self.tensor[:, 2] % width
        self.width = width
    
    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes

        Returns:
            torch.Tensor: a vector of length N with areas of each box
        """

        box = self.tensor

        # Add width of image to the x2 coordinate if x2 < x1 to get width of box
        boxes_widths = torch.where(box[:, 2] < box[:, 0], self.width + box[:, 2] - box[:, 0], box[:, 2] - box[:,0])
        boxes_heights = box[:, 3] - box[:, 1]
        area = boxes_widths * boxes_heights
        return area

    def clip(self, box_size: Tuple[int, int]) -> None:
        """
        Clip (in place) the boxes by modding x coordinates by img_width and limiting y coordinates to the range [0, height]

        Args:
            box_size: The clipping box's size
        """
        h, w = box_size
        x1 = self.tensor[:, 0] % self.width
        y1 = self.tensor[:, 1].clamp(min=0, max=h)
        x2 = self.tensor[:, 2] % self.width
        y2 = self.tensor[:, 3].clamp(min=0, max=h)
        self.tensor = torch.stack((x1, y1, x2, y2), dim=1)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty if either of its sides is not larger than the threshold

        Returns:
            torch.Tensor: a binary vector of length N which represents whether or not each box is empty (False), or non-empty (True)
        """
        box = self.tensor

        # Calculate widths of boxes, add width of image to x2 if x2 < x1 to get width of box
        widths = torch.where(box[:, 2] < box[:, 0], self.width + box[:, 2] - box[:, 0], box[:, 2] - box[:, 0])
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item) -> "CircularBoxes":
        """
        Args:
            item: int, slice, or a BoolTensor

        Returns:
            Boxes: Create a new :class:`CircularBoxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `CircularBoxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of `CircularBoxes`.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned CircularBoxes might share storage with this CircularBoxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return CircularBoxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return CircularBoxes(b)

    @classmethod
    def cat(cls, boxes_list: List["CircularBoxes"]) -> "CircularBoxes":
        """
        Concatenates a list of CircularBoxes into a single CircularBoxes object

        Arguments:
            boxes_list (list[CircularBoxes])

        Returns:
            CircularBoxes: the concatenated CircularBoxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, CircularBoxes) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0), width=boxes_list[0].width)
        return cat_boxes



# ----- IOU methods -----

def separate_boxes(boxes1: CircularBoxes, boxes2: CircularBoxes) -> Tuple:
    """
    Given two lists of boxes of length N and length M, separate each box into two boxes.
    Useful for when boxes cross the origin and loop to the beginning. If x2 < x1, separates the box into two boxes that span (x1, WIDTH-1) and (0, x2)

    Args:
        boxes1, boxes2 (CircularBoxes): two CircularBoxes, contains N and M boxes, respectively
    
    Returns:
        Tuple of Tensors: (Nx4, Mx4) tensors with each row being (x1start, x1end, x2start, x2end). If the original box does not cross 0 then x2start and x2end are both 0.
    """
    img_width = boxes1.width
    boxes1, boxes2 = boxes1.tensor, boxes2.tensor

    # Full tensors for box1 and box2 shapes
    box1_0s = torch.full((boxes1.shape[0], ), 0).to(device=boxes1.device)
    box2_0s = torch.full((boxes2.shape[0], ), 0).to(device=boxes1.device)
    box1_maxs = torch.full((boxes1.shape[0], ), img_width-0.01).to(device=boxes1.device)
    box2_maxs = torch.full((boxes2.shape[0], ), img_width-0.01).to(device=boxes1.device)

    # Separate each box into 2 boxes, one at beginning and one at end, or maintain original box
    box1_xs = torch.where((boxes1[:, 2] < boxes1[:, 0]).expand(4, boxes1.shape[0]).transpose(0,1), 
                          torch.stack((boxes1[:, 0], box1_maxs, box1_0s, boxes1[:,2])).transpose(0,1), 
                          torch.stack((boxes1[:, 0], boxes1[:, 2], box1_0s, box1_0s)).transpose(0,1)
                          )
    
    box2_xs = torch.where((boxes2[:, 2] < boxes2[:, 0]).expand(4, boxes2.shape[0]).transpose(0,1), 
                          torch.stack((boxes2[:, 0], box2_maxs, box2_0s, boxes2[:,2])).transpose(0,1), 
                          torch.stack((boxes2[:, 0], boxes2[:, 2], box2_0s, box2_0s)).transpose(0,1)
                          )
    
    return box1_xs, box2_xs
    

def circular_pairwise_intersection(boxes1: CircularBoxes, boxes2: CircularBoxes) -> torch.Tensor:
    """
    Given two lists of boxes of length N and length M, compute the intersection area between all N x M pairs of boxes
    Note that boxes MUST be (x1, y1, x2, y2), where x1, y1 is the bottom left corner of the box, and x2, y2 is the top right corner of the box. x2 may be < x1

    Args:
        boxes1, boxes2 (CircularBoxes) : two CircularBoxes, contains N and M boxes, respectively
    
    Returns:
        Tensor: a N x M matrix with the intersection value of the nth box in first list of boxes and the mth box in second list of boxes in the [n, m] entry
    """
    # Make sure both boxes objects are on images with the same width
    assert boxes1.width == boxes2.width 

    boxes1_t, boxes2_t = boxes1.tensor, boxes2.tensor

    # Calculate the intersection heights for each pair of boxes
    intersect_heights = (torch.min(boxes1_t[:, None, 3], boxes2_t[:, 3]) - torch.max(boxes1_t[:, None, 1], boxes2_t[:, 1])).clip(min=0) # [N,M]

    # Separate each box into 2 boxes
    boxes1_xs, boxes2_xs = separate_boxes(boxes1, boxes2)

    # Calculate the intersection widths for each pair of boxes.
    # Since for each box it is split into 2, for each pair of boxes we calculate the sum of intersections between the 4 boxes
    width1 = (torch.min(boxes1_xs[:, None, 1], boxes2_xs[:, 1]) - torch.max(boxes1_xs[:, None, 0], boxes2_xs[:, 0])).clip(min=0)
    width2 = (torch.min(boxes1_xs[:, None, 1], boxes2_xs[:, 3]) - torch.max(boxes1_xs[:, None, 0], boxes2_xs[:, 2])).clip(min=0)
    width3 = (torch.min(boxes1_xs[:, None, 3], boxes2_xs[:, 1]) - torch.max(boxes1_xs[:, None, 2], boxes2_xs[:, 0])).clip(min=0)
    width4 = (torch.min(boxes1_xs[:, None, 3], boxes2_xs[:, 3]) - torch.max(boxes1_xs[:, None, 2], boxes2_xs[:, 2])).clip(min=0)
    intersect_widths = width1 + width2 + width3 + width4

    # Intersection area is the product of the intersection width and the intersection height
    intersection = intersect_heights * intersect_widths

    return intersection

def circular_pairwise_iou(boxes1: CircularBoxes, boxes2: CircularBoxes) -> torch.Tensor:
    """
    Given two lists of boxes of length N and length M, compute the IOU between all N x M pairs of boxes
    Note that boxes MUST be (x1, y1, x2, y2), where x1, y1 is the bottom left corner of the box, and x2, y2 is the top right corner of the box. x2 may be < x1

    Args:
        boxes1, boxes2 (CircularBoxes) : two CircularBoxes, contains N and M boxes, respectively
    
    Returns:
        Tensor: a N x M matrix with the IOU value of the nth box in first list of boxes and the mth box in second list of boxes in the [n, m] entry
    """

    area1 = boxes1.area()
    area2 = boxes2.area()
    intersection_areas = circular_pairwise_intersection(boxes1, boxes2)
    union_areas = area1[:, None] + area2 - intersection_areas

    # handle empty boxes
    iou = torch.where(intersection_areas > 0, intersection_areas/union_areas, torch.zeros(1, dtype=intersection_areas.dtype, device=intersection_areas.device))
    return iou

