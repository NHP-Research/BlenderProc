"""Allows the sampling 3D Front scenes"""

import random
from typing import List

import numpy as np

from blenderproc.python.types.MeshObjectUtility import MeshObject

from scipy import ndimage
import matplotlib.pyplot as plt

class Front3DPointInRoomSampler:
    """
    Allows the sampling 3D Front scenes
    """

    def __init__(self, front3d_objects: List[MeshObject], amount_of_objects_needed_per_room: int = 2):
        """ Collects the floors of all rooms with at least N objects.

        :param front3d_objects: The list of front3d objects that should be considered.
        :param amount_of_objects_needed_per_room: The number of objects a rooms needs to have, such that it is
                                                  considered for sampling.
        """
        front3d_objects = [obj for obj in front3d_objects if obj.has_cp("is_3D_future")]

        floor_objs = [obj for obj in front3d_objects if obj.get_name().lower().startswith("floor")]

        # count objects per floor -> room
        floor_obj_counters = {obj.get_name(): 0 for obj in floor_objs}
        counter = 0
        for obj in front3d_objects:
            name = obj.get_name().lower()
            if "wall" in name or "ceiling" in name:
                continue
            counter += 1

            for floor_obj in floor_objs:
                is_above = floor_obj.position_is_above_object(obj.get_location())
                if is_above:
                    floor_obj_counters[floor_obj.get_name()] += 1
        self.used_floors = [obj for obj in floor_objs if
                            floor_obj_counters[obj.get_name()] > amount_of_objects_needed_per_room]

        self.above_objects = []
        for floor in self.used_floors:
            self.above_objects.append([obj for obj in front3d_objects if
                                    #    obj.has_cp("room_id") and
                                    # "wall" not in obj.get_name().lower() and
                                    # "ceiling" not in obj.get_name().lower() and
                                    floor.position_is_above_object(obj.get_location())])

    def sample(self, height: float, max_tries: int = 1000) -> np.ndarray:
        """ Samples a point inside one of the loaded Front3d rooms.

        The points are uniformly sampled along x/y over all rooms.
        The z-coordinate is set based on the given height value.

        :param height: The height above the floor to use for the z-component of the point.
        :param max_tries: The maximum number of times sampling above the floor should be tried.
        :return: The sampled point.
        """
        for _ in range(max_tries):
            # Sample room via floor objects
            floor_obj = random.choice(self.used_floors)

            # Get min/max along x/y-axis from bounding box of room
            bounding_box = floor_obj.get_bound_box()
            min_corner = np.min(bounding_box, axis=0)
            max_corner = np.max(bounding_box, axis=0)

            # Sample uniformly inside bounding box
            point = np.array([
                random.uniform(min_corner[0], max_corner[0]),
                random.uniform(min_corner[1], max_corner[1]),
                floor_obj.get_location()[2] + height
            ])

            # Check if sampled pose is above the floor to make sure its really inside the room
            if floor_obj.position_is_above_object(point):
                return point

        raise RuntimeError("Cannot sample any point inside the loaded front3d rooms.")

    def get_blank_position(self, h: float, room_index: int, resolution: tuple[int,int] = (1000, 1000)) -> np.ndarray:
        """ Returns the center of the given room.

        :param room_index: The index of the room to get the center for.
        :return: The center of the room.
        """
        room = self.used_floors[room_index]
        bounding_box = room.get_bound_box()

        min_corner = np.min(bounding_box, axis=0)
        max_corner = np.max(bounding_box, axis=0)

        width = max_corner[0] - min_corner[0]
        height = max_corner[1] - min_corner[1]

        scale = min(resolution[0] / width, resolution[1] / height)

        width = np.round(width * scale).astype(int)
        height = np.round(height * scale).astype(int)

        # fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        # axs[0].axis('equal')
        # axs[1].axis('equal')
        # axs[2].axis('equal')
        # rect = plt.Rectangle((min_corner[0], min_corner[1]), max_corner[0] - min_corner[0], max_corner[1] - min_corner[1], linewidth=1, edgecolor='r', facecolor='none')
        # axs[1].add_patch(rect)

        sub_rect = np.array([width, height, 0, 0])

        floor = np.zeros((height, width), dtype=np.uint8)
        floor[1:-2, 1:-2] = 1
        # print("Floor shape: ", floor.shape)
        for obj in self.above_objects[room_index]:
            obj_box = obj.get_bound_box()
            # print("Object box: ", obj_box)
            obj_min = np.min(obj_box, axis=0)
            obj_max = np.max(obj_box, axis=0)

            obj_z = obj_min[2]

            if obj_z > room.get_location()[2] + h:
                continue

            # axs[1].add_patch(plt.Rectangle((obj_min[0], obj_min[1]), obj_max[0] - obj_min[0], obj_max[1] - obj_min[1], linewidth=1, edgecolor='g', facecolor='g'))

            left = np.round((obj_min[0] - min_corner[0]) * scale).astype(int) 
            right = np.round((obj_max[0] - min_corner[0]) * scale).astype(int)
            bot = np.round((obj_min[1] - min_corner[1]) * scale).astype(int)
            top = np.round((obj_max[1] - min_corner[1]) * scale).astype(int)

            sub_rect[0] = min(sub_rect[0], left)
            sub_rect[1] = min(sub_rect[1], bot)
            sub_rect[2] = max(sub_rect[2], right)
            sub_rect[3] = max(sub_rect[3], top)

            # print(left, right, bot, top)

            floor[bot:top, left:right] = 0

        sub_rect[0] = max(sub_rect[0], 0)
        sub_rect[1] = max(sub_rect[1], 0)
        sub_rect[2] = min(sub_rect[2], width)
        sub_rect[3] = min(sub_rect[3], height)

        center = np.array([width, height]) / 2
        radius = np.linalg.norm(max_corner[:2] - center)

        circle_map = np.ones((height, width), dtype=np.uint8)
        circle_map[(center[1].astype(int), center[0].astype(int))] = 0
        circle_distance = radius - ndimage.distance_transform_edt(circle_map)
        distance = (ndimage.distance_transform_edt(floor) + circle_distance / 2) * floor
        position = np.array(ndimage.maximum_position(distance[sub_rect[1]:sub_rect[3], sub_rect[0]:sub_rect[2]]))
        position[0] += sub_rect[1]
        position[1] += sub_rect[0]
        position = position / scale

        # # Original floor
        # axs[0].imshow(distance, origin='lower')
        # axs[0].set_title('Original Floor')
        # axs[0].scatter([position[1] * scale], [position[0] * scale], color='red')
        
        # # Draw a rectangle from min_corner to max_corner

        # # Plot the point (min_corner[0] + position[1], min_corner[1] + position[0])
        # axs[1].add_patch(plt.Rectangle((min_corner[0] + sub_rect[0] / scale, min_corner[1] + sub_rect[1] / scale), (sub_rect[2] - sub_rect[0]) / scale, (sub_rect[3] - sub_rect[1]) / scale, linewidth=1, edgecolor='y', facecolor='none'))
        # axs[1].scatter([min_corner[0] + position[1]], [min_corner[1] + position[0]], color='blue')

        # axs[1].set_title(f'Room: {room.get_name()}')

        # axs[2].imshow(circle_distance, origin='lower')

        # plt.savefig(f'floor_{room.get_name()}.png')
        # plt.close()
        # print("Saved image:", f'floor_{room.get_name()}.png')

        alpha = 1

        return np.array([
            (min_corner[0] + position[1]) * alpha + (min_corner[0] + max_corner[0]) / 2 * (1 - alpha),
            (min_corner[1] + position[0]) * alpha + (min_corner[1] + max_corner[1]) / 2 * (1 - alpha),
            room.get_location()[2] + h
        ])
    
        return np.array([
            (min_corner[0] + max_corner[0]) / 2,
            (min_corner[1] + max_corner[1]) / 2,
            room.get_location()[2] + height
        ])