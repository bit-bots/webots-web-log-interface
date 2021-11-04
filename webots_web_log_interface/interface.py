import re
import os
import json
import numpy as np
import xml.etree.ElementTree as ET

from typing import Dict

from functools import lru_cache
# Fix for python < 3.9
try:
    from functools import cache
except ImportError:
    cache = lru_cache(maxsize=None)

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    pass

class WebotsGameLogParser:
    """
    The if the root class of the interface.
    It connects both the x3d and json parser and offers some high level example plots.
    """
    def __init__(self, log_folder: str, verbose=True):
        super().__init__()

        self.log_folder = log_folder

        x3d_file = [file for file in os.listdir(self.log_folder) if file.endswith(".x3d")][0]
        self.x3d = X3DParser(os.path.join(log_folder, x3d_file))

        game_json_file = [file for file in os.listdir(self.log_folder) if file.endswith(".json")][0]
        self.game_data = GameJsonParser(os.path.join(log_folder, game_json_file))

        if verbose:
            print(f"Duration: {self.get_max_player_timestamp() / 60 :.2f} Minutes")
            print(f"Players: {', '.join(self.x3d.get_player_names())}")

    def plot_ball_path(self):
        """
        Creates a combined plot with the paths for all the players
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-3, 3)
        ax.axis('equal')
        ax.plot(*self.game_data.get_translations_for_id(self.x3d.get_ball_id()).T[0:2])
        plt.show()

    def plot_player_paths(self):
        """
        Creates a combined plot with the paths for all the players
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-3, 3)
        ax.axis('equal')
        for bot in self.x3d.get_player_names():
            ax.plot(*self.game_data.get_translations_for_id(self.x3d.get_player_id(bot)).T[0:2])
        #ax.plot(*self.game_data.get_translations_for_id(self.x3d.get_ball_id()).T[0:2], linewidth=5)
        plt.show()

    def plot_path(self, id: int):
        """
        Creates a plot with the path for a certain object
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(*self.game_data.get_translations_for_id(id).T)
        plt.show()

    def get_max_player_timestamp(self) -> float:
        """
        Returns the last timestamp where any of the players has movement data
        """
        return max(self.game_data.get_timestamps_for_id(id).max() for id in self.x3d.get_player_ids())


class GameJsonParser:
    """
    Parses time series information from the .json log
    """
    def __init__(self, jsonfile: str):
        self.jsonfile = jsonfile

        # Load file contents
        with open(self.jsonfile, "r") as f:
            self.data = json.load(f)

        # Cache poses for id
        self._poses_for_id_cache = {}

    @property
    def get_time_step_size(self) -> int:
        """
        Returns the basic time step in ms.
        """
        return self.data["basicTimeStep"]

    def _parse_str_vector(self, vec: str) -> np.ndarray:
        """
        Converts a space divided string vector into a NumPy array.
        """
        return np.array([float(num) for num in vec.split(" ")], dtype=float)

    def get_poses_for_id(self, id: int) -> Dict[str, np.ndarray]:
        """
        Gets all the pose data for an object and
        return a list of dicts containing the time and the pose.
        """
        if not id in self._poses_for_id_cache.keys():
            poses = []
            for frame in self.data["frames"]:
                time = frame["time"]
                if "poses" in frame.keys():
                    for sim_object in frame["poses"]:
                        if sim_object["id"] == id:
                            if "translation" in sim_object.keys():
                                translation = self._parse_str_vector(sim_object["translation"])
                            if "rotation" in sim_object.keys():
                                rotation = self._parse_str_vector(sim_object["rotation"])
                            poses.append({
                                "time": time,
                                "trans": translation,
                                "rot": rotation
                            })
                            break
            # Clean out of bounds poses
            poses = self.cleanup_poses(poses)
            # Map the list of dicts to a dict of ndarrays to save memory
            self._poses_for_id_cache[id] = {key: np.array([p[key] for p in poses], dtype=float) for key in poses[0].keys()}

        return self._poses_for_id_cache[id]

    def get_translations_for_id(self, id: int) -> np.ndarray:
        """
        Returns an NumPy array with all translations for an object (in meters)
        """
        return self.get_poses_for_id(id)["trans"]

    def get_orientations_for_id(self, id: int) -> np.ndarray:
        """
        Returns an NumPy array with all orientations for an object (as axis angle)
        """
        return self.get_poses_for_id(id)["rot"]

    def get_timestamps_for_id(self, id: int) -> np.ndarray:
        """
        Returns an NumPy array with all timesteps for an object (in seconds)
        """
        # Get correct array and scale it to seconds
        return self.get_poses_for_id(id)["time"] / 1000

    def get_velocity_vectors_for_id(self, id: int):
        """
        Calcs vel vecs for a given id (in m/s)
        """
        Δtrans = np.diff(self.get_translations_for_id(id), axis=0)
        Δtime = np.diff(self.get_timestamps_for_id(id))
        return Δtrans / Δtime[:, np.newaxis]

    def get_velocities_for_id(self, id: int):
        """
        Calcs vels for a given id (in m/s)
        """
        return np.linalg.norm(self.get_velocity_vectors_for_id(id), axis=1)

    def cleanup_poses(self, poses: [dict])-> [dict]:
        """
        Removes out of bounds poses caused by e.g. the referee
        """
        def in_bounds(pose: dict, treshold: float = 4.5) -> bool:
            return all(treshold > np.abs(pose["trans"]))

        return list(filter(in_bounds, poses))

    def get_interpolated_translations(self, id: int, start: float = None, stop: float = None, step_size: float = 0.1) -> np.ndarray:
        """
        Returns the translation data interpolated to fit a given step size
        """
        # Get the data that should be interpolated
        fp = self.get_translations_for_id(id)
        xp = self.get_timestamps_for_id(id)
        # Add default values if necessary. Use the first and last timestep for the given id.
        if start is None: start = xp[0]
        if stop is None: stop = xp[-1]
        # Create the array of query values
        x_steps = np.arange(start=start, stop=stop, step=step_size)
        # Interpolate for each axis
        inter = np.vstack([np.interp(x_steps, xp, fp[:, i]) for i in range(3)])
        return inter.T

    def get_interpolated_orientations(self, id: int, start: float = None, stop: float = None, step_size: float = 0.1) -> np.ndarray:
        """
        Returns the orientation data interpolated to fit a given step size
        """
        # Get the data that should be interpolated
        fp = self.get_orientations_for_id(id)
        xp = self.get_timestamps_for_id(id)
        # Add default values if necessary. Use the first and last timestep for the given id.
        if start is None: start = xp[0]
        if stop is None: stop = xp[-1]
        # Create the array of query values
        x_steps = np.arange(start=start, stop=stop, step=step_size)
        # The orientation is in axes angles, so we need to split the angle in two components
        sin_angles = np.sin(fp[:, 3])
        cos_angles = np.cos(fp[:, 3])
        fp = [fp[:, 0], fp[:, 1], fp[:, 2], sin_angles, cos_angles]
        # Interpolate for each axis
        inter = np.vstack([np.interp(x_steps, xp, fpe) for fpe in fp])
        # Convert sin cos vector back to an single angle
        inter[3] = np.arctan2(inter[3], inter[4])
        return inter[:4].T


class X3DParser:
    """
    Parses the .x3d file in the gamelogs, that contains all the 3d object information.
    """
    def __init__(self, x3dfile: str):
        super(X3DParser).__init__()
        self.x3d_file_path = x3dfile

        # Load file contents
        with open(self.x3d_file_path, "r") as f:
            x3d_file_raw_content = f.read()

        self.xml_root = ET.fromstring(x3d_file_raw_content)

    @cache
    def get_players(self) -> [dict]:
        """
        Get the names of the loaded players
        """
        def is_player(node: ET.Element) -> bool:
            return "player" in node.attrib.get("name", "")

        def simplify_dict(node: ET.Element) -> dict:
            return {"id": int(node.attrib["id"][1:]),
                    "name": str(node.attrib["name"])}

        return list(map(simplify_dict, filter(is_player, self.xml_root.iter("Transform"))))

    def get_player_names(self) -> [str]:
        """
        Returns a list with all player names
        """
        return list(map(lambda x: x["name"], self.get_players()))

    def get_player_id(self, name: str) -> int or None:
        """
        Returns the object id for a given player name
        """
        return (list(map(lambda x: x["id"], filter(lambda x: x["name"] == name, self.get_players()))) + [None])[0]

    @cache
    def get_object_id(self, name: str) -> int or None:
        """
        Returns the object id for a object name
        """
        def equals_name(node: ET.Element) -> bool:
            return node.attrib.get("name", "") == name

        def simplify_to_id(node: ET.Element) -> id:
            return int(node.attrib["id"][1:])

        # Get all nodes from xml tree that match the name exactly
        nodes_filtered_by_name = filter(equals_name, self.xml_root.iter("Transform"))

        # Reduce them to their integer id
        ids_from_filtered_nodes = map(simplify_to_id, nodes_filtered_by_name)

        # Select the first candidate or None if there is no match
        first_or_none = (list(ids_from_filtered_nodes) + [None])[0]

        return first_or_none

    def get_player_ids(self) -> [int]:
        """
        Returns a list with all player object ids
        """
        return list(map(self.get_player_id, self.get_player_names()))

    def get_ball_id(self) -> int:
        """
        Returns the object id of the ball
        """
        return self.get_object_id("robocup soccer ball")
