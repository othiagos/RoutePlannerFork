from GeometryOperations import (draw_cylinder_with_hemisphere, compute_central_hemisphere_area, intersect_plane_sphere)
from typing import Tuple, Dict, Any, List
from numpy import ndarray, dtype, floating, float_, bool_
from numpy._typing import _64Bit
from scipy.spatial import ConvexHull, Delaunay
from CoppeliaInterface import CoppeliaInterface
from config import parse_settings_file
from random import sample
import numpy as np
import pyvista as pv
import os
import sys
import pickle
import shutil
import subprocess
import csv
import ast
import cv2 as cv
import datetime
import platform
import pymeshlab
import multiprocessing

# Variables loaded from config.yaml
CA_max = -1  # Bigger number the route has more points
max_route_radius = -1  # Bigger number the route increase the maximum points of view radius.
points_per_sphere = -1  # Density of points in the radius. If the number increase density decrease
height_proportion = -1  # The proportion of the tallest z height to make the cylinder
max_visits = -1  # Define the maximum number of times that the point can be visited
max_iter = -1  # Maximum number of iterations to try...catch a subgroup
T_max = -1  # Maximum travel budget
n_resolution = -1  # Number of subdivisions of the horizontal discretization
points_per_unit = -1
scale_to_height_spiral = 1.5  # Scale multiplied by the object target centroid Z to compute the spiral trajectory Z
search_size = 20  # Size of the random points that will be used to search the next position of the UAV.
number_of_line_points = 10  # The number of the points that will be used to define a line that will be verified if is through the convex hull
feature_extractor_file_name = 'config/feature_extractor.ini'
exhaustive_matcher_file_name = 'config/exhaustive_matcher.ini'
mapper_file_name = 'config/mapper.ini'
image_undistorter_file_name = 'config/image_undistorter.ini'
patch_match_stereo_file_name = 'config/patch_match_stereo.ini'
stereo_fusion_file_name = 'config/stereo_fusion.ini'
poisson_mesher_file_name = 'config/poisson_mesher.ini'


def run_colmap_program(colmap_folder: str, workspace_folder: str, images_folder: str) -> None:
    if platform.system() == 'Windows':
        run_colmap(os.path.join(colmap_folder, 'COLMAP.bat'), workspace_folder, str(images_folder))

    if platform.system() == 'Linux':
        run_colmap('colmap', workspace_folder, images_folder)


def write_config_file(config_file_name: str, workspace_folder: str, config_lines: list[str]):
    config_path = os.path.join(workspace_folder, os.path.basename(config_file_name)).replace("\\", "/")
    with open(config_path, 'w') as config_file:
        config_file.writelines(config_lines)
    return config_path


def execute_colmap_command(colmap_exec: str, command: str, config_file_path: str):
    process = subprocess.Popen([colmap_exec, command, '--project_path', config_file_path])
    process.communicate() # Wait for the process to finish


def run_colmap(colmap_exec: str, workspace_folder: str, image_folder: str):
    """
    Execute the COLMAP script on Windows
    :param colmap_folder: Folder where is stored the COLMAP.bat file
    :param workspace_folder: Folder where the COLMAP results will be stored
    :param image_folder: Folder to images used for reconstruction. There is no name pattern to images
    :return: Nothing
    """
    print('Executing colmap script ...')
    try:
        # Extract features
        with open(feature_extractor_file_name, 'r') as feature_config_file_read:
            feature_extractor_config_str = feature_config_file_read.readlines()
            feature_extractor_config_str[3] = f'database_path={workspace_folder}/database.db\n'
            feature_extractor_config_str[4] = f'image_path={image_folder}\n'

        feature_config_path = write_config_file(feature_extractor_file_name, 
                                                workspace_folder,
                                                feature_extractor_config_str)
        execute_colmap_command(colmap_exec, 'feature_extractor', feature_config_path)

        # Perform exhaustive matching
        with open(exhaustive_matcher_file_name, 'r') as exhaustive_matcher_file_read:
            exhaustive_matcher_config_str = exhaustive_matcher_file_read.readlines()
            exhaustive_matcher_config_str[3] = f'database_path={workspace_folder}/database.db\n'

        exhaustive_matcher_config_path = write_config_file(exhaustive_matcher_file_name, 
                                                           workspace_folder, 
                                                           exhaustive_matcher_config_str)
        execute_colmap_command(colmap_exec, 'exhaustive_matcher', exhaustive_matcher_config_path)
        
        # Create sparse folder
        sparse_dir = os.path.join(workspace_folder, 'sparse').replace("\\", "/")
        os.mkdir(sparse_dir)
        
        # Run the mapper
        with open(mapper_file_name, 'r') as mapper_file_read:
            mapper_config_str = mapper_file_read.readlines()
            mapper_config_str[3] = f'database_path={workspace_folder}/database.db\n'
            mapper_config_str[4] = f'image_path={image_folder}\n'
            mapper_config_str[5] = f'output_path={sparse_dir}\n'
        
        mapper_config_path = write_config_file(mapper_file_name, 
                                               workspace_folder, 
                                               mapper_config_str)
        execute_colmap_command(colmap_exec, 'mapper', mapper_config_path)


        if settings['dense model'] == 1:
            # Create dense folder
            dense_dir = os.path.join(workspace_folder, 'dense').replace("\\", "/")
            os.mkdir(dense_dir)

        for folder in os.listdir(sparse_dir):

            if settings['dense model'] == 0:
                break

            sub_dense_dir = os.path.join(dense_dir, folder).replace("\\", "/")
            sub_sparse_dir = os.path.join(sparse_dir, folder).replace("\\", "/")
            os.mkdir(sub_dense_dir)

            # Undistort images
            with open(image_undistorter_file_name, 'r') as image_undistorter_file_read:
                image_undistorter_config_str = image_undistorter_file_read.readlines()
                image_undistorter_config_str[0] = f'image_path={image_folder}\n'
                image_undistorter_config_str[1] = f'input_path={sub_sparse_dir}\n'
                image_undistorter_config_str[2] = f'output_path={sub_dense_dir}\n'

            image_undistorter_config_path = write_config_file(image_undistorter_file_name, 
                                                              workspace_folder, 
                                                              image_undistorter_config_str)
            execute_colmap_command(colmap_exec, 'image_undistorter',image_undistorter_config_path)

            # Perform stereo matching
            with open(patch_match_stereo_file_name, 'r') as patch_match_stereo_file_read:
                patch_match_stereo_config_str = patch_match_stereo_file_read.readlines()
                patch_match_stereo_config_str[3] = f'workspace_path={sub_dense_dir}\n'
            
            patch_match_stereo_config_path = write_config_file(patch_match_stereo_file_name, 
                                                               workspace_folder, 
                                                               patch_match_stereo_config_str)
            execute_colmap_command(colmap_exec, 'patch_match_stereo', patch_match_stereo_config_path)

            # Perform stereo fusion
            with open(stereo_fusion_file_name, 'r') as stereo_fusion_file_read:
                stereo_fusion_config_str = stereo_fusion_file_read.readlines()
                stereo_fusion_config_str[3] = f'workspace_path={sub_dense_dir}\n'
                stereo_fusion_config_str[6] = f'output_path={sub_dense_dir}/fused.ply\n'
            
            stereo_fusion_config_path = write_config_file(stereo_fusion_file_name, 
                                                          workspace_folder, 
                                                          stereo_fusion_config_str)
            execute_colmap_command(colmap_exec, 'stereo_fusion', stereo_fusion_config_path)

            # Generate mesh using Poisson meshing
            with open(poisson_mesher_file_name, 'r') as poisson_mesher_file_read:
                poisson_mesher_config_str = poisson_mesher_file_read.readlines()
                poisson_mesher_config_str[3] = f'input_path={sub_dense_dir}/fused.ply\n'
                poisson_mesher_config_str[4] = f'output_path={sub_dense_dir}/meshed-poisson.ply\n'

            poisson_mesher_config_path = write_config_file(poisson_mesher_file_name, 
                                                           workspace_folder, 
                                                           poisson_mesher_config_str)
            execute_colmap_command(colmap_exec, 'poisson_mesher', poisson_mesher_config_path)
        
        print("Script executed successfully.")
    except Exception as e:
        print("An error occurred:", e)
        raise RuntimeError('Colmap could not be executed correctly')


def statistics_colmap(colmap_folder_sc, workspace_folder_sc, MNRE_array=np.empty(0)) -> ndarray | None:
    print('Creating colmap statistics.')
    i = 0
    try:
        while True:
            statistic_folder = os.path.join(workspace_folder_sc, f'sparse/{i}/')
            if os.path.exists(statistic_folder):
                # exec_name = ''
                if platform.system() == 'Windows':
                    colmap_exec = colmap_folder_sc + 'COLMAP.bat'
                if platform.system() == 'Linux':
                    colmap_exec = 'colmap'

                with subprocess.Popen([colmap_exec, 'model_analyzer', '--path', statistic_folder], shell=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
                    output, stderr = process.communicate(timeout=10)  # Capture stdout and stderr. The result is shown on stderr

                # Check if there were any errors
                if process.returncode != 0:
                    print("Error executing script:")
                    print(stderr)
                else:
                    if output is not None:
                        print(output)
                    if stderr is None:
                        print('model_analyzer do not create any data')
                        return
                    else:
                        print(stderr)
                        # Read data from COLMAP output
                        points_value_idx = stderr.find(':', stderr.find('Points')) + 1
                        number_of_points = int(stderr[points_value_idx: stderr.find('\n', points_value_idx)])  # Number of points
                        error_value_idx = stderr.find(':', stderr.find('Mean reprojection error')) + 1
                        error_value = float(stderr[error_value_idx: stderr.find('p', error_value_idx)])  # Reconstruction error
                        MNRE = error_value / number_of_points  # Compute de Mean Normalized Reconstruction Error

                        # Save important data to file
                        with open(statistic_folder + 'MNRE.txt', 'w') as statistic_file:
                            statistic_file.write(f'MNRE: {MNRE}\n')
                            statistic_file.write(f'Mean reprojection error: {error_value}\n')
                            statistic_file.write(f'Points: {number_of_points}')

                        with open(statistic_folder + 'stat.txt', 'w') as stat_file:
                            stat_file.write(stderr)

                        MNRE_array = np.concatenate((MNRE_array, [MNRE]))
                    print("COLMAP data model analyzer executed successfully.")
            else:
                break
            # statistic_file.close()
            # stat_file.close()
            i += 1
        return MNRE_array
    except Exception as e:
        print("An error occurred:", e)
        return None


def is_point_inside(point, hull):
    """
    Verify is a point is inside a Delaunay convex hull
    :param point: Point to be evaluated
    :param hull: The convex hull computed by Delaunay function of Scipy
    :return point_in_hull: Boolean denoting if point is inside the hull True=Yes, False=No
    """
    # Check if the given point is within the convex hull
    point_in_hull = hull.find_simplex(point) >= 0
    return point_in_hull


def is_line_through_convex_hull(hull, line):
    """
    Verify if a line pass by a Delaunay convex hull
    :param hull: he convex hull computed by Delaunay function of Scipy
    :param line: Points on a line
    :return: Boolean denoting if line goes through the hull True=Yes, False=No
    """
    for point in line:
        if is_point_inside(point, hull):
            return True
    return False


def points_along_line(start_point, end_point, num_points):
    """
    Returns points in a line on 3D space
    :param start_point: Start point of the line
    :param end_point:  End point of the line
    :param num_points: Number of points between start and end point
    :return points: The points in the line
    """
    # Generate num_points equally spaced between start_point and end_point
    x = np.linspace(start_point[0], end_point[0], num_points)
    y = np.linspace(start_point[1], end_point[1], num_points)
    z = np.linspace(start_point[2], end_point[2], num_points)
    points = np.column_stack((x, y, z))
    return points

def subgroup_formation(targets_border_sf: dict, points_of_view_contribution_sf: dict, target_points_of_view_sf: dict,
                       positions: dict = None) -> tuple[dict, int]:
    """
    Forms the subgroups of points of view around each object.
    :param positions:
    :param targets_border_sf: Dictionary with the convex hull computed by Delaunay function of Scipy for each target object
    :param points_of_view_contribution_sf: Dictionary with a reward of each point of view. Each key of the dictionary is an object target
    :param target_points_of_view_sf: Dictionary with the positions of each point of view around each target object
    :return S: Dictionary with subgroup. Each key is a target object
    :return length: Total number of subgroups
    """
    print('Starting subgroup formation')
    S = {}
    contribution = 0
    subgroup_idx = 0
    is_first_target = True
    cont_target = 0
    length = 0
    # Get the points of view for each target object
    for target, points in target_points_of_view_sf.items():
        if positions is None:
            object_area = 1.0
        else:
            object_area = ConvexHull(positions[target]).volume
        CA_max_sf = CA_max * object_area/10
        S[target] = []
        # Create a subgroup 0 with position and orientation equals to zero. This subgroup is the start and end subgroup
        if subgroup_idx == 0:
            S[target].append([])
            S[target][-1].append((subgroup_idx, subgroup_idx, 0, 0, 0.0, 0.0, 0, 0))
            subgroup_idx += 1
        visits_to_position = np.zeros(points.shape[0])  # Number of visits to a point of view. Used to determine the maximum number of visits to a point of view
        if is_first_target:
            visits_to_position[0] += max_visits + 1
        indexes_of_ini_points = list(range(points.shape[0]))
        random_points = sample(indexes_of_ini_points, len(indexes_of_ini_points))  # Selects randomly the index of points to form the groups
        show_number_of_points = 0
        for i in random_points:
            CA = 0
            total_distance = 0
            S[target].append([])
            prior_idx = i
            max_idx = -1
            idx_list = [i]
            show_number_of_points += 1
            iteration = 0
            while CA < CA_max_sf and iteration < max_iter:
                iteration += 1
                indexes_of_points = np.random.randint(low=0, high=points.shape[0], size=search_size)  # Select randomly the index of points where the drone can go
                max_contribution = 0
                for index in indexes_of_points:
                    distance_p2p = np.linalg.norm(target_points_of_view_sf[target][prior_idx, :3] - target_points_of_view_sf[target][index, :3])
                    contribution = abs(abs(points_of_view_contribution_sf[target][index]) - distance_p2p)  #
                    if contribution > max_contribution:
                        max_idx = index
                        max_contribution = abs(points_of_view_contribution_sf[target][index])

                if max_idx == -1:
                    continue

                if visits_to_position[max_idx] > max_visits:
                    continue
                is_line_through_convex_hull_sf = False
                for target_compare, hull_sf in targets_border_sf.items():
                    line_points = points_along_line(target_points_of_view_sf[target][prior_idx, :3], target_points_of_view_sf[target][max_idx, :3], number_of_line_points)
                    is_line_through_convex_hull_sf = is_line_through_convex_hull(hull_sf, line_points)
                    if is_line_through_convex_hull_sf:
                        break
                if is_line_through_convex_hull_sf:
                    continue

                idx_list.append(max_idx)
                distance_p2p = np.linalg.norm(
                    target_points_of_view_sf[target][prior_idx, :3] - target_points_of_view_sf[target][max_idx, :3])
                total_distance = distance_p2p + total_distance
                CA += contribution
                visits_to_position[max_idx] += 1
                prior_idx_s = length + prior_idx
                max_idx_s = length + max_idx
                # Dictionary with subgroups by object target. Each target has n subgroups. And each subgroup has your elements which is composed by a tuple with:
                S[target][-1].append((subgroup_idx,  # General index of the group considering all subgroups of all objects
                                      prior_idx,  # Index of the previous visited point of view. This index is by target
                                      max_idx,  # Index of the next visited point of view. This index is by target
                                      distance_p2p,  # Euclidean distance between the start and end points.
                                      total_distance,  # Total travelled distance until the point max_idx. The last element of the subgroup will have the total travelled distance in subgroup
                                      CA,  #  Total reward of the subgroup until the max_idx point. The last element of the subgroup will have the total reward for the subgroup
                                      prior_idx_s,  # Index of the previous visited point of view. This index considering all target
                                      max_idx_s))  # Index of the next visited point of view. This index is considering all target
                prior_idx = max_idx
            # If the above step do not reach the CA minimum shows a message to user.
            # if iteration >= max_iter - 1:
            #     print('Decrease CA_max')
            #     print(f'{CA=}')
            #     print(f'{len(S[target][-1])=}')
            # If the subgroup is empty remove it from the subgroup list

            if len(S[target][-1]) == 0 or CA < settings['CA_min']:
                S[target].pop()
            else:
                subgroup_idx += 1

        length += len(S[target])  # Compute the total length of the subgroups
        is_first_target = False
        cont_target += 1
        print(f'{target=} has {len(S[target])=} groups')
    return S, length


def find_route(S_fr: dict):
    """
    NOT USED!!!
    Find a route based on better subgroup reward only
    :param S_fr: Dictionary with subgroups
    :return route: Subgroups that forms a route
    """
    print('Starting finding a route ...')
    route = {}
    for target, s_fr in S_fr.items():
        CA_fr_max = -1
        Si_chose = []
        for Si_fr in S_fr[target]:
            if Si_fr[-1][-1] > CA_fr_max:
                Si_chose = Si_fr
                CA_fr_max = Si_fr[-1][-1]
        route[target] = Si_chose
    return route


def get_points_to_route(route_points_gpfr: list[tuple], points_table_gpfr: list[ndarray]) -> ndarray:
    """
    Make an array with the point chose for the route. Separate it from the subgroups.
    :param route_points_gpfr: Dictionary of subgroups that forms a route
    :param points_table_gpfr: Dictionary of points of view
    :return:  Array with points
    """
    route_result_points_gpfr = np.empty(0)
    for point_pair in route_points_gpfr:
        if len(route_result_points_gpfr) == 0:
            route_result_points_gpfr = points_table_gpfr[point_pair[0]]
            continue
        route_result_points_gpfr = np.row_stack((route_result_points_gpfr, points_table_gpfr[point_pair[1]]))
    return route_result_points_gpfr


def save_points(route_sp: dict, targets_points_of_view_sr: dict):
    """
    UNDER CONSTRUCTION!!!!!
    Save the points of each route
    :param route_sp: Dictionary with subgroups with route
    :param targets_points_of_view_sr: Points of view to be converted in a route
    :return: Nothing
    """
    print('Starting saving ...')
    route_points = np.empty([0, 6])
    for target, data_s in route_sp.items():
        for data in data_s:
            point_start = targets_points_of_view_sr[target][data[0]]
            point_end = targets_points_of_view_sr[target][data[1]]
            route_points = np.row_stack((route_points, point_end))
    np.savetxt('positions.csv', route_points, delimiter=',')


def initializations(copp_i) -> tuple[
    dict[Any, ndarray[Any, dtype[floating[_64Bit] | float_]] | ndarray[Any, dtype[Any]]], dict[Any, Delaunay], dict[Any, tuple[ndarray[Any, dtype[Any]], float]], dict[Any, tuple[ndarray[Any, dtype[Any]], float]]]:
    """
    Function to get the points from CoppeliaSim. The points of each object cannot be at the same plane, at least one
    must be a different plane. On CoppeliaSim you must add discs around the object to form a convex hull these points
    must call Disc[0], Disc[1], ... , Disc[n]. These points must be son of a plane named O[0], O[1], ... , O[n]. These
    objects in CoppeliaSim scene must have the property Object is model on window Scene Object Properties checked. To
    access these properties, you can only double-click on an object.
    :return positions: Dictionary with all positions of the viewpoints in each target object
    :return target_hull_i: Dictionary with the convex hull computed by Delaunay for each target object
    :return centroid_points_i: Dictionary with the centroid of the convex hull for each target object
    :return radius_i: Dictionary with the radius of the convex hull
    """
    positions = {}
    j = 0
    targets_hull_i = {}
    centroid_points_i = {}
    radius_i = {}
    settings = parse_settings_file('config.yaml')
    for object_name_i in settings['object names']:
        copp_i.handles[object_name_i] = copp_i.sim.getObject(f"./{object_name_i}")
        if copp_i.handles[object_name_i] < 0:
            break
        positions[object_name_i] = np.empty([0, 3])
        i = 0
        # Read the points of objects.
        # The object in CoppeliaSim must have primitive forms, discs with positions of external points.
        # Disc names must be Disc[0], Disc[1], ..., Disc[n]
        while True:
            points_names = f"./{object_name_i}/Disc"
            handle = copp_i.sim.getObject(points_names, {'index': i, 'noError': True})
            if handle < 0:
                break
            positions[object_name_i] = np.row_stack((positions[object_name_i], copp_i.sim.getObjectPosition(handle, copp_i.sim.handle_world)))
            i += 1

        targets_hull_i[object_name_i] = Delaunay(positions[object_name_i])  # Compute the convex hull of the target objects
        centroid_points_i[object_name_i], radius_i[object_name_i] = _centroid_poly(positions[object_name_i])  # Compute the centroid of objects
        j = j + 1

    return positions, targets_hull_i, centroid_points_i, radius_i


def _centroid_poly(poly: np.ndarray) -> tuple[ndarray[Any, dtype[floating[Any]]], float]:
    """
    Compute the centroid point for a Delaunay convex hull
    :param poly: Delaunay convex hull
    :return tmp_center: Geometric center position of the target object
    """
    T = Delaunay(poly).simplices
    n = T.shape[0]
    W = np.zeros(n)
    C = np.zeros(3)

    for m in range(n):
        sp = poly[T[m, :], :]
        sp += np.random.normal(0, 1e-10, sp.shape)
        W[m] = ConvexHull(sp).volume
        C += W[m] * np.mean(sp, axis=0)

    tmp_center = C / np.sum(W)
    max_distance = 0.0
    for m in range(n):
        sp = poly[T[m, :], :2]
        for spl in sp:
            distance = np.linalg.norm(spl - tmp_center[:2])
            if distance > max_distance:
                max_distance = distance

    return tmp_center, max_distance


def get_geometric_objects_cell(geometric_objects):
    for i in range(geometric_objects.n_cells):
        yield geometric_objects.get_cell(i)


def find_normal_vector(point1, point2, point3):
    vec1 = np.array(point2) - np.array(point1)
    vec2 = np.array(point3) - np.array(point1)
    cross_vec = np.cross(vec1, vec2)
    return cross_vec / np.linalg.norm(cross_vec)


def euler_angles_from_normal(normal_vector):
    """
    Computes Euler angles (in degrees) based on a normal vector of direction.

    Args:
    - normal_vector: A numpy array representing the normal vector of direction.

    Returns:
    - Euler angles (in degrees) as a tuple (roll, pitch, yaw).
    """
    # Normalize the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # Calculate yaw angle
    yaw = np.arctan2(normal_vector[1], normal_vector[0]) * 180 / np.pi

    # Calculate pitch angle
    pitch = np.arcsin(-normal_vector[2]) * 180 / np.pi

    # Calculate roll angle
    roll = np.arctan2(normal_vector[2], np.sqrt(normal_vector[0] ** 2 + normal_vector[1] ** 2)) * 180 / np.pi

    return yaw, pitch, roll


def draw_cylinders_hemispheres(centroid_points_pf: dict,
                               radius_pf: dict,
                               target_points_pf: dict) -> tuple[dict[Any, ndarray[Any, dtype[Any]] | ndarray[Any, dtype[floating[_64Bit] | float_]]], dict[Any, list[float] | list[Any]], list[ndarray[Any, dtype[Any]]]]:
    """
    Draw the hemispheres and the cylinders around the object
    :param centroid_points_pf: Dictionary of arrays of points with the central points of the objects
    :param radius_pf: Computed radius of the cylinders around each object
    :param target_points_pf: Computed points for the convex hull of the objects
    :return: vector_of_points: Dictionary with points around each object
    :return: Dictionary for weight of each point
    """
    print('Starting showing data')
    # Create a plotter
    plotter = pv.Plotter()
    vector_points_pf = {}  # Dictionary with points of view around each object
    vector_points_weight_pf = {}  # Dictionary of weights to each point
    central_area_computed = False  # Verify if the weight to each point in the normal was computed
    computed_area_by_hemisphere = []  # Stores the computed area to each point computed the first time
    is_included_first_group = False
    conversion_table = []
    for target in centroid_points_pf.keys():
        cy_direction = np.array([0, 0, 1])
        cy_hight = height_proportion * (np.max(target_points_pf[target][:, 2]) - np.min(target_points_pf[target][:, 2]))
        r_mesh = radius_pf[target]
        h = np.cos(np.pi / n_resolution) * r_mesh
        # l = np.sqrt(np.abs(4 * h ** 2 - 4 * r_mesh ** 2))

        # Find the radius of the spheres
        meshes = draw_cylinder_with_hemisphere(plotter, cy_direction, cy_hight, n_resolution, r_mesh,
                                               centroid_points_pf[target], 0.0)
        cylinder = meshes['cylinder']['mesh']
        if not is_included_first_group:
            vector_points_pf[target] = np.array([-2.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            vector_points_weight_pf[target] = [0.0]
            conversion_table.append(np.array([-2.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
            is_included_first_group = True
        else:
            vector_points_pf[target] = np.empty([0, 6])
            vector_points_weight_pf[target] = []
        count_hemisphere = 0
        route_radius_dch = int(np.fix(max_route_radius / points_per_unit))
        weights = (route_radius_dch - 1) * [0.0]
        for cell in get_geometric_objects_cell(cylinder):
            hemisphere_radius = meshes['hemispheres'][count_hemisphere]['radius']  #
            pos_cell = cell.center
            points_cell = cell.points[:3]
            norm_vec = find_normal_vector(*points_cell)
            yaw, pitch, roll = euler_angles_from_normal(-norm_vec)
            for k in range(1, route_radius_dch):
                camera_distance = ((points_per_sphere * k) + 1) * hemisphere_radius
                point_position = pos_cell + camera_distance * norm_vec
                if (count_hemisphere == 0 or count_hemisphere == n_resolution or
                        count_hemisphere == cylinder.n_cells - n_resolution):
                    spherical_area_dc, reach_maximum, frustum_planes, cam_pos = (
                        compute_central_hemisphere_area(norm_vec, pos_cell, hemisphere_radius, camera_distance, plotter,
                                                        float(settings['perspective angle']),
                                                        near_clip_ccha=float(settings['near clip']),
                                                        far_clip_ccha=float(settings['far clip'])))
                    area = get_side_hemisphere_area(cylinder.n_cells,
                                                    meshes,
                                                    frustum_planes,
                                                    count_hemisphere)
                    weight = spherical_area_dc + area
                    weights[k - 1] = weight
                else:
                    weight = weights[k - 1]
                vector_points_pf[target] = np.row_stack((vector_points_pf[target], np.concatenate((point_position, np.array([yaw, pitch, roll])))))
                conversion_table.append(np.concatenate((point_position, np.array([yaw, pitch, roll]))))
                vector_points_weight_pf[target].append(weight)
            count_hemisphere += 1

        points0 = vector_points_pf[target][:, :3]
        point_cloud0 = pv.PolyData(points0)
        plotter.add_mesh(point_cloud0)

        # cylinder.plot(show_edges=True)
        plotter.add_mesh(cylinder, show_edges=True)

        points = target_points_pf[target]
        point_cloud = pv.PolyData(points)
        plotter.add_mesh(point_cloud)

    # plotter.show()
    return vector_points_pf, vector_points_weight_pf, conversion_table


def point_between_planes(point, planes: ndarray):
    x, y, z = point
    count_true = 0
    for i in range(planes.shape[0]):
        for j in range(i + 1, planes.shape[0]):
            A1, B1, C1, D1 = planes[i]
            A2, B2, C2, D2 = planes[j]
            if A1 * x + B1 * y + C1 * z + D1 < 0 and A2 * x + B2 * y + C2 * z + D2 > 0:
                count_true += 1
            if A1 * x + B1 * y + C1 * z + D1 > 0 and A2 * x + B2 * y + C2 * z + D2 < 0:
                count_true += 1
    if count_true >= 2:
        return True
    else:
        return False


def get_side_hemisphere_area(count_plane_gsha: int,
                             meshes_gsha: dict,
                             frustum_planes: list,
                             central_hemisphere_gsha: int) -> float:
    tmpidxs = 49 * [[]]
    number_of_elements = 0
    tmpidxs[number_of_elements] = central_hemisphere_gsha
    number_of_elements += 1
    for count_idx in range(1, 3):
        tmpidxs[number_of_elements] = (central_hemisphere_gsha + count_idx) % n_resolution + (
                central_hemisphere_gsha // n_resolution) * n_resolution
        number_of_elements += 1
        tmpidxs[number_of_elements] = (central_hemisphere_gsha - count_idx) % n_resolution + (
                central_hemisphere_gsha // n_resolution) * n_resolution
        number_of_elements += 1
    list_idx = tmpidxs.copy()
    total_elements = number_of_elements
    if central_hemisphere_gsha > n_resolution:
        for l in range(total_elements):
            list_idx[number_of_elements] = list_idx[l] - n_resolution
            number_of_elements += 1
    tmpidxs = list_idx.copy()
    total_elements = number_of_elements
    if central_hemisphere_gsha < count_plane_gsha - n_resolution:
        for l in range(total_elements):
            list_idx[number_of_elements] = list_idx[l] + n_resolution
            number_of_elements += 1

    list_idx = list_idx[:number_of_elements]
    area = 0
    for hemisphere_idx in list_idx[1:]:
        ct_pt = np.array(meshes_gsha['hemispheres'][hemisphere_idx]['center'])
        is_in = False
        intersection_points = []
        for plane_gsha in frustum_planes:
            distance = (abs(np.dot(plane_gsha[:3], meshes_gsha['hemispheres'][hemisphere_idx]['center']) + plane_gsha[3]) / np.sqrt(plane_gsha[0] ** 2 + plane_gsha[1] ** 2 + plane_gsha[2] ** 2))
            if distance < meshes_gsha['hemispheres'][hemisphere_idx]['radius']:
                x = (-plane_gsha[3] - meshes_gsha['hemispheres'][hemisphere_idx]['center'][1] * plane_gsha[1] -
                     meshes_gsha['hemispheres'][hemisphere_idx]['center'][2] * plane_gsha[2]) / plane_gsha[0]
                point_pi = np.array([x, meshes_gsha['hemispheres'][hemisphere_idx]['center'][1],
                                     meshes_gsha['hemispheres'][hemisphere_idx]['center'][2]])
                intersection_points = intersect_plane_sphere(np.array(plane_gsha[:3]),
                                                             point_pi,
                                                             np.array(
                                                                 meshes_gsha['hemispheres'][hemisphere_idx]['center']),
                                                             meshes_gsha['hemispheres'][hemisphere_idx]['radius'])
                is_in = True
                break
        alpha = 1
        if not is_in:
            if not point_between_planes(ct_pt, np.array(frustum_planes)):
                area += 2 * alpha * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'] ** 2
            else:
                area += 0
        else:
            if point_between_planes(ct_pt, np.array(frustum_planes)):
                area += alpha * 2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'] * np.linalg.norm(
                    intersection_points[0] - intersection_points[1])
            else:
                area += alpha * (2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'] *
                                 np.linalg.norm(intersection_points[0] - intersection_points[1]) +
                                 2 * np.pi * meshes_gsha['hemispheres'][hemisphere_idx]['radius'])
    return area


def get_points_route(vector_points_gpr: dict, route_gpr: dict):
    route_points = {}
    for target, data_s in route_gpr.items():
        route_points[target] = np.empty([0, 6])
        for data in data_s:
            point_start = vector_points_gpr[target][data[0]]
            point_end = vector_points_gpr[target][data[1]]
            route_points[target] = np.row_stack((route_points[target], point_end))
    return route_points


def plot_route(centroid_points_pf: dict, radius_pf: dict, target_points_pf: dict, vector_points_pr: dict):
    print('Starting showing data')
    # Create a plotter
    plotter = pv.Plotter()
    vector_points_pf = {}
    str_color = ['red', 'green', 'black']
    count_color = 0
    for target in centroid_points_pf.keys():
        cy_direction = np.array([0, 0, 1])
        n_resolution = 36
        cy_hight = height_proportion * np.max(target_points_pf[target][:, 2])
        r_mesh = radius_pf[target]
        h = np.cos(np.pi / n_resolution) * r_mesh
        l = np.sqrt(np.abs(4 * h ** 2 - 4 * r_mesh ** 2))

        # Find the radius of the spheres
        z_resolution = int(np.ceil(cy_hight / l))

        cylinder = pv.CylinderStructured(
            center=centroid_points_pf[target],
            direction=cy_direction,
            radius=r_mesh,
            height=1.0,
            theta_resolution=n_resolution,
            z_resolution=z_resolution,
        )

        points0 = vector_points_pr[target][:, :3]
        point_cloud0 = pv.PolyData(points0)
        plotter.add_mesh(point_cloud0, color=str_color[count_color])
        # arrow_direction = pv.Arrow(start=points0[0], direction=vector_points_pr[target][0, 3:])
        # plotter.add_mesh(arrow_direction, color=str_color[count_color])

        # cylinder.plot(show_edges=True)
        plotter.add_mesh(cylinder, show_edges=True)

        points = target_points_pf[target]
        point_cloud = pv.PolyData(points)
        plotter.add_mesh(point_cloud, color=str_color[count_color])
        count_color += 1

    # plotter.show()


def quadcopter_control(sim, client, quad_target_handle, quad_base_handle, route_qc: dict):
    """
    This method is used to move the quadcopter in the CoppeliaSim scene to the position pos.
    :param route_qc:
    :param client:
    :param sim:
    :param quad_base_handle: The handle to get the quadcopter current position
    :param quad_target_handle:  The handle to the target of the quadcopter. This handle is used to position give the
    position that the quadcopter must be after control.
    :return: A boolean indicating if the quadcopter reach the target position.
    """
    for target, position_orientation in route_qc.items():
        for i in range(position_orientation.shape[0]):
            cone_name = './' + target + f'/Cone'
            handle = sim.getObject(cone_name, {'index': i, 'noError': True})
            if handle < 0:
                break
            cone_pos = list(position_orientation[i, :3])
            sim.setObjectPosition(handle, cone_pos)
        for each_position in position_orientation:
            pos = list(each_position[:3])
            next_point_handle = sim.getObject('./new_target')
            sim.setObjectPosition(next_point_handle, pos)
            orientation = list(np.deg2rad(each_position[3:]))
            orientation_angles = [0.0, 0.0, orientation[0]]
            # sim.setObjectOrientation(quad_target_handle, [0.0, 0.0, orientation[0]], sim.handle_world)
            # while sim.getSimulationTime() < t_stab:
            #     print(sim.getObjectOrientation(quad_base_handle, sim.handle_world))
            #     client.step()
            #     continue
            # orientation_angles = sim.yawPitchRollToAlphaBetaGamma(orientation[0], orientation[2], orientation[1])
            # pos = sim.getObjectPosition(quad_base_handle, sim.handle_world)
            # camera_handle = sim.getObject('./O[0]/Cone[19]')
            # sim.setObjectOrientation(camera_handle, orientation_angles)
            # sim.setObjectPosition(camera_handle, pos)
            # client.step()
            # sim.setObjectOrientation(quad_target_handle, sim.handle_world, orientation_angles)
            total_time = sim.getSimulationTime() + settings['total simulation time']
            stabilized = False
            while sim.getSimulationTime() < total_time:
                diff_pos = np.subtract(pos, sim.getObjectPosition(quad_base_handle, sim.handle_world))
                norm_diff_pos = np.linalg.norm(diff_pos)
                if norm_diff_pos > 0.5:
                    delta_pos = 0.1 * diff_pos
                    new_pos = list(sim.getObjectPosition(quad_base_handle, sim.handle_world) + delta_pos)
                else:
                    new_pos = pos

                sim.setObjectPosition(quad_target_handle, new_pos)
                diff_ori = np.subtract(orientation_angles,
                                       sim.getObjectOrientation(quad_base_handle, sim.handle_world))
                norm_diff_ori = np.linalg.norm(diff_ori)

                if norm_diff_ori > 0.08:
                    delta_ori = 0.3 * diff_ori
                    new_ori = list(sim.getObjectOrientation(quad_base_handle, sim.handle_world) + delta_ori)
                else:
                    new_ori = orientation_angles
                sim.setObjectOrientation(quad_target_handle, new_ori)
                t_stab = sim.getSimulationTime() + settings['time to stabilize']
                while sim.getSimulationTime() < t_stab:
                    diff_pos = np.subtract(new_pos,
                                           sim.getObjectPosition(quad_base_handle, sim.handle_world))
                    diff_ori = np.subtract(new_ori,
                                           sim.getObjectOrientation(quad_base_handle, sim.handle_world))
                    norm_diff_pos = np.linalg.norm(diff_pos)
                    norm_diff_ori = np.linalg.norm(diff_ori)
                    if norm_diff_pos < 0.1 and norm_diff_ori < 0.05:
                        stabilized = True
                        break
                    client.step()
                diff_pos = np.subtract(pos, sim.getObjectPosition(quad_base_handle, sim.handle_world))
                diff_ori = np.subtract(orientation_angles, sim.getObjectOrientation(quad_base_handle, sim.handle_world))
                norm_diff_pos = np.linalg.norm(diff_pos)
                norm_diff_ori = np.linalg.norm(diff_ori)
                if norm_diff_pos < 0.1 and norm_diff_ori < 0.05:
                    stabilized = True
                    break
                client.step()
            if not stabilized:
                print('Time short')


def quadcopter_control_direct_points(sim, client, vision_handle: int,
                                     route_qc: ndarray, filename_qcdp: str, directory_name_qcdp: str):
    """
    This method is used to move the quadcopter in the CoppeliaSim scene to the position pos.
    :param route_qc:
    :param client:
    :param sim:
    :param quad_base_handle: The handle to get the quadcopter current position
    :param quad_target_handle:  The handle to the target of the quadcopter. This handle is used to position give the
    position that the quadcopter must be after control.
    :return: A boolean indicating if the quadcopter reach the target position.
    """
    count_image = 0

    for point_qcdp in route_qc:
        pos = list(point_qcdp[:3])
        orientation = list(np.deg2rad(point_qcdp[3:]))
        orientation_angles = [0.0, 0.0, orientation[0]]
        quadcopter_handle = sim.getObject('./base_vision')
        sim.setObjectPosition(quadcopter_handle, pos, sim.handle_world)
        sim.setObjectOrientation(quadcopter_handle, orientation_angles, sim.handle_world)

        total_time = sim.getSimulationTime() + settings['total simulation time']
        while sim.getSimulationTime() < total_time:
            client.step()

        get_image(sim, count_image, filename_qcdp, vision_handle, directory_name_qcdp)
        count_image += 1


def compute_edge_weight_matrix(S_cewm: dict, targets_points_of_view_cewm: dict[Any, ndarray]) -> ndarray:
    print('Starting computing distance matrix')

    i = 0
    j = 0
    total_length = 0
    for _, points_start_cewm in targets_points_of_view_cewm.items():
        total_length += points_start_cewm.shape[0]

    edge_weight_matrix_cewm = np.zeros([total_length,total_length])
    for _,points_start_cewm in targets_points_of_view_cewm.items():
        for pt1 in points_start_cewm:
            for _,points_end_cewm in targets_points_of_view_cewm.items():
                    for pt2 in points_end_cewm:
                        edge_weight_matrix_cewm[i, j] = np.linalg.norm(pt1[:3] - pt2[:3]) + np.linalg.norm(np.deg2rad(pt1[3:]) - np.deg2rad(pt2[3:]))
                        j += 1
            i += 1
            j = 0


    # edge_weight_matrix_cewm = np.zeros([length_start,length_start])
    #
    # for target_cewm, S_cewm_start in S_cewm.items():
    #     # if i == 0 and j == 0:
    #     #     edge_weight_matrix_cewm = np.zeros([2 * ((len(S_cewm_start) - 1) * len(settings['object names'])) - 1,
    #     #                                         2 * ((len(S_cewm_start) - 1) * len(settings['object names'])) - 1])
    #     for Si_cewm_start in S_cewm_start:
    #         j = 0
    #         # count_target_i = 0
    #         if Si_cewm_start[-1][6] > length_start + len(S_cewm_start) or Si_cewm_start[-1][7] > length_start + len(S_cewm_start):
    #             print('There are something wrong')
    #         length_end = 0
    #         idx1 = Si_cewm_start[-1][1]  # - count_target*targets_points_of_view_cewm[target_cewm].shape[0]
    #         conversion_table = 7000 * [[]]
    #         for target_cewm_i, S_cewm_end in S_cewm.items():
    #             for Si_cewm_end in S_cewm_end:
    #                 idx2 = Si_cewm_end[0][1]  # - count_target_i*targets_points_of_view_cewm[target_cewm_i].shape[0]
    #                 pt1 = targets_points_of_view_cewm[target_cewm][idx1]
    #                 pt2 = targets_points_of_view_cewm[target_cewm_i][idx2]
    #                 if i != j:
    #                     edge_weight_matrix_cewm[Si_cewm_start[-1][6], Si_cewm_end[0][7]] = np.linalg.norm(pt1 - pt2)
    #                 conversion_table[j] = [Si_cewm_end[0][0], j]
    #                 j += 1
    #             length_end += len(S_cewm_end)
    #         i += 1
    #     length_start += len(S_cewm_start)
    # i -= 1
    # j -= 1
    # edge_weight_matrix_cewm = edge_weight_matrix_cewm[:i, :j]
    print(f'size of edge matrix: {edge_weight_matrix_cewm.shape[0]} x {edge_weight_matrix_cewm.shape[0]}')
    return edge_weight_matrix_cewm


def ConvertArray2String(fileCA2S, array: ndarray):
    np.set_printoptions(threshold=10000000000)
    np.savetxt(fileCA2S, array, fmt='%.3f', delimiter=' ')
    return fileCA2S


def read_problem_file(filename: str) -> dict:
    read_fields = {}
    line_count = 0
    try:
        with open(filename, 'r') as file:
            for line in file:
                # Split the line using ":" as the delimiter
                parts = line.strip().split(':')
                # Ensure there are two parts after splitting
                if len(parts) == 2:
                    if parts[1].isdigit():
                        read_fields[parts[0]] = int(parts[1])
                    else:
                        read_fields[parts[0]] = parts[1]
                else:
                    read_fields[f'{line_count}'] = parts[0]
    except FileNotFoundError:
        print("File not found:", filename)
    except Exception as e:
        print("An error occurred:", e)
    return read_fields


fieldnames = ['NAME: ', 'TYPE: ', 'COMMENT: ', 'DIMENSION: ', 'TMAX: ', 'START_CLUSTER: ', 'END_CLUSTER: ',
              'CLUSTERS: ', 'SUBGROUPS: ', 'DUBINS_RADIUS: ', 'EDGE_WEIGHT_TYPE: ', 'EDGE_WEIGHT_FORMAT: ',
              'EDGE_WEIGHT_SECTION', 'GTSP_SUBGROUP_SECTION: ', 'GTSP_CLUSTER_SECTION: ']


def copy_file(source_path, destination_path):
    try:
        # Copy the file from source_path to destination_path
        shutil.copy(source_path, destination_path)
        print(f"File copied successfully from {source_path} to {destination_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def write_problem_file(dir_wpf: str, filename_wpf: str, edge_weight_matrix_wpf: ndarray, number_of_targets: int,
                       S_wpf: dict, subgroup_size_wpf: int):
    print('Starting writing problem file')
    subgroup_count = 0

    # Create the directory
    os.makedirs(dir_wpf, exist_ok=True)

    complete_file_name = dir_wpf + filename_wpf + '.cops'
    # print(f'{complete_file_name=}')
    GTSP_CLUSTER_SECTION_str = []
    with open(complete_file_name, 'w') as copsfile:
        for field_wpf in fieldnames:
            if field_wpf == 'NAME: ':
                copsfile.write(field_wpf + filename_wpf + settings['directory name'] + '\n')
            elif field_wpf == 'TYPE: ':
                copsfile.write(field_wpf + 'TSP\n')
            elif field_wpf == 'COMMENT: ':
                copsfile.write(field_wpf + 'Optimization for reconstruction\n')
            elif field_wpf == 'DIMENSION: ':
                copsfile.write(field_wpf + str(edge_weight_matrix_wpf.shape[0]) + '\n')
            elif field_wpf == 'TMAX: ':
                copsfile.write(field_wpf + str(T_max) + '\n')
            elif field_wpf == 'START_CLUSTER: ':
                copsfile.write(field_wpf + '0\n')
            elif field_wpf == 'END_CLUSTER: ':
                copsfile.write(field_wpf + '0\n')
            elif field_wpf == 'CLUSTERS: ':
                copsfile.write(field_wpf + str(number_of_targets) + '\n')
            elif field_wpf == 'SUBGROUPS: ':
                copsfile.write(field_wpf + str(subgroup_size_wpf) + '\n')
            elif field_wpf == 'DUBINS_RADIUS: ':
                copsfile.write(field_wpf + '50' + '\n')
            elif field_wpf == 'EDGE_WEIGHT_TYPE: ':
                copsfile.write(field_wpf + 'IMPLICIT' + '\n')
            elif field_wpf == 'EDGE_WEIGHT_FORMAT: ':
                copsfile.write(field_wpf + 'FULL_MATRIX' + '\n')
            elif field_wpf == 'EDGE_WEIGHT_SECTION':
                copsfile.close()
                with open(complete_file_name, 'a') as copsfile:
                    copsfile.write(field_wpf + '\n')
                    ConvertArray2String(copsfile, edge_weight_matrix_wpf)
            elif field_wpf == 'GTSP_SUBGROUP_SECTION: ':
                with open(complete_file_name, 'a') as copsfile:
                    copsfile.write(f"{field_wpf}cluster_id cluster_profit id-vertex-list\n")
                    count_cluster = 0
                    GTSP_CLUSTER_SECTION_str = [[]] * len(settings['object names'])
                    for target_wpf, S_spf in S_wpf.items():
                        GTSP_CLUSTER_SECTION_str[count_cluster] = [[]] * (len(S_spf) + 1)
                        GTSP_CLUSTER_SECTION_str[count_cluster][0] = f'{count_cluster} '
                        count_idx = 1
                        for lS_spf in S_spf:
                            copsfile.write(f"{lS_spf[0][0]} {lS_spf[-1][5]} {lS_spf[0][6]} " +
                                           ' '.join(str(vertex[7]) for vertex in lS_spf) + '\n')
                            GTSP_CLUSTER_SECTION_str[count_cluster][count_idx] = f'{lS_spf[0][0]} '
                            count_idx += 1
                        count_cluster += 1
            elif field_wpf == 'GTSP_CLUSTER_SECTION: ':
                with open(complete_file_name, 'a') as copsfile:
                    copsfile.write(f'{field_wpf} set_id id-cluster-list\n')
                    for cluster_idxs in GTSP_CLUSTER_SECTION_str:
                        copsfile.writelines(cluster_idxs)
                        copsfile.write('\n')

    copsfile.close()


def execute_script(name_cops_file: str) -> None:
    print('Executing COPS ...')
    try:
        # Execute the script using subprocess
        process = subprocess.Popen([settings['python'], settings['COPS path'] + 'tabu_search.py',
                                    f"--path={settings['COPS dataset']}" + name_cops_file], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

        # Wait for the process to finish
        stdout, stderr = process.communicate()

        # Check if there were any errors
        if process.returncode != 0:
            print("Error executing script:")
            print(stderr.decode('utf-8'))
        else:
            print("Script executed successfully.")
    except Exception as e:
        print("An error occurred:", e)


def read_route_csv_file(file_path, S_rrcf: dict, targets_points_of_vew_rrcf: dict) -> tuple[
    ndarray, float, list[ndarray]]:
    route_rrcf = np.empty([0, 6])
    route_by_group = [np.empty([0, 6])] * len(settings['object names'])
    travelled_distance = 0
    try:
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=';')
            for row in csv_reader:
                route_str = row[8]
    except Exception as e:
        print(f"An error occurred: {e}")
    chose_subgroups = ast.literal_eval(route_str.replace('  ', ','))
    bigger_idx = 0
    table_rrcf = []
    for target_rrcf, points_rrcf in targets_points_of_vew_rrcf.items():
        table_rrcf.append([target_rrcf, bigger_idx, bigger_idx + points_rrcf.shape[0]])
        bigger_idx += points_rrcf.shape[0] + 1
    count_group = 0
    is_group_zero = True
    is_group_zero_zero = True
    for S_idx_rrcf in chose_subgroups:
        for information_rrcf in table_rrcf:
            # if information_rrcf[1] <= S_idx_rrcf <= information_rrcf[2]:
            is_first_element = True
            for group_rrcf in S_rrcf[information_rrcf[0]]:
                # print(f'Group size: {len(group_rrcf)}')
                for element in group_rrcf:
                    if element[0] == S_idx_rrcf:
                        # print(f'Size of selected group: {len(group_rrcf)}')
                        pt_idx_prior = element[1]
                        pt_idx_post = element[2]
                        pt_prior_coordinates = targets_points_of_vew_rrcf[information_rrcf[0]][pt_idx_prior]
                        pt_post_coordinates = targets_points_of_vew_rrcf[information_rrcf[0]][pt_idx_post]
                        travelled_distance += np.linalg.norm(pt_post_coordinates[:3] - pt_prior_coordinates[:3])
                        if is_first_element:
                            route_rrcf = np.row_stack((route_rrcf, pt_prior_coordinates, pt_post_coordinates))
                            is_first_element = False
                            if is_group_zero_zero:
                                is_group_zero_zero = False
                            else:
                                route_by_group[count_group] = np.row_stack((route_by_group[count_group],
                                                                            pt_prior_coordinates,
                                                                            pt_post_coordinates))
                        else:
                            route_rrcf = np.row_stack((route_rrcf, pt_post_coordinates))
                            route_by_group[count_group] = np.row_stack(
                                (route_by_group[count_group], pt_post_coordinates))
        if is_group_zero:
            is_group_zero = False
        else:
            count_group += 1
    print(f'{travelled_distance=}')
    route_rrcf = np.row_stack((route_rrcf, route_rrcf[0]))
    return route_rrcf, travelled_distance, route_by_group


def get_image(sim, sequence: int, file_name: str, vision_handle: int, directory_name_gi: str):
    """
    Method used to get the image from vision sensor on coppeliaSim and save the image in a file.
    The vision handle must be previously loaded.
    :param vision_handle: Vison sensor handle to CoppeliaSim vision sensor.
    :param file_name: File name to saved image
    :param sequence: Parameter not used yet
    :return: Nothing
    """
    img, resolution = sim.getVisionSensorImg(vision_handle)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)

    # Define the directory name
    directory_name = directory_name_gi

    # Specify the path where you want to create the directory
    path = settings['path']  # You can specify any desired path here

    # Construct the full path
    full_path = os.path.join(path, directory_name)

    # Check if the directory already exists
    if not os.path.exists(full_path):
        # Create the directory
        os.makedirs(full_path)

    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Extract individual components
    year = str(current_datetime.year)
    month = str(current_datetime.month)
    day = str(current_datetime.day)
    hour = str(current_datetime.hour)
    minute = str(current_datetime.minute)

    filename = (full_path + '/' + file_name + '_' + day + '_' + month + '_' + hour + '_' + minute + '_' +
                str(sequence) + '.' + settings['extension'])

    # In CoppeliaSim images are left to right (x-axis), and bottom to top (y-axis)
    # (consistent with the axes of vision sensors, pointing Z outwards, Y up)
    # and color format is RGB triplets, whereas OpenCV uses BGR:
    img = cv.flip(cv.cvtColor(img, cv.COLOR_BGR2RGB), 0)

    cv.imwrite(filename, img)


def generate_spiral_points(box_side_gsp, step):
    x, y = 0, 0
    points = [[x, y]]

    directions = [(step, 0), (0, step), (-step, 0), (0, -step)]  # Right, Up, Left, Down
    direction_index = 0
    steps = 1
    step_count = 0

    while True:
        dx, dy = directions[direction_index % 4]
        x += dx
        y += dy
        points.append([x, y])
        step_count += 1

        if step_count == steps:
            direction_index += 1
            step_count = 0
            if direction_index % 2 == 0:
                steps += 1

        if not abs(x) < box_side_gsp / 2 or not abs(y) < box_side_gsp / 2:
            points.pop()
            break

    return points


def get_single_target_spiral_trajectory(centroid_points_gstst: ndarray, radius_gstst: float, parts_gstst: float):
    print('Generating esprial trajectory over one target')
    step = radius_gstst / parts_gstst
    points_gstst = generate_spiral_points(radius_gstst, step)

    points_gstst = np.array([np.hstack((np.array(p), 0)) for p in points_gstst])
    directions_gstst = np.zeros([1, 3])
    for p in points_gstst[1:]:
        directions_gstst = np.row_stack((directions_gstst, euler_angles_from_normal(-p)))
    points_gstst = points_gstst + centroid_points_gstst

    return points_gstst, directions_gstst


def get_spiral_trajectories(centroids_gst: dict, radius_gst: dict, parts_gst: int) -> tuple[
    ndarray[Any, dtype[Any]], dict[Any, ndarray[Any, dtype[bool_]]], dict[Any, Any], int]:
    print('Generating spiral trajectories')
    # plotter_gst = pv.Plotter()
    route_gst = np.zeros([1, 6])
    route_by_target_gst = {}
    spiral_target_distance_gst = {}
    total_distance_gst = 0
    for target_gst, centroid_gst in centroids_gst.items():
        radius_box_gst = 2 * radius_gst[target_gst]
        centroid_gst[2] = centroid_gst[2] + scale_to_height_spiral * centroid_gst[2]
        spiral_point, spiral_direction = get_single_target_spiral_trajectory(centroid_gst, radius_box_gst, parts_gst)
        spiral_target_distance_gst[target_gst] = 0
        for count_point in range(spiral_point.shape[0] - 1):
            spiral_target_distance_gst[target_gst] += np.linalg.norm(spiral_point[count_point, :3] - spiral_point[count_point + 1, :3])
        route_by_target_gst[target_gst] = np.column_stack((spiral_point, spiral_direction))
        route_gst = np.row_stack((route_gst, route_by_target_gst[target_gst]))
        total_distance_gst += spiral_target_distance_gst[target_gst]
    route_gst = np.row_stack((route_gst, np.zeros([1, 6])))
    # Create lines connecting each pair of adjacent points
    # lines = []
    # for i in range(route_gst.shape[0] - 1):
    #     lines.append([2, i, i + 1])  # Each line is represented as [num_points_in_line, point1_index, point2_index]
    # lines = np.array(lines, dtype=np.int_)

    # plotter_gst.add_mesh(pv.PolyData(route_gst[:, :3]))
    # plotter_gst.add_mesh(pv.PolyData(route_gst[:, :3], lines=lines))
    #
    # plotter_gst.show_grid()
    # plotter_gst.show()
    return route_gst, route_by_target_gst, spiral_target_distance_gst, total_distance_gst


def convex_hull(experiment: int):
    print('Starting convex hull ...')
    global settings
    settings = load_variables()

    with open(f"./build.obj", "rb") as file:
        positions = pickle.load(file)
        target_hull = pickle.load(file)
        centroid_points = pickle.load(file)
        radius = pickle.load(file)

    targets_points_of_view, points_of_view_contribution, conversion_table = draw_cylinders_hemispheres(
        centroid_points,
        radius,
        positions)
    S, subgroup_size = subgroup_formation(target_hull, points_of_view_contribution, targets_points_of_view, positions)
    edge_weight_matrix = compute_edge_weight_matrix(S, targets_points_of_view)
    name_cops_file = settings['COPS problem'] + str(experiment)
    write_problem_file(settings['COPS dataset'],
                        name_cops_file,
                        edge_weight_matrix,
                        len(settings['object names']),
                        S,
                        subgroup_size)
    execute_script(name_cops_file)


    with open(settings['save path'] + f'variables/convex_hull_{experiment}.var', 'wb') as file:
        pickle.dump(S, file)  
        pickle.dump(targets_points_of_view, file)  
        pickle.dump(centroid_points, file)  
        pickle.dump(radius, file)
    print('Ending convex hull')

def view_point(copp: CoppeliaInterface, experiment: int):

    with open(settings['save path'] + f'variables/convex_hull_{experiment}.var', 'rb') as file:
        S = pickle.load(file)
        targets_points_of_view = pickle.load(file)
        centroid_points = pickle.load(file)
        radius = pickle.load(file)
    
    main_route, travelled_distance_main, route_by_group = read_route_csv_file(
        settings['COPS result'] + settings['COPS problem'] + str(experiment) + '.csv', S, targets_points_of_view)
    parts_to_spiral = 10 #np.fix(main_route.shape[0]/40)

    spiral_routes, spiral_route_by_target, spiral_target_distance, travelled_spiral_distance = (
        get_spiral_trajectories(centroid_points, radius, parts_to_spiral))
    #
    #
    copp.handles[settings['vision sensor names']] = copp.sim.getObject(settings['vision sensor names'])
    # vision_handle = copp.handles[settings['vision sensor names']]
    # filename = settings['filename']

    # Get the current date and time
    current_datetime = datetime.datetime.now()
    month = str(current_datetime.month)
    day = str(current_datetime.day)
    hour = str(current_datetime.hour)
    minute = str(current_datetime.minute)
    
    # directory_name = settings['directory name'] + f'_exp_{experiment}_{day}_{month}_{hour}_{minute}'
    # spiral_directory_name = settings['directory name'] + f'_spriral_exp_{experiment}_{day}_{month}_{hour}_{minute}'
    # quadcopter_control_direct_points(copp.sim, copp.client, vision_handle, main_route, filename, directory_name)
    #
    # copp.sim.setObjectOrientation(vision_handle, [-np.pi, np.pi / 3, -np.pi / 2], copp.sim.handle_parent)
    #
    # quadcopter_control_direct_points(copp.sim,
    #                                  copp.client,
    #                                  vision_handle,
    #                                  spiral_routes,
    #                                  'spiral_route',
    #                                  spiral_directory_name)

    spiral_route_key = spiral_route_by_target.keys()
    for route, spiral_key, count_group in zip(route_by_group, spiral_route_key, range(len(route_by_group))):
    # for route, count_group in zip(route_by_group, range(len(route_by_group))):
        filename = settings['filename']
        vision_handle = copp.handles[settings['vision sensor names']]

        group_name = f'_exp_{experiment}_group_{count_group}_{day}_{month}_{hour}_{minute}'
        directory_name = settings['directory name'] + group_name

        copp.sim.setObjectOrientation(vision_handle, [0, np.pi / 2, np.pi / 2], copp.sim.handle_parent)
        
        quadcopter_control_direct_points(copp.sim, copp.client,  vision_handle, route, filename, directory_name)

        copp.sim.setObjectOrientation(vision_handle, [-np.pi, np.pi / 3, -np.pi / 2], copp.sim.handle_parent)

        spiral_route = spiral_route_by_target[spiral_key]
        spiral_group_name = f'_spriral_exp_{experiment}_group_{count_group}_{day}_{month}_{hour}_{minute}'
        spiral_directory_name = settings['directory name'] + spiral_group_name
        #
        quadcopter_control_direct_points(copp.sim,
                                         copp.client,
                                         vision_handle,
                                         spiral_route,
                                         'spiral_route',
                                         spiral_directory_name)

    with open(settings['save path'] + f'variables/view_point_{experiment}.var', 'wb') as file:
        pickle.dump(travelled_distance_main, file)
        pickle.dump(travelled_spiral_distance, file)
        pickle.dump(spiral_route_by_target, file)
        pickle.dump(route_by_group, file)
        pickle.dump(spiral_target_distance, file)
        pickle.dump(day, file)  
        pickle.dump(month, file)  
        pickle.dump(hour, file)  
        pickle.dump(minute, file)

def remove_unused_files(workspace_folder):
    """
    Remove the stereo folder and the images folder in the dense folder
    """
    dense_folder = os.path.join(workspace_folder, 'dense')
    for folder in os.listdir(dense_folder):
        stereo_folder = os.path.join(folder, 'stereo')
        images_folder = os.path.join(folder, 'images')

        stereo_folder_path = os.path.join(dense_folder, stereo_folder)
        images_folder_path = os.path.join(dense_folder, images_folder)

        shutil.rmtree(stereo_folder_path)
        shutil.rmtree(images_folder_path)


def point_cloud(experiment: int) -> None:
    with open(settings['save path'] + f'variables/view_point_{experiment}.var', 'rb') as f:
        travelled_distance_main = pickle.load(f)
        travelled_spiral_distance = pickle.load(f)
        spiral_route_by_target = pickle.load(f)
        route_by_group = pickle.load(f)
        spiral_target_distance = pickle.load(f)
        day = pickle.load(f)
        month = pickle.load(f)
        hour = pickle.load(f)
        minute = pickle.load(f)


    # Get the current date and time
    workspace_folder = os.path.join(settings['workspace folder'], f'exp_{experiment}_{day}_{month}_{hour}_{minute}')
    spiral_workspace_folder = os.path.join(settings['workspace folder'],
                                           f'spiral_exp_{experiment}_{day}_{month}_{hour}_{minute}')
    
    # directory_name = settings['directory name'] + f'_exp_{experiment}_{day}_{month}_{hour}_{minute}'
    spiral_directory_name = settings['directory name'] + f'_spriral_exp_{experiment}_{day}_{month}_{hour}_{minute}'

    colmap_folder = settings['colmap folder']

    # remove folder if exist
    if os.path.exists(workspace_folder):
        shutil.rmtree(workspace_folder)

    # Create the directory
    os.makedirs(workspace_folder)
    
    with open(workspace_folder + '/distance.txt', 'w') as distance_file:
        distance_file.write(str(travelled_distance_main))
    
    # images_folder = str(os.path.join(settings['path'], directory_name))
    # run_colmap_program(colmap_folder, workspace_folder, images_folder)
    # statistics_colmap(colmap_folder, workspace_folder)

    # remove folder if exist
    if os.path.exists(spiral_workspace_folder):
        shutil.rmtree(spiral_workspace_folder)
    
    # Create the directory
    # os.makedirs(spiral_workspace_folder)
    
    # with open(spiral_workspace_folder + '/distance.txt', 'w') as distance_file:
    #     distance_file.write(str(travelled_spiral_distance))
    
    # spiral_images_folder = str(os.path.join(settings['path'], spiral_directory_name))
    # run_colmap_program(colmap_folder, spiral_workspace_folder, spiral_images_folder)
    # statistics_colmap(colmap_folder, spiral_workspace_folder)

    MNRE_array = np.empty(0)
    spriral_route_key = spiral_route_by_target.keys()
    for route, spiral_key, count_group in zip(route_by_group, spriral_route_key, range(len(route_by_group))):
    # for route, count_group in zip(route_by_group, range(len(route_by_group))):
        image_directory_name = (settings['directory name'] +
                          f'_exp_{experiment}_group_{count_group}_{day}_{month}_{hour}_{minute}')

        workspace_folder = os.path.join(settings['workspace folder'],
                                        f'exp_{experiment}_{day}_{month}_{hour}_{minute}_group_{count_group}')

        # remove folder if exist
        if os.path.exists(workspace_folder):
            shutil.rmtree(workspace_folder)

        # Create the directory
        os.makedirs(workspace_folder)

        travelled_distance_main = 0
        for i in range(route.shape[0]):
            for j in range(i + 1, route.shape[0]):
                travelled_distance_main += np.linalg.norm(route[i, :3] - route[j, :3])

        with open(workspace_folder + '/distance.txt', 'w') as distance_file:
            distance_file.write(str(travelled_distance_main))

        images_folder = str(os.path.join(settings['path'], image_directory_name))
        run_colmap_program(colmap_folder, workspace_folder, images_folder)
        MNRE_array = statistics_colmap(colmap_folder, workspace_folder, MNRE_array)
        remove_unused_files(workspace_folder)

        spiral_workspace_folder = os.path.join(settings['workspace folder'],
                                        f'spiral_exp_{experiment}_{day}_{month}_{hour}_{minute}_group_{count_group}')

        spiral_directory_name = settings['directory name'] + f'_spriral_exp_{experiment}_group_{count_group}_{day}_{month}_{hour}_{minute}'

        # remove folder if exist
        if os.path.exists(spiral_workspace_folder):
            shutil.rmtree(spiral_workspace_folder)

        # Create the directory
        os.makedirs(spiral_workspace_folder)

        with open(spiral_workspace_folder + '/distance.txt', 'w') as distance_file:
            distance_file.write(str(spiral_target_distance[spiral_key]))

        spiral_images_folder = str(os.path.join(settings['path'], spiral_directory_name))
        run_colmap_program(colmap_folder, spiral_workspace_folder, spiral_images_folder)
        statistics_colmap(colmap_folder, spiral_workspace_folder)
        remove_unused_files(spiral_workspace_folder)


def mesh_analysis(experiment: int):
    print('Initiating mesh analysis')
    with open(settings['save path'] + f'variables/view_point_{experiment}.var', 'rb') as f:
        spiral_directory_name = pickle.load(f)
        directory_name = pickle.load(f)
        day = pickle.load(f)
        month = pickle.load(f)
        hour = pickle.load(f)
        minute = pickle.load(f)

    # Get the current date and time
    workspace_folder = os.path.join(settings['workspace folder'], f'exp_{experiment}_{day}_{month}_{hour}_{minute}')
    spiral_workspace_folder = os.path.join(settings['workspace folder'], f'spiral_exp_{experiment}_{day}_{month}_{hour}_{minute}')

    workspace_folder = os.path.normpath(workspace_folder)
    dense_folder_ma = os.path.join(workspace_folder, 'dense')
    if os.path.exists(dense_folder_ma):
        for work_dir_structure_ma in os.walk(dense_folder_ma):
            for work_dir_ma in work_dir_structure_ma[1]:
                mesh_file = os.path.join(work_dir_ma, 'meshed-poisson.ply')
                if not os.path.exists(mesh_file):
                    continue
                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(mesh_file)
                ms.show_polyscope()


def update_current_experiment(value_stage: float) -> None:
    with open(settings['save path'] + '.progress', 'wb') as file:
        pickle.dump(value_stage, file)


def execute_experiment() -> None:
    # Create the directory
    os.makedirs(settings['save path'] + 'variables/', exist_ok=True)

    with open(settings['save path'] + '.progress', 'rb') as f:
        last_expe = pickle.load(f)

    try:
        if len(sys.argv) < 2:

            if last_expe != 0:
                next_experiment = int(last_expe)
                if np.isclose(last_expe - next_experiment, 0.1):
                    copp = CoppeliaInterface(settings)

                    view_point(copp, next_experiment)
                    update_current_experiment(next_experiment + 0.2)

                    copp.sim.stopSimulation()
                    last_expe += 0.1

                if np.isclose(last_expe - next_experiment, 0.2):
                    point_cloud(next_experiment)
                    update_current_experiment(next_experiment + 1)

                    last_expe = next_experiment + 1

            for experiment in range(settings['number of trials']):
                if experiment < last_expe:
                    continue

                proc = multiprocessing.Process(target=convex_hull, args=(experiment,))
                proc.start()
                proc.join()
                update_current_experiment(float(experiment + 0.1))


                copp = CoppeliaInterface(settings)
                view_point(copp, experiment)

                update_current_experiment(float(experiment + 0.2))
                copp.sim.stopSimulation()
                del copp

                point_cloud(experiment)
                update_current_experiment(float(experiment + 1))

                mesh_analysis(experiment)
                update_current_experiment(float(experiment + 1))

            os.remove(settings['save path'] + '.progress')
            return

        if sys.argv[1] == 'convex_hull':
            for experiment in range(settings['number of trials']):
                if experiment < last_expe:
                    continue

                proc = multiprocessing.Process(target=convex_hull, args=(experiment,))
                proc.start()
                proc.join()
                update_current_experiment(float(experiment + 1))

            os.remove(settings['save path'] + '.progress')
            return

        if sys.argv[1] == 'view_point':
            copp = CoppeliaInterface(settings)
            for experiment in range(settings['number of trials']):
                if experiment < last_expe:
                    continue

                view_point(copp, experiment)
                update_current_experiment(float(experiment + 1))

            os.remove(settings['save path'] + '.progress')
            copp.sim.stopSimulation()
            return

        if sys.argv[1] == 'point_cloud':
            for experiment in range(settings['number of trials']):
                if experiment < last_expe:
                    continue

                point_cloud(experiment)
                update_current_experiment(float(experiment + 1))

            os.remove(settings['save path'] + '.progress')
            return

        if sys.argv[1] == 'mesh_analysis':
            for experiment in range(settings['number of trials']):
                if experiment < last_expe:
                    continue

                mesh_analysis(experiment)
                update_current_experiment(float(experiment + 1))

            os.remove(settings['save path'] + '.progress')
            return

    except RuntimeError as e:
        print("An error occurred:", e)

def load_variables():
    settings = parse_settings_file('config.yaml')
    global CA_max
    CA_max = float(settings['CA_max'])
    global max_route_radius
    max_route_radius = float(settings['max route radius'])
    global points_per_sphere
    points_per_sphere = float(settings['points per sphere'])
    global height_proportion
    height_proportion = float(settings['height proportion'])
    global max_visits
    max_visits = int(settings['max visits'])
    global max_iter
    max_iter = int(settings['max iter'])
    global T_max
    T_max = float(settings['T_max'])
    global n_resolution
    n_resolution = int(settings['n resolution'])
    global points_per_unit
    points_per_unit = float(settings['points per unit'])

    save_path = settings['save path']

    path = settings['path']
    COPS_dataset = settings['COPS dataset']
    COPS_result = settings['COPS result']
    workspace_folder = settings['workspace folder']

    settings['path'] = os.path.join(save_path, path)
    settings['COPS dataset'] = os.path.join(save_path, COPS_dataset)
    settings['COPS result'] = os.path.join(save_path, COPS_result)
    settings['workspace folder'] = os.path.join(save_path, workspace_folder)

    return settings


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    settings = load_variables()

    save_path = settings['save path']

    os.makedirs(save_path, exist_ok=True)

    # check if file not exits
    if not os.path.isfile(settings['save path'] + '.progress'):
        with open(settings['save path'] + '.progress', 'wb') as file:
            pickle.dump(0.0, file)

    execute_experiment()
