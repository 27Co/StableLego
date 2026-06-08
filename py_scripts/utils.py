import json

import networkx as nx
import numpy as np


def load_json(fname):
    f = open(fname)
    content = json.load(f)
    f.close()
    return content


def write_json(graph, output_dir):
    json_graph = json.dumps(graph, indent=4)
    with open(output_dir, "w") as outfile:
        outfile.write(json_graph)
    outfile.close()


def brick_equal(b1, b2):
    return b1["brick_id"] == b2["brick_id"] and b1["x"] == b2["x"] and b1["y"] == b2["y"] and b1["z"] == b2["z"] and b1[
        "ori"] == b2["ori"]


def constr_pair_exists(constr_set, name1, name2):
    candidate1 = name1 + "-" + name2
    candidate2 = name2 + "-" + name1
    return candidate1 in constr_set or candidate2 in constr_set


def constr_exists(constr_set, name):
    return name in constr_set


def construct_world_grid(lego, world_dimension, brick_library):
    world_grid = np.zeros(world_dimension)
    for key in lego.keys():
        brick = lego[key]
        brick_id = str(brick["brick_id"])
        if (brick["ori"] == 0):
            h = brick_library[brick_id]["height"]
            w = brick_library[brick_id]["width"]
        else:
            w = brick_library[brick_id]["height"]
            h = brick_library[brick_id]["width"]
        brick_x = brick["x"]
        brick_y = brick["y"]
        brick_z = brick["z"]
        for i in range(brick_x, brick_x + h):
            for j in range(brick_y, brick_y + w):
                world_grid[i, j, brick_z] = 1
    return world_grid


def construct_graph_voxel(task_graph, lego_lib, world_dim=[48, 48, 48]):  # Task graph z starts from 0
    voxel = np.zeros((world_dim[0], world_dim[1], world_dim[2] + 1), dtype=np.uint16)
    voxel_key = np.zeros((world_dim[0], world_dim[1], world_dim[2] + 1), dtype=np.uint16)
    lego_graph = nx.Graph()
    lego_graph.add_node(0)
    min_z = 1000

    # Nodes
    for k in task_graph.keys():
        node = task_graph[k]
        brick_id, x, y, z, ori = node["brick_id"], node["x"], node["y"], node["z"], node["ori"]
        h, w = lego_lib[str(brick_id)]["height"], lego_lib[str(brick_id)]["width"]
        if (ori == 1):
            h, w = w, h
        if (np.any(voxel[x:x + h, y:y + w, z])):
            raise Exception("Invalid lego structure! Overlapping bricks!")
        voxel[x:x + h, y:y + w, z] = int(brick_id)
        voxel_key[x:x + h, y:y + w, z] = int(k)
        node_key = gen_key_from_brick(node)
        lego_graph.add_node(node_key, x=x, y=y, z=z, brick_id=brick_id, ori=ori)
        min_z = min(min_z, z)
    if (min_z != 0 and len(task_graph) > 0):
        raise Exception("Invalid lego structure! z does not start from 0!")

    # Edges
    for k in task_graph.keys():
        node = task_graph[k]
        brick_id, x, y, z, ori = node["brick_id"], node["x"], node["y"], node["z"], node["ori"]
        h, w = lego_lib[str(brick_id)]["height"], lego_lib[str(brick_id)]["width"]
        cur_key = gen_key_from_brick(node)
        if (ori == 1):
            h, w = w, h
        for i in range(x, x + h):
            for j in range(y, y + w):
                if (i - 1 >= 0 and voxel_key[i - 1, j, z] != 0 and voxel_key[i - 1, j, z] != voxel_key[i, j, z]):
                    neighbor_key = voxel_key[i - 1, j, z]
                    neighbor = task_graph[str(neighbor_key)]
                    lego_graph.add_edge(gen_key_from_brick(neighbor), cur_key)
                if (i + 1 < world_dim[0] and voxel_key[i + 1, j, z] != 0 and voxel_key[i + 1, j, z] != voxel_key[
                    i, j, z]):
                    neighbor_key = voxel_key[i + 1, j, z]
                    neighbor = task_graph[str(neighbor_key)]
                    lego_graph.add_edge(gen_key_from_brick(neighbor), cur_key)
                if (j - 1 >= 0 and voxel_key[i, j - 1, z] != 0 and voxel_key[i, j - 1, z] != voxel_key[i, j, z]):
                    neighbor_key = voxel_key[i, j - 1, z]
                    neighbor = task_graph[str(neighbor_key)]
                    lego_graph.add_edge(gen_key_from_brick(neighbor), cur_key)
                if (j + 1 < world_dim[1] and voxel_key[i, j + 1, z] != 0 and voxel_key[i, j + 1, z] != voxel_key[
                    i, j, z]):
                    neighbor_key = voxel_key[i, j + 1, z]
                    neighbor = task_graph[str(neighbor_key)]
                    lego_graph.add_edge(gen_key_from_brick(neighbor), cur_key)
                if (z == 0):
                    if (voxel[i, j, z + 1] != 0):
                        neighbor_key = voxel_key[i, j, z + 1]
                        neighbor = task_graph[str(neighbor_key)]
                        lego_graph.add_edge(gen_key_from_brick(neighbor), cur_key)
                    lego_graph.add_edge(str(0), cur_key)
                else:
                    for dz in [-1, 1]:
                        if (voxel[i, j, z + dz] != 0):
                            neighbor_key = voxel_key[i, j, z + dz]
                            neighbor = task_graph[str(neighbor_key)]
                            lego_graph.add_edge(gen_key_from_brick(neighbor), cur_key)
    return voxel, voxel_key, lego_graph


def is_four_pt_connection(brick_id_int, brick_lib):
    brick_id_int = int(brick_id_int)
    h, w = brick_lib[str(brick_id_int)]["height"], brick_lib[str(brick_id_int)]["width"]
    if (min(w, h) < 2):
        four_pt_connections = 1
    else:
        four_pt_connections = 0
    return four_pt_connections


def gen_key(x, y, z):
    return "X: " + str(x) + ", Y: " + str(y) + ", Z: " + str(z)


def gen_key_voxel(x, y, z):
    return str(x) + "_" + str(y) + "_" + str(z)


def gen_key_from_brick(brick):
    return str(brick["brick_id"]) + "_" + str(brick["x"]) + "_" + str(brick["y"]) + "_" + str(brick["z"]) + "_" + str(
        brick["ori"])


def out_boundary(pt, brick_x, brick_y, h, w):
    x = pt[0]
    y = pt[1]
    if (x < brick_x or x >= brick_x + h or y < brick_y or y >= brick_y + w):
        return True
    return False

