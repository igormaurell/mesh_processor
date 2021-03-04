import numpy as np
from plyfile import PlyData, PlyElement

from mpl_toolkits import mplot3d
from matplotlib import pyplot, colors

from random import randint

import argparse

from tqdm import tqdm

import sys
sys.setrecursionlimit(10000)

#sphere parameters
center = (0,0,0)
radius = 56.4078

# sphere = {
#     'type': 'sphere',
#     'coefficients': None,
#     'face_indices': None,
#     'location': (0.0, 0.0, 0.0),
#     'radius':  56.4078,
#     'vert_indices': None,
#     'vert_parameters': None,
#     'x_axis': (0.0, 0.0, 0.0),
#     'y_axis': None,
#     'z_axis': (-1.0, 0.0, 0.0), 
# }

sphere1 = {
    'type': 'sphere',
    'coefficients': None,
    'face_indices': None,
    'location': (0.0, 0.0, 0.0),
    'radius':  0.02,
    'vert_indices': None,
    'vert_parameters': None,
    'x_axis': (0.0, 0.0, 0.0),
    'y_axis': None,
    'z_axis': (-1.0, 0.0, 0.0), 
}

sphere2 = {
    'type': 'sphere',
    'coefficients': None,
    'face_indices': None,
    'location': (-0.038, 0.0, 0.0),
    'radius':  0.027,
    'vert_indices': None,
    'vert_parameters': None,
    'x_axis': (0.0, 0.0, 0.0),
    'y_axis': None,
    'z_axis': (-1.0, 0.0, 0.0), 
}

features = [sphere1, sphere2]

mesh_vertexes = None
mesh_faces = None

vertex_graph = None

feature_vertexes_distance = None

def distance_points(p1, p2):
    p1a = np.array(list(p1)[0:3])
    p2a = np.array(list(p2)[0:3])
    return np.linalg.norm(p2a - p1a, ord=2)

def distance_point_sphere(point, surface):
    center = surface['location']
    radius = surface['radius']
    d = distance_points(point, center) - radius
    return d

def distance_point_plane(point, surface):
    x0,y0,z0 = list(point)[0:3]
    a,b,c,d = surface['coefficients']
    normal = np.array([a, b, c])
    return abs(a*x0 + b*y0 + c*z0 + d)/np.linalg.norm(normal, ord = 2)


def distance_point_torus(point, surface):
    pass

def distance_point_cylinder(point, surface):
    pass

def distance_point_cone(point, surface):
    pass

def distance_point_line(point, curve):
    p1 = np.array(list(curve['location'])[0:3])
    v = np.array(list(curve['direction'])[0:3])
    p2 = np.array(list(point)[0:3])
    cp = np.cross(v,(p2-p1))
    return np.linalg.norm(cp, ord=2)/np.linalg.norm(v, ord=2)

def distance_point_circle(point, curve):
    pass


def vertex_processor(vertex):
    point = mesh_vertexes[vertex]
    d = distance_point_sphere(point)

def face_processor(face):
    for vertex in face:
        vertex_processor(vertex)

def mount_graph():
    print('Mounting vertex adjacency graph...')
    for face in tqdm(mesh_faces):
        face = face[0]

        vertex_graph[face[0]].append(face[1])
        vertex_graph[face[0]].append(face[2])

        vertex_graph[face[1]].append(face[0])
        vertex_graph[face[1]].append(face[2])

        vertex_graph[face[2]].append(face[0])
        vertex_graph[face[2]].append(face[1])
    print(len(vertex_graph))
    print('Done.')

def feature_vertex_matching():
    print('Matching features and vertexes...')
    for i, feature in tqdm(enumerate(features)):
        #isso pode ser trocado por dicionario de funcoes
        if feature['type'] == 'sphere':
            distance_function = distance_point_sphere
        elif feature['type'] == 'plane':
            distance_function = distance_point_plane
        count = 0
        #isso vai ser trocado por algo mais inteligante (kd-tree, octree ou quadtree)
        for j, vertex in enumerate(mesh_vertexes):
            ds = distance_function(vertex, feature)
            if ds < 0.001:
                count += 1
                do = distance_points(vertex, feature['location'])
                #index, distance to surface, distance to origin
                feature_vertexes_distance[i][j] = (ds, do)
    print(len(feature_vertexes_distance[0]))
    print(len(feature_vertexes_distance[1]))
    print('Done.')

def dfsUtil(vertexes_dict, v, ds, visited, cc, ds_acc):
    if visited[v] == False:
        cc.append(v)
        ds_acc = (ds_acc[0] + ds[0], ds_acc[1] + ds[1])
        visited[v] = True
        adjacency = vertex_graph[v]
        for a in adjacency:
            if visited[a] == False and a in vertexes_dict.keys():
                cc, ds_acc = dfsUtil(vertexes_dict, a, vertexes_dict[a], visited, cc, ds_acc)
   
    return cc, ds_acc

def found_best_connected_component(feature_index, vertexes_dict):
    print(feature_index, len(vertexes_dict))
    visited = [False] * len(vertex_graph)
    components = []
    distances = []
    for v, ds in vertexes_dict.items():
        if visited[v] == False:
            component, distances_accumulator = dfsUtil(vertexes_dict, v, ds, visited, [], (0.0, 0.0))
            components.append(component)
            ds, do = distances_accumulator
            distances.append((ds/len(component), do/len(component)))
    
    #verifica qual a melhor componente
    components.sort(key=len)
    return components[-1]


def found_features_connected_component():
    print('Founding features connected component...')
    components = []
    for i, vertexes_dict in tqdm(enumerate(feature_vertexes_distance)):
        component = found_best_connected_component(i, vertexes_dict)
        components.append(component)
    return components
    print('Done.')

def compute_features_face_indices(features_vertex_indices):
    print('Computing mesh indices...')
    features_faces_indices = [[] for i in range(0, len(features_vertex_indices))]
    for i, face in tqdm(enumerate(mesh_faces)):
        face = face[0]
        face_features = []
        for vertex in face:
            face_features.append([])
            for j, indices in enumerate(features_vertex_indices):
                if vertex in indices:
                    face_features[-1].append(j)
        
        acc = [0]*len(features_vertex_indices)
        for vertex_features in face_features:
            for feature in vertex_features:
                acc[feature] += 1
        
        face_feature = acc.index(max(acc))
        features_faces_indices[face_feature].append(i)
    return features_faces_indices
    print("Done.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mesh Processor.')
    parser.add_argument('input', type=str, help='input file in .stl or .ply.')
    # parser.add_argument('output', type=str, help='output folder.')
    # parser.add_argument('--rescale_factor', type = float, default = 1, help='first rescale applied to the point cloud, used to change the measurement unit.')
    # parser.add_argument('--noise_limit', type = float, default=0.01, help='limit noise applied to the point cloud.')
    # parser.add_argument('--centralize', type = bool, default=True, help='bool to centralize or not.')
    # parser.add_argument('--align', type = bool, default=True, help='bool to canonical alignment or not.')
    # parser.add_argument('--cube_rescale_factor', type = float, default=1, help='argument to make the point cloud lie in a unit cube, the factor multiplies all the dimensions of result cube.')
    args = vars(parser.parse_args())

    inputname = args['input']
    # outputname = args['output']
    # rescale_factor = args['rescale_factor']
    # noise_limit = args['noise_limit']
    # is_centralize = args['centralize']
    # is_align = args['align']
    # cube_rescale_factor = args['cube_rescale_factor']

    if inputname[inputname.index('.'):] == '.stl':
        print('Reading .stl file...')
        mesh_file = mesh.Mesh.from_file(inputname)
        mesh_array = mesh_file.vectors
        exit()
    elif inputname[inputname.index('.'):] == '.ply':
        print('Reading .ply file...')
        plydata = PlyData.read(inputname)
        mesh_vertexes = plydata['vertex'].data
        vertexes = plydata['vertex'].count
        mesh_faces = plydata['face'].data
        faces = plydata['face'].count
    else:
        print('{} file type can not be processed.'.format(inputname[inputname.index('.'):]))
        exit()

    print('{} faces and {} vertexes.'.format(faces, vertexes))
    print('Done.')

    vertex_graph = [[] for i in range(0,vertexes)]
    mount_graph()

    feature_vertexes_distance = [{} for i in range(0,len(features))]
    feature_vertex_matching()

    features_vertex_indices = found_features_connected_component()


    features_face_indices = compute_features_face_indices(features_vertex_indices)


    features_vectors = []

    for ffi in features_face_indices:
        face_array = []
        for i, face in enumerate(ffi):
            face_vertex = mesh_faces[face]
            #print(i)
            vertex_array = []
            for vertex in face_vertex[0]:
                vertex_vector = list(mesh_vertexes[vertex])
                #print(vertex_vector)
                vertex_array.append(vertex_vector[0:3])
            face_array.append(vertex_array)
        features_vectors.append(np.array(face_array))


    # Create a new plot
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)

    colors_list = dict(colors.BASE_COLORS, **colors.CSS4_COLORS)
    scale = np.empty((0), dtype=float)

    for fv in features_vectors:
        print(fv.shape)
        collection = mplot3d.art3d.Poly3DCollection(fv)
        color = list(colors_list.values())[randint(0, len(colors_list.values()))]
        collection.set_facecolor(color)
        axes.add_collection3d(collection)
        scale = np.append(scale, fv.flatten())

    # Auto scale to the mesh size
    axes.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    pyplot.show()

    
    