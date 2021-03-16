import numpy as np
from plyfile import PlyData, PlyElement

from mpl_toolkits import mplot3d
from matplotlib import pyplot, colors

from random import randint

from math import sqrt

import argparse

from tqdm import tqdm

import pywavefront

from statistics import quantiles

import yaml

import sys
sys.setrecursionlimit(10000)

# #sphere parameters
# center = (0,0,0)
# radius = 56.4078

# # sphere = {
# #     'type': 'sphere',
# #     'coefficients': None,
# #     'face_indices': None,
# #     'location': (0.0, 0.0, 0.0),
# #     'radius':  56.4078,
# #     'vert_indices': None,
# #     'vert_parameters': None,
# #     'x_axis': (0.0, 0.0, 0.0),
# #     'y_axis': None,
# #     'z_axis': (-1.0, 0.0, 0.0), 
# # }

# sphere1 = {
#     'type': 'sphere',
#     'coefficients': None,
#     'face_indices': None,
#     'location': (0.0, 0.0, 0.0),
#     'radius':  0.02,
#     'vert_indices': None,
#     'vert_parameters': None,
#     'x_axis': (0.0, 0.0, 0.0),
#     'y_axis': None,
#     'z_axis': (-1.0, 0.0, 0.0), 
# }

# sphere2 = {
#     'type': 'sphere',
#     'coefficients': None,
#     'face_indices': None,
#     'location': (-0.038, 0.0, 0.0),
#     'radius':  0.027,
#     'vert_indices': None,
#     'vert_parameters': None,
#     'x_axis': (0.0, 0.0, 0.0),
#     'y_axis': None,
#     'z_axis': (-1.0, 0.0, 0.0), 
# }

# features = [sphere1, sphere2]

features = None

mesh_vertexes = None
mesh_faces = None

vertex_graph = None

feature_vertexes_distance = None

distance_threshold = 0

def load_features(dir):
    with open(dir) as f:
        features = yaml.load(f, Loader=yaml.FullLoader)
    return features

def distance_points(A, B):
    AB = B - A
    return np.linalg.norm(AB, ord=2)


def distance_point_line(point, curve):
    A = np.array(list(curve['location'])[0:3])
    v = np.array(list(curve['direction'])[0:3])
    P = np.array(list(point)[0:3])
    #AP vector
    AP = P - A
    #equation to calculate distance between point and line using a direction vector
    return np.linalg.norm(np.cross(v, AP), ord=2)/np.linalg.norm(v, ord=2)

def distance_point_circle(point, curve):
    A = np.array(list(curve['location'])[0:3])
    n = np.array(list(curve['z_axis'])[0:3])
    P = np.array(list(point)[0:3])
    radius = curve['radius']

    AP = P - A
    #orthogonal distance between point and circle plane
    dist_point_plane = np.dot(AP, n)/np.linalg.norm(n, ord=2)

    #projection P in the circle plane and calculating the distance to the center
    P_p = P - dist_point_plane*n/np.linalg.norm(n, ord=2)
    dist_pointproj_center = np.linalg.norm(P_p - A, ord=2)
    #if point is outside the circle arc, the distance to the curve is used 
    a = dist_pointproj_center - radius
    b = np.linalg.norm(P - P_p, ord=2)
    return sqrt(a**2 + b**2)


def distance_point_sphere(point, surface):
    A = np.array(list(curve['location'])[0:3])
    P = np.array(list(point)[0:3])
    radius = surface['radius']
    #simple, distance from point to the center minus the sphere radius
    return abs(distance_points(P, A) - radius)


def distance_point_plane(point, surface):
    A = np.array(list(surface['location'])[0:3])
    n = np.array(list(surface['z_axis'])[0:3])
    P = np.array(list(point)[0:3])
    AP = P - A
    #orthogonal distance between point and plane
    return abs(np.dot(AP, n)/np.linalg.norm(n, ord=2))


def distance_point_torus(point, surface):
    A = np.array(list(surface['location'])[0:3])
    n = np.array(list(surface['z_axis'])[0:3])
    P = np.array(list(point)[0:3])
    max_radius = surface['max_radius']
    min_radius = surface['min_radius']
    radius = (max_radius - min_radius)/2

    AP = P - A
    #orthogonal distance to the torus plane 
    h = np.dot(AP, n)/np.linalg.norm(n, ord = 2)

    line = surface
    line['direction'] = surface['z_axis']
    #orthogonal distance to the revolution axis line 
    d = distance_point_line(point, line)

    #projecting the point in the torus plane
    P_p = P - h*n/np.linalg.norm(n, ord=2)
    #getting the direction vector, using center as origin, to the point projected
    v = (P_p - A)/np.linalg.norm((P_p - A), ord=2)
    #calculating the center of circle in the direction of the input point
    B = (min_radius + radius)*v + A

    return abs(distance_points(B, P) - radius)


def distance_point_cylinder(point, surface):
    radius = surface['radius']
    surface['direction'] = surface['z_axis']
    #simple distance from point to the revolution axis line minus radius
    return abs(distance_point_line(point, surface) - radius)

def distance_point_cone(point, surface):
    A = np.array(list(surface['location'])[0:3])
    v = np.array(list(surface['z_axis'])[0:3])
    B = np.array(list(surface['apex'])[0:3])
    P = np.array(list(point)[0:3])
    radius = surface['radius']

    #height of cone
    h = distance_points(A, B)

    AP = P - A
    #point projected in the revolution axis line
    P_p = A + np.dot(AP, v)/np.dot(v, v) * v

    #distance from center of base to point projected
    dist_AP = distance_points(P_p, A)
    #distance from apex to the point projected
    dist_BP = distance_points(P_p, B)

    #if point is below the center of base, return the distance to the circle base line
    if dist_BP > dist_AP and dist_BP >= h:
        AP = P - A
        signal_dist_point_plane = np.dot(AP, v)/np.linalg.norm(v, ord=2)

        #projection P in the circle plane and calculating the distance to the center
        P_p = P - signal_dist_point_plane*v/np.linalg.norm(v, ord=2)
        dist_pointproj_center = np.linalg.norm(P_p - A, ord=2)
        #if point is outside the circle arc, the distance to the curve is used 
        if dist_pointproj_center > radius:
            #not using distance_point_circle function to not repeat code
            a = dist_pointproj_center - radius
            b = np.linalg.norm(P - P_p, ord=2)
            return sqrt(a**2 + b**2)
        
        #if not, the orthogonal distance to the circle plane is used
        return abs(signal_dist_point_plane)
    #if point is above the apex, return the distance from point to apex
    elif dist_AP > dist_BP and dist_AP >= h:
        return distance_points(P, B)

    #if not, calculate the radius of the circle in this point height 
    r = radius*dist_BP/h

    #distance from point to the point projected in the revolution axis line minus the current radius
    return abs(distance_points(P, P_p) - r)

distance_functions = {
    'line': distance_point_line,
    'circle': distance_point_circle,
    'sphere': distance_point_sphere,
    'plane': distance_point_plane,
    'torus': distance_point_torus,
    'cylinder': distance_point_cylinder,
    'cone': distance_point_cone
}

infinity_geometries = ['line', 'plane', 'cylinder']

def mount_graph():
    print('Mounting vertex adjacency graph...')
    for face in tqdm(mesh_faces):
        vertex_graph[face[0]].append(face[1])
        vertex_graph[face[0]].append(face[2])

        vertex_graph[face[1]].append(face[0])
        vertex_graph[face[1]].append(face[2])

        vertex_graph[face[2]].append(face[0])
        vertex_graph[face[2]].append(face[1])
    print('Done.')

def feature_vertex_matching():
    print('Matching features and vertexes...')
    for i, feature in tqdm(enumerate(features)):
        #isso pode ser trocado por dicionario de funcoes
        feature_type = feature['type'].lower()
        if feature['type'].lower() not in distance_functions.keys():
            continue
        count = 0
        #isso vai ser trocado por algo mais inteligante (kd-tree, octree ou quadtree)
        for j, vertex in enumerate(mesh_vertexes):
            ds = distance_functions[feature['type'].lower()](vertex, feature)
            if ds < distance_threshold:
                count += 1
                do = distance_points(np.array(list(vertex)[0:3]), np.array(list(feature['location'])[0:3]))
                #index, distance to surface, distance to origin
                feature_vertexes_distance[i][j] = (ds, do)
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
    visited = [False] * len(vertex_graph)
    components = []
    lengths = []
    distances_surface = []
    distances_origin = []
    for v, ds in vertexes_dict.items():
        if visited[v] == False:
            component, distances_accumulator = dfsUtil(vertexes_dict, v, ds, visited, [], (0.0, 0.0))
            components.append(component)
            ds, do = distances_accumulator

            lengths.append(len(component))
            distances_surface.append(ds/len(component))
            distances_origin.append(do/len(component))
    
    #qual componente deve ser selecionada? depende do tipo de feature? usamos ds, do ou o tamanho da componente?
    if len(components) > 1:
        print('\n')
        print(feature_index, 'em processo de selecao.')
        print('Lengths:', lengths)
        print('DS:', distances_surface)
        print('DO:', distances_origin)
        lower_ql = quantiles(lengths)[0]
        print('Lower QL:', lower_ql)
        higher_qds = quantiles(distances_surface)[-1]
        print('Higher QDS:', higher_qds)

        mask = [0 for i in range(0, len(components))]
        for i, comp in enumerate(components):
            if lengths[i] <= lower_ql:
                mask[i]+= 1
            if distances_surface[i] >= higher_qds:
                mask[i]+= 1
        
        print('Mask:', mask)

        value_to_process = 0
        if mask.count(0) == 0:
            print('Todos sao outliers.')
            if mask.count(1) == 0:
                print('Pior caso')
                #worst case, but, all the components must be tested
                value_to_process = 2
            elif mask.count(1) == 1:
                print('Um se salvou')
                #exists one that is an outlier just in one metric
                return components[mask.index(1)]
            else:
                print('Alguns se salvaram')
                #exists more than one that is an outliers just in one metric
                value_to_process = 1
        elif mask.count(0) == 1:
            print('Um nao eh outlier')
            #exists just one that is not an outlier in both metrics
            return components[mask.index(0)]
        
        #here, one of the remaining components must be choosen
        #infinity geometries have a better comparison using distance from origin
        if features[feature_index]['type'].lower() in infinity_geometries:
            best_do = float('inf')
            best_index = 0
            for i, do in enumerate(distances_origin):
                if mask[i] == value_to_process and do < best_do:
                    best_do = do
                    best_index = i
            print('Infinity.')
            print(best_index, best_do)
        #in infinity geometries, the bigger component is used
        else:
            best_length = float('inf')
            best_index = 0
            for i, l in enumerate(lengths):
                if mask[i] == value_to_process and l < best_length:
                    best_length = l
                    best_index = i
            print(best_index, best_length)
        
        return components[best_index]
    elif len(components) == 1:
        return components[0]
    else:
        return []


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
    parser.add_argument('input', type=str, help='input file in .ply.')
    # parser.add_argument('output', type=str, help='output folder.')
    parser.add_argument('--distance_threshold', type = float, default = 0.001, help='distance threshold to consider a vertex as possible inlier of a feature.')
    parser.add_argument('--features_yaml', type = str, default='', help='load features from a yaml file.')
    # parser.add_argument('--centralize', type = bool, default=True, help='bool to centralize or not.')
    # parser.add_argument('--align', type = bool, default=True, help='bool to canonical alignment or not.')
    # parser.add_argument('--cube_rescale_factor', type = float, default=1, help='argument to make the point cloud lie in a unit cube, the factor multiplies all the dimensions of result cube.')
    args = vars(parser.parse_args())

    inputname = args['input']
    distance_threshold = args['distance_threshold']
    features_dir = args['features_yaml']

    if inputname[inputname.index('.'):] == '.ply':
        print('Reading .ply file...')
        plydata = PlyData.read(inputname)
        mesh_vertexes = plydata['vertex'].data
        vertexes = plydata['vertex'].count
        mesh_faces_or = plydata['face'].data
        mesh_faces = np.empty(shape=(mesh_faces_or.shape[0],mesh_faces_or[0][0].shape[0]))
        for i, mesh in enumerate(mesh_faces_or):
            face = mesh[0]
            mesh_faces[i] = face
        faces = plydata['face'].count
        print(plydata['normal'])
        exit()
    elif inputname[inputname.index('.'):] == '.obj':
        print('Reading .obj file...')
        objdata = pywavefront.Wavefront(inputname, collect_faces=True)
        mesh_vertexes = np.array(list(objdata.vertices))
        vertexes = mesh_vertexes.shape[0]
        mesh_normals = np.array(list(objdata.normals))
        mesh_faces = np.empty(shape=(0,3), dtype=np.int32)
        for mesh in objdata.mesh_list:
            for face in mesh.faces:
                face_array = np.array([face], dtype=np.int32)
                mesh_faces = np.append(mesh_faces, face_array, axis=0)
        faces = mesh_faces.shape[0]
        exit()
    else:
        print('{} file type can not be processed.'.format(inputname[inputname.index('.'):]))
        exit()

    print('{} faces and {} vertexes.'.format(faces, vertexes))
    print('Done.')

    vertex_graph = [[] for i in range(0,vertexes)]
    mount_graph()

    features = load_features(features_dir)
    features = features['curves'] + features['surfaces']

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
            for vertex in face_vertex:
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

    for i, fv in enumerate(features_vectors):
        collection = mplot3d.art3d.Poly3DCollection(fv)
        color_key = list(colors_list.keys())[randint(0, len(colors_list.keys())-1)]
        color = colors_list[color_key]
        collection.set_facecolor(color)
        axes.add_collection3d(collection)
        scale = np.append(scale, fv.flatten())

        fvi = set(features_vertex_indices[i])
        print(len(fvi))
        gt  = set(features[i]['vert_indices'])
        print(len(gt))

        iou = len((fvi & gt))/len((fvi | gt))

        print(i, features[i]['type'], fv.shape, color_key, 'IoU: {}'.format(iou))

    # Auto scale to the mesh size
    axes.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    pyplot.show()