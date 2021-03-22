import numpy as np
#from plyfile import PlyData, PlyElement

from mpl_toolkits import mplot3d
from matplotlib import pyplot, colors

from random import randint

from math import sqrt, acos, pi

import argparse

from tqdm import tqdm

#import pywavefront

import pymesh

from statistics import quantiles

import yaml

import sys
sys.setrecursionlimit(10000)

features = None

mesh_vertices = None
vertex_normal = None
mesh_faces = None

vertex_graph = None

feature_vertices_deviation = None

distance_threshold = 0
angle_threshold = 0

possible_connected_components = []

def load_features(dir):
    with open(dir) as f:
        features = yaml.load(f, Loader=yaml.FullLoader)
    return features

'''
- Distance and angle deviation functions
'''

def distance_points(A, B):
    AB = B - A
    return np.linalg.norm(AB, ord=2)

def angle_vectors(n1, n2):
    c = np.dot(n1, n2)/(np.linalg.norm(n1, ord=2)*np.linalg.norm(n2, ord=2)) 
    return acos(c)
        
def deviation_point_line(point, normal, curve):
    A = np.array(list(curve['location'])[0:3])
    v = np.array(list(curve['direction'])[0:3])
    P = np.array(list(point)[0:3])
    n_p = np.array(list(normal)[0:3])
    #AP vector
    AP = P - A
    #equation to calculate distance between point and line using a direction vector
    return np.linalg.norm(np.cross(v, AP), ord=2)/np.linalg.norm(v, ord=2), abs(pi/2 - angle_vectors(v, n_p))

def deviation_point_circle(point, normal, curve):
    A = np.array(list(curve['location'])[0:3])
    n = np.array(list(curve['z_axis'])[0:3])
    P = np.array(list(point)[0:3])
    n_p = np.array(list(normal)[0:3])
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

    #calculanting tangent vector to the circle in that point
    n_pp = (P_p - A)/np.linalg.norm((P_p - A), ord=2)
    t = np.cross(n_pp, n)
    return sqrt(a**2 + b**2), abs(pi/2 - angle_vectors(t, n_p))


def deviation_point_sphere(point, normal, surface):
    A = np.array(list(surface['location'])[0:3])
    P = np.array(list(point)[0:3])
    n_p = np.array(list(normal)[0:3])
    radius = surface['radius']
    #normal of the point projected in the sphere surface
    AP = P - A
    n_pp = AP/np.linalg.norm(AP, ord=2)
    #simple, distance from point to the center minus the sphere radius
    #angle between normals
    return abs(distance_points(P, A) - radius), angle_vectors(n_pp, n_p)


def deviation_point_plane(point, normal, surface):
    A = np.array(list(surface['location'])[0:3])
    n = np.array(list(surface['z_axis'])[0:3])
    P = np.array(list(point)[0:3])
    n_p = np.array(list(normal)[0:3])
    AP = P - A
    #orthogonal distance between point and plane
    #angle between normals
    angle = angle_vectors(n, n_p)
    if angle > pi/2:
        angle = pi - angle

    return abs(np.dot(AP, n)/np.linalg.norm(n, ord=2)), angle


def deviation_point_torus(point, normal, surface):
    A = np.array(list(surface['location'])[0:3])
    n = np.array(list(surface['z_axis'])[0:3])
    P = np.array(list(point)[0:3])
    n_p = np.array(list(normal)[0:3])
    max_radius = surface['max_radius']
    min_radius = surface['min_radius']
    radius = (max_radius - min_radius)/2

    AP = P - A
    #orthogonal distance to the torus plane 
    h = np.dot(AP, n)/np.linalg.norm(n, ord = 2)

    line = surface
    line['direction'] = surface['z_axis']
    #orthogonal distance to the revolution axis line 
    d = deviation_point_line(point, normal, line)[0]

    #projecting the point in the torus plane
    P_p = P - h*n/np.linalg.norm(n, ord=2)
    #getting the direction vector, using center as origin, to the point projected
    v = (P_p - A)/np.linalg.norm((P_p - A), ord=2)
    #calculating the center of circle in the direction of the input point
    B = (min_radius + radius)*v + A

    BP = P - B
    n_pp = BP/np.linalg.norm(BP, ord=2)
    print(n_pp)

    return abs(distance_points(B, P) - radius), angle_vectors(n_pp, n_p)


def deviation_point_cylinder(point, normal, surface):
    A = np.array(list(surface['location'])[0:3])
    radius = surface['radius']
    surface['direction'] = surface['z_axis']
    P = np.array(list(point)[0:3])
    n_p = np.array(list(normal)[0:3])
    #normal of the point projected in the sphere surface
    AP = P - A
    n_pp = AP/np.linalg.norm(AP, ord=2)
    #simple distance from point to the revolution axis line minus radius
    return abs(deviation_point_line(point, normal, surface)[0] - radius), angle_vectors(n_pp, n_p)

def deviation_point_cone(point, normal, surface):
    A = np.array(list(surface['location'])[0:3])
    v = np.array(list(surface['z_axis'])[0:3])
    B = np.array(list(surface['apex'])[0:3])
    P = np.array(list(point)[0:3])
    radius = surface['radius']
    n_p = np.array(list(normal)[0:3])

    #height of cone
    h = distance_points(A, B)

    AP = P - A
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
            #not using distance_point_circle function to not repeat operations
            a = dist_pointproj_center - radius
            b = np.linalg.norm(P - P_p, ord=2)
            n_pp = (P_p - A)/np.linalg.norm((P_p - A), ord=2)
            t = np.cross(n_pp, v)
            return sqrt(a**2 + b**2), abs(pi/2 - angle_vectors(t, n_p))
        
        #if not, the orthogonal distance to the circle plane is used
        return abs(signal_dist_point_plane), angle_vectors(-v, n_p)
    #if point is above the apex, return the distance from point to apex
    elif dist_AP > dist_BP and dist_AP >= h:
        return distance_points(P, B), angle_vectors(v, n_p)

    #if not, calculate the radius of the circle in this point height 
    r = radius*dist_BP/h

    d = (P - P_p)/np.linalg.norm((P-P_p), ord=2)

    vr = r * d

    P_s = P_p + vr

    s = (P_s - B)/np.linalg.norm((P_s - B), ord=2)

    t = np.cross(d, v)

    n_pp = np.cross(t, s)

    #distance from point to the point projected in the revolution axis line minus the current radius
    return abs(distance_points(P, P_p) - r), angle_vectors(n_pp, n_p)

deviation_functions = {
    'line': deviation_point_line,
    'circle': deviation_point_circle,
    'sphere': deviation_point_sphere,
    'plane': deviation_point_plane,
    'torus': deviation_point_torus,
    'cylinder': deviation_point_cylinder,
    'cone': deviation_point_cone
}

#somente um teste, isso deve ser melhor definido
min_points = {
    'line': 2,
    'circle': 3,
    'sphere': 3,
    'plane': 3,
    'torus': 3,
    'cylinder': 3,
    'cone': 3
}

infinity_geometries = ['line', 'plane', 'cylinder']

'''
- Graph part
'''

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
    print('Matching features and vertices...')
    for i, feature in tqdm(enumerate(features)):
        #isso pode ser trocado por dicionario de funcoes
        feature_type = feature['type'].lower()
        if feature['type'].lower() not in deviation_functions.keys():
            continue
        count = 0
        #isso vai ser trocado por algo mais inteligante (kd-tree, octree ou quadtree)
        for j, vertex in enumerate(mesh_vertices):
            ds, angle = deviation_functions[feature['type'].lower()](vertex, vertex_normal[j, :], feature)
            if ds < distance_threshold:
                count += 1
                do = distance_points(np.array(list(vertex)[0:3]), np.array(list(feature['location'])[0:3]))
                #index, distance to surface, distance to origin
                feature_vertices_deviation[i][j] = (ds, angle, do)
    print('Done.')

def dfsUtil(vertices_dict, v, info, visited, cc, info_acc):
    if visited[v] == False:
        cc.append(v)
        info_acc = (info_acc[0] + info[0], info_acc[1] + info[1], info_acc[2] + info[2])
        visited[v] = True
        adjacency = vertex_graph[v]
        for a in adjacency:
            if visited[a] == False and a in vertices_dict.keys():
                cc, info_acc = dfsUtil(vertices_dict, a, vertices_dict[a], visited, cc, info_acc)
   
    return cc, info_acc

def found_possible_connected_components(feature_index, vertices_dict):
    visited = [False] * len(vertex_graph)
    components = []
    lengths = []
    distances_surface = []
    angles = []
    distances_origin = []
    for v, info in vertices_dict.items():
        if visited[v] == False:
            component, info_accumulator = dfsUtil(vertices_dict, v, info, visited, [], (0.0, 0.0, 0.0))
            
            if len(component) >= min_points[features[feature_index]['type'].lower()]:
                components.append(component)
                ds_acc, angle_acc, do_acc = info_accumulator

                lengths.append(len(component))
                distances_surface.append(ds_acc/len(component))
                angles.append(angle_acc/len(component))
                distances_origin.append(do_acc/len(component))
    
    #qual componente deve ser selecionada? depende do tipo de feature? usamos ds, do ou o tamanho da componente?
    if len(components) > 1:
        print('\n')
        print(feature_index, 'em processo de selecao.')
            

        print('Lengths:', lengths)
        print('DS:', distances_surface)
        print('ANG:', angles)
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
    for i, vertices_dict in tqdm(enumerate(feature_vertices_deviation)):
        component = found_possible_connected_components(i, vertices_dict)
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
    parser.add_argument('input', type=str, help='input file in .obj, .off, .ply, .stl, .mesh (MEDIT), .msh (Gmsh) and .node/.face/.ele (Tetgen) formats.')
    # parser.add_argument('output', type=str, help='output folder.')
    parser.add_argument('--distance_threshold', type = float, default = 0.001, help='distance threshold to consider a vertex as possible inlier of a feature.')
    parser.add_argument('--angle_threshold', type = float, default = 0.44, help='angle threshold to consider a vertex as possible inlier of a feature.')
    parser.add_argument('--features_yaml', type = str, default='', help='load features from a yaml file.')
    # parser.add_argument('--centralize', type = bool, default=True, help='bool to centralize or not.')
    # parser.add_argument('--align', type = bool, default=True, help='bool to canonical alignment or not.')
    # parser.add_argument('--cube_rescale_factor', type = float, default=1, help='argument to make the point cloud lie in a unit cube, the factor multiplies all the dimensions of result cube.')
    args = vars(parser.parse_args())

    inputname = args['input']
    distance_threshold = args['distance_threshold']
    angle_threshold = args['angle_threshold']
    features_dir = args['features_yaml']

    print('Reading {} file...'.format(inputname[inputname.index('.'):]))
    mesh = pymesh.load_mesh(inputname)
    mesh.add_attribute('vertex_normal')
    mesh_vertices = mesh.vertices
    n_vertices = mesh.num_vertices
    vertex_normal = mesh.get_attribute('vertex_normal')
    vertex_normal = np.reshape(vertex_normal, (int(vertex_normal.shape[0]/3), 3))
    mesh_faces = mesh.faces
    n_faces = mesh.num_faces

    print('{} faces and {} vertices.'.format(n_faces, n_vertices))
    print('Done.')

    vertex_graph = [[] for i in range(0,n_vertices)]
    mount_graph()

    features = load_features(features_dir)
    features = features['curves'] + features['surfaces']
    # print(features[45]['x_axis'])
    # print(features[53]['x_axis'])
    # exit()

    feature_vertices_deviation = [{} for i in range(0,len(features))]
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
                vertex_vector = list(mesh_vertices[vertex])
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





#  component_indices = []
#     for c in components:
#         c_set = set(c)
#         if c_set in possible_connected_components:
#             component_indices.append(possible_connected_components.index(c_set))
#         else:
#             possible_connected_components.append(c_set)
#             component_indices.append(len(possible_connected_components) - 1)