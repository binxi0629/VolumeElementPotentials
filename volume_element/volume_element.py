from pymatgen.io.vasp import Poscar, Outcar
import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.spatial import Delaunay


'''
Utility and foundation functions
'''
def read_structure(fname:str):
    abc, xyz, atom = [], [], []
    poscar = Poscar.from_file(fname)
    lattice = poscar.as_dict()['structure']['lattice']['matrix']
    for site in poscar.as_dict()['structure']['sites']:
        abc.append(site['abc'])
        xyz.append(site['xyz'])
        atom.append(site['label'])
    return np.array(lattice), np.array(abc), np.array(xyz), atom


def fetch_pos_from_outcar(fname:str):
    ...


# This method neglects z-direction
# It only adds images at the nearest quadrants of the center atom (See the logic below)
def minimum_pbc_2d(abc:np.ndarray, atom:list, center:np.ndarray):
    # Along x-direction, add right image if center is at the right side, vice versa
    a_shift = 1. if center[0] > 0.5 else -1.
    img_a = abc + np.array([a_shift, 0., 0.])
    # Along y-direction, add top image if center is at the top side, vice versa
    b_shift = 1. if center[1] > 0.5 else -1.
    img_b = abc + np.array([0., b_shift, 0.])
    # Along xy-direction (corners), add extra top-right image if top and right image are added, vice versa
    img_ab = abc + np.array([a_shift, b_shift, 0.])
    # return 4 times the original atom list because 3 images are added
    return np.concatenate([abc, img_a, img_b, img_ab], axis=0), atom * 4


# This method finds first and second nearest neighbours (of different kind of atoms than center atom) in fractional coords
def find_nn(abc: np.ndarray, atom: list, lat: np.ndarray, center: np.ndarray, center_atom: str, fnn_noncen=3,
            snn_noncen=3, fnn_cen=6):
    atom = np.array(atom)
    # Shift the center atom to origin, for distance calculation
    shifted_abc = abc - center
    dist = np.dot(np.dot(shifted_abc, lat), np.dot(shifted_abc, lat).T).diagonal() # 240x1

    # Basically filter and get the coords and dist of Nitrogen atoms, dropping Boron atoms (that is, it only extracts other kinds of atoms (N) than the center atom (B) for this case)
    abc_filt_noncen = abc[atom != center_atom]
    dist_filt_noncen = dist[atom != center_atom]
    # Sort all Nitrogen atoms (in frac coords) by their distance compared to center
    idx_sort_noncen = np.argsort(dist_filt_noncen)
    # Extract 3 atoms with shortest distance as first NN, followed by 3 atoms as second NN
    first_nn_abc_noncen = abc_filt_noncen[idx_sort_noncen[:fnn_noncen]]
    second_nn_abc_noncen = abc_filt_noncen[idx_sort_noncen[fnn_noncen:fnn_noncen + snn_noncen]]

    # Same for Boron
    abc_filt_cen = abc[atom == center_atom]
    dist_filt_cen = dist[atom == center_atom]
    idx_sort_cen = np.argsort(dist_filt_cen)
    # Extract the first 6 NN atoms
    first_nn_abc_cen = abc_filt_cen[idx_sort_cen[1:1 + fnn_cen]]
    return first_nn_abc_noncen, second_nn_abc_noncen, first_nn_abc_cen

def find_sec_nn(abc: np.ndarray, atom: list, lat: np.ndarray, center: np.ndarray, center_atom: str, fnn_noncen=3,
            snn_noncen=3, fnn_cen=6, tnn_noncen=6, tnn_cen=6):
    atom = np.array(atom)
    # Shift the center atom to origin, for distance calculation
    shifted_abc = abc - center
    dist = np.dot(np.dot(shifted_abc, lat), np.dot(shifted_abc, lat).T).diagonal()
    # Basically filter and get the coords and dist of Nitrogen atoms, dropping Boron atoms (that is, it only extracts other kinds of atoms (N) than the center atom (B) for this case)
    abc_filt_noncen = abc[atom != center_atom]
    dist_filt_noncen = dist[atom != center_atom]
    # Sort all Nitrogen atoms (in frac coords) by their distance compared to center
    idx_sort_noncen = np.argsort(dist_filt_noncen)
    # Extract 3 atoms with shortest distance as first NN, followed by 3 atoms as second NN
    first_nn_abc_noncen = abc_filt_noncen[idx_sort_noncen[:fnn_noncen]]
    second_nn_abc_noncen = abc_filt_noncen[idx_sort_noncen[fnn_noncen:fnn_noncen + snn_noncen]]
    third_nn_abc_noncen = abc_filt_noncen[idx_sort_noncen[fnn_noncen+snn_noncen:fnn_noncen + snn_noncen+tnn_noncen]]

    # Same for Boron
    abc_filt_cen = abc[atom == center_atom]
    dist_filt_cen = dist[atom == center_atom]
    idx_sort_cen = np.argsort(dist_filt_cen)
    # Extract the first 6 NN atoms
    first_nn_abc_cen = abc_filt_cen[idx_sort_cen[1:1 + fnn_cen]]
    second_nn_abc_cen = abc_filt_cen[idx_sort_cen[1+fnn_cen:1+fnn_cen+tnn_cen]]
    return first_nn_abc_noncen, second_nn_abc_noncen, third_nn_abc_noncen, first_nn_abc_cen, second_nn_abc_cen

# This method is for obtaining the 7 points of the area element, includes computing the non-atom vertices by the triangle-averaging method (using second NN and first NN)
def compute_area_vertices(fnn_abc_noncen: np.ndarray, snn_abc_noncen: np.ndarray, fnn_abc_cen: np.ndarray,
                          lat: np.ndarray, center: np.ndarray, center_idx: int, cyclic_sort=False,
                          shift_to_center=False):

    # Area element vertices, the center and first NN atoms must be the vertices
    vertices = np.array(fnn_abc_noncen)
    # Find the non-atom sites for vertices
    for site in snn_abc_noncen:
        # For each atom of first NN, shift it as origin and calculate the dist of 1st NNs
        fnn_noncen_dist = np.dot(np.dot(fnn_abc_noncen - site, lat), np.dot(fnn_abc_noncen - site, lat).T).diagonal()
        fnn_cen_dist = np.dot(np.dot(fnn_abc_cen - site, lat), np.dot(fnn_abc_cen - site, lat).T).diagonal()
        # The two closer first NN points to the second NN atom site forms the hexagon (for averaging)
        close_points_noncen = fnn_abc_noncen[fnn_noncen_dist != fnn_noncen_dist.max()]
        close_points_cen = fnn_abc_cen[np.argsort(fnn_cen_dist)[:2]]
        # The average of 6 vertices gives the non-atom site vertex for area element
        non_atom_vert = (center + site + close_points_noncen.sum(axis=0) + close_points_cen.sum(axis=0)) / 6
        vertices = np.append(vertices, [non_atom_vert], axis=0)
    # Sort all six vertices by angles (such that they are in cyclic order)
    # Basically get phi = arctan(y / x) with quadrants
    if cyclic_sort:
        shifted_vert = vertices - center
        # Angle 0deg start from y-axis clockwise
        phi = np.arctan2((shifted_vert[:, 0]), (shifted_vert[:, 1]))
        # print(phi)
        vertices = vertices[np.argsort(phi), :]
    # print(nn_avg_img)
    if shift_to_center:
        return np.append([[0, 0, 0]], vertices - center, axis=0)
    else:
        # Return vertices as [center, ... cyclic-sorted vertices ...]
        return np.append([center], vertices, axis=0)


def compute_snn_area_vertices(fnn_abc_noncen: np.ndarray, snn_abc_noncen: np.ndarray, tnn_abc_noncen: np.ndarray,
                                 fnn_abc_cen: np.ndarray, snn_abc_cen: np.ndarray,
                                lat: np.ndarray, center: np.ndarray, center_idx: int, cyclic_sort=False,
                                shift_to_center=False):
    # Area element vertices, the center and first NN atoms must be the vertices
    snn_vertices = np.array(snn_abc_noncen)
    # Find the non-atom sites for vertices
    for site in fnn_abc_noncen:
        # For each atom of second NN, shift it as origin and calculate the dist of 3rd NNs
        fnn_noncen_dist = np.dot(np.dot(tnn_abc_noncen - site, lat), np.dot(tnn_abc_noncen - site, lat).T).diagonal()


        # The two closer first NN points to the second NN atom site forms the hexagon (for averaging)
        idx_sort_noncen = np.argsort(fnn_noncen_dist)


        #two noncen vertices
        close_points_noncen = tnn_abc_noncen[idx_sort_noncen[:2]] #2
        fnn_cen_dist = np.dot(np.dot(fnn_abc_cen - close_points_noncen[0], lat), np.dot(fnn_abc_cen - close_points_noncen[0], lat).T).diagonal()
        snn_cen_dist = np.dot(np.dot(snn_abc_cen - close_points_noncen[0], lat), np.dot(snn_abc_cen - close_points_noncen[0], lat).T).diagonal()
        idx_sort_fnn_cen = np.argsort(fnn_cen_dist)
        idx_sort_snn_cen = np.argsort(snn_cen_dist)

        close_points_fnn_cen = fnn_abc_cen[idx_sort_fnn_cen[:2]] # 2
        close_points_snn_cen = snn_abc_cen[idx_sort_snn_cen[0]] # 1

        # The average of 6 vertices gives the non-atom site vertex for area element
        non_atom_vert = (site + close_points_noncen.sum(axis=0) + close_points_fnn_cen.sum(axis=0) + close_points_snn_cen) / 6
        snn_vertices = np.append(snn_vertices, [non_atom_vert], axis=0)
    # Sort all six vertices by angles (such that they are in cyclic order)
    # Basically get phi = arctan(y / x) with quadrants
    if cyclic_sort:
        shifted_vert = snn_vertices - center
        # Angle 0deg start from y-axis clockwise
        phi = np.arctan2((shifted_vert[:, 0]), (shifted_vert[:, 1]))
        # print(phi)
        snn_vertices = snn_vertices[np.argsort(phi), :]
    # print(nn_avg_img)
    if shift_to_center:
        return np.append([[0, 0, 0]], snn_vertices - center, axis=0)
    else:
        # Return vertices as [center, ... cyclic-sorted vertices ...]
        return np.append([center], snn_vertices, axis=0)


def triangle_area(vert):
    # The norm of the cross product of two sides is twice the area
    n = np.cross(vert[1] - vert[0], vert[2] - vert[0])
    return np.linalg.norm(n) / 2

def compute_all_areas(vert_abc, lat, tri_idx=None):
    # Rescaling
    vert_xyz = np.dot(vert_abc, lat)
    area = 0
    # If triangulation indices is not given, Delaunay (trivial) triangulation of hexagon is done automatically
    if tri_idx is None:
        tri = Delaunay(vert_xyz[:, :2])
        tri_vert = np.array([vert_xyz[idx] for idx in tri.simplices])
    else:
        tri_vert = np.array([vert_xyz[idx] for idx in tri_idx])
    for vert in tri_vert:
        area += triangle_area(vert)
    return tri_vert, area


def compute_union_area(template_coords, elem_coords, tri_idx=None):
    """

    :param template_coords: the coords. of template hexagon (6x2)
    :param elem_coords: the coords. of given area element (6x2)
    :param tri_idx: triangulation indices, if None: Delaunay (trivial) triangulation of hexagon is done automatically
    :return: triangles coords. and the union area
    """
    area = 0
    union_area = np.concatenate((template_coords, np.array(elem_coords)), axis=0)
    # If triangulation indices is not given, Delaunay (trivial) triangulation of hexagon is done automatically
    if tri_idx is None:
        tri = Delaunay(union_area[:,:2])
        tri_vert = np.array([union_area[idx] for idx in tri.simplices])
    else:
        tri_vert = np.array([union_area[idx] for idx in tri_idx])

    for vert in tri_vert:
        area += triangle_area(vert)

    return tri_vert, area


def area_element_optimizer(func, x0, lr=1e-3, delta_y=1e-5):
    min_y = func(x0)
    opt_x = x0
    max_epochs = int((2*math.pi) // lr)+1
    x_trail = x0

    for epoch in range(max_epochs):
        x_trail += +lr
        y_trail = func(x_trail)
        if min_y > y_trail:
            min_y = y_trail
            opt_x = x_trail

    return min_y, opt_x


def align_with_template(all_ele, initial_alpha=0):

    area_element_template = np.array([[-7.25511541e-01, - 1.25637484e+00, 0], [-1.45073445e+00, - 1.59299988e-06,  0],
                                        [-7.25511549e-01,  1.25636863e+00,  0], [7.25222400e-01,  1.25637038e+00,  0],
                                        [1.45044524e+00, - 2.86353100e-06,  0], [7.25222731e-01, - 1.25637333e+00,  0]])

    def align_step(alpha=initial_alpha, return_verts=False):
        rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
        new_elem_coords = np.dot(rot_matrix, elem_coords.T)
        tri_verts, union_area = compute_union_area(area_element_template, new_elem_coords.T)
        if return_verts:
            return tri_verts, union_area
        else:
            return np.array(union_area)

    new_ele = []
    flag = False
    for elem_coords in all_ele:

        if not flag:
            # FIXME: For debuging use, comment the next following lines
            # tri_verts, union_area = compute_union_area(template_coords=area_element_template, elem_coords=elem_coords)
            # union_area_verts = np.concatenate((area_element_template, np.array(elem_coords)), axis=0)
            # verts_batch = np.array([area_element_template, elem_coords])
            # plot_area_elements(verts_batch)
            # # plot_triangulation(tri_verts, union_area, union_area_verts, center_to_origin, atom)
            # print("initial union area:", union_area)

            # alpha = minimize(align_step, initial_alpha, method='nelder-mead', options={'xatol': 1e-3, 'disp': True})
            _, alpha = area_element_optimizer(align_step, initial_alpha, lr=5e-3)
            print("Rotational alignment (radian)", alpha)
            rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
            new_elem_coords = np.dot(rot_matrix, elem_coords.T).T

            # FIXME: For debuging use, comment the next three lines
            # tri_verts, union_area = align_step(alpha, return_verts=True)
            # # plot_triangulation(tri_verts, union_area, union_area_verts, center_to_origin, atom)
            # verts_batch = np.array([area_element_template, new_elem_coords])
            # plot_area_elements(verts_batch)
            # print(f"optimized union area: {union_area} | Optimized rotation angle {alpha}")

            # reordering the permutation
            shift = 0

            for i in range(len(new_elem_coords)):
                if new_elem_coords[i, 0] > 1.0: # idx=4
                    shift = 0 if i == 4 else 4-i
                    break

            # reordering
            idx = list(range(len(elem_coords)))
            if shift >= 0:
                new_idx = [i - shift if i - shift > -6 else i - shift + 6 for i in idx]
            else:
                new_idx = [i - shift if i - shift < 6 else i - shift - 6 for i in idx]

            flag = True
            new_ele.append(new_elem_coords[new_idx])
        else:
            new_elem_coords = np.dot(rot_matrix, elem_coords.T).T
            new_ele.append(new_elem_coords[new_idx])

    return np.array(new_ele)


def plot_area_elements(ele_vert_batch):
    fig = plt.figure()

    ele_vert_batch = np.array(ele_vert_batch)
    if len(ele_vert_batch.shape) == 3:
        for ele_vert in ele_vert_batch:
            ele_vert = np.append(ele_vert, [ele_vert[0]], axis=0)
            xs, ys, zs = zip(*ele_vert)
            plt.plot(xs, ys)
    else:
        ele_vert = np.append(ele_vert_batch, [ele_vert_batch[0]], axis=0)
        xs, ys, zs = zip(*ele_vert)

        plt.plot(xs, ys)
        idx=1
        for x, y in zip(xs, ys):
            # if idx > 7: break
            label = "{}".format(idx)
            idx+=1

            plt.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center

    plt.title("Visualization of the area element")
    plt.show()


def plot_triangulation(tri_vert, area, xyz, atom, fname="area_element", fid="0"):
    fig = plt.figure(figsize=(8, 8))
    for coords in tri_vert:
        coords = np.append(coords, [coords[0]], axis=0)
        xs, ys, zs = zip(*coords)
        plt.plot(xs, ys, c="b")
    # for ele in np.unique(atom):
    #     xyz_at = xyz[np.array(atom) == ele]
    #     plt.scatter(xyz_at[:, 0], xyz_at[:, 1], label=ele)
    plt.title(f"Area = {area}")
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.savefig(f"./img/{fname}_{fid}.png")
    plt.show()
    plt.clf()

    return

'''
Compute all area elements function

lat: lattice cell vectors (for rescaling) 3x3
abc: frac coords of all atoms 60x3
atom: atom type of all atoms
n_center: assumed first 30 atoms are center of area elements (all B atoms)
return_frac: return all vertices as frac coords, otherwise rescale the vertices
ignore_center: reject the center atom of area elements as one of the vertices
'''

def all_element_vertices(lat:np.ndarray,
                         abc:np.ndarray,
                         atom:list,
                         n_center=30,
                         return_frac=True,
                         ignore_center=True,
                         cyclic_sort=False,
                         shift_to_center=False,
                         center_atom='B'):

    center_atoms_range = list(range(n_center)) if center_atom == 'B' else list(range(n_center, 2*n_center, 1))

    all_ele = []
    for center_idx in center_atoms_range:
        center, center_atom = abc[center_idx], atom[center_idx]
        abc_pbc, atom_pbc = minimum_pbc_2d(abc, atom, center)
        fnn_noncen, snn_noncen, fnn_cen = find_nn(abc_pbc, atom_pbc, lat, center, center_atom)
        # first_nn, second_nn = find_nn(abc_pbc, atom_pbc, lat, center, center_atom)
        area_vert_abc = compute_area_vertices(fnn_noncen, snn_noncen, fnn_cen, lat, center, center_idx, cyclic_sort=cyclic_sort, shift_to_center=shift_to_center)
        if ignore_center:
            area_vert_abc = area_vert_abc[1:]
        if return_frac:
            all_ele.append(area_vert_abc)
        else:
            all_ele.append(np.dot(area_vert_abc, lat))
    return np.array(all_ele)


def all_snn_element_vertices(lat:np.ndarray,
                         abc:np.ndarray,
                         atom:list,
                         n_center=30,
                         return_frac=True,
                         ignore_center=True,
                         cyclic_sort=False,
                         shift_to_center=False,
                         center_atom='B'):

    center_atoms_range = list(range(n_center)) if center_atom == 'B' else list(range(n_center, 2*n_center, 1))

    all_ele = []
    for center_idx in center_atoms_range:
        center, center_atom = abc[center_idx], atom[center_idx]
        abc_pbc, atom_pbc = minimum_pbc_2d(abc, atom, center)
        fnn_noncen, snn_noncen, tnn_noncen, fnn_cen, snn_cen = find_sec_nn(abc_pbc, atom_pbc, lat, center, center_atom)
        # first_nn, second_nn = find_nn(abc_pbc, atom_pbc, lat, center, center_atom)

        area_vert_abc = compute_snn_area_vertices(fnn_noncen, snn_noncen, tnn_noncen, fnn_cen, snn_cen, lat, center,
                                                  center_idx,
                                                  cyclic_sort=cyclic_sort, shift_to_center=shift_to_center)
        if ignore_center:
            area_vert_abc = area_vert_abc[1:]
        if return_frac:
            all_ele.append(area_vert_abc)
        else:
            all_ele.append(np.dot(area_vert_abc, lat))
    return np.array(all_ele)

def collect_snn_area_vertices(lat:np.ndarray,
                         abc:np.ndarray,
                         atom:list,
                         n_center=30,
                         return_frac=True,
                         ignore_center=True,
                         cyclic_sort=False,
                         shift_to_center=False,
                         center_atom='B'):
    center_atoms_range = list(range(n_center)) if center_atom == 'B' else list(range(n_center, 2 * n_center, 1))

    all_ele = []
    for center_idx in center_atoms_range:
        center, center_atom = abc[center_idx], atom[center_idx]
        abc_pbc, atom_pbc = minimum_pbc_2d(abc, atom, center)
        fnn_noncen, snn_noncen, fnn_cen = find_nn(abc_pbc, atom_pbc, lat, center, center_atom)

        vertices = np.concatenate([fnn_noncen, snn_noncen, fnn_cen], axis=0)
        idx_tmp = np.argsort(vertices.sum(axis=1))
        vertices = vertices[idx_tmp, :]

        # temp =[]
        # Find the non-atom sites for vertices


        # for site in snn_noncen:
        #     # For each atom of first NN, shift it as origin and calculate the dist of 1st NNs
        #     fnn_noncen_dist = np.dot(np.dot(fnn_noncen - site, lat),
        #                              np.dot(fnn_noncen - site, lat).T).diagonal()
        #     fnn_cen_dist = np.dot(np.dot(fnn_cen - site, lat), np.dot(fnn_cen - site, lat).T).diagonal()
        #     # The two closer first NN points to the second NN atom site forms the hexagon (for averaging)
        #     close_points_noncen = fnn_noncen[fnn_noncen_dist != fnn_noncen_dist.max()]
        #     close_points_cen = fnn_cen[np.argsort(fnn_cen_dist)[:2]]
        #
        #     temp_vertices = np.concatenate(([center], close_points_noncen, close_points_cen, [site]), axis=0)
        #     shifted_vert = temp_vertices - temp_vertices.sum(axis=0) / 6
        #     # Angle 0deg start from y-axis clockwise
        #     phi = np.arctan2((shifted_vert[:, 0]), (shifted_vert[:, 1]))
        #     # print(phi)
        #     temp_vertices = temp_vertices[np.argsort(phi), :]
        #
        #     temp.append(temp_vertices)

        # vertices = np.concatenate((temp[0], temp[1], temp[2]), axis=0)

        if shift_to_center:
            vertices = np.append([[0, 0, 0]], vertices - center, axis=0)
        else:
            # Return vertices as [center, ... cyclic-sorted vertices ...]
            vertices = np.append([center], vertices, axis=0)

        if ignore_center:
            area_vert_abc = vertices[1:]
        if return_frac:
            all_ele.append(vertices)
        else:
            all_ele.append(np.dot(vertices, lat))

    return np.array(all_ele)


def plot_elements(ele_vert:np.ndarray, atom_pos:np.ndarray, atom:list, snn_ele=None,  fname="./img/all_ele.png"):
    fig, ax = plt.subplots()
    colors = ['blue', 'red']
    i = 0
    for ele in np.unique(atom):

        target_pos = atom_pos[np.array(atom) == ele]
        plt.scatter(target_pos[:, 0], target_pos[:, 1], label=ele,s=100, color=colors[i])
        i+=1

    # plot random cutoff circles
    c1 = plt.Circle((atom_pos[37, 0], atom_pos[37, 1]), 2, color='limegreen', fill=False, linewidth=1.5)
    c2 = plt.Circle((atom_pos[38, 0], atom_pos[38, 1]), 2, color='limegreen', fill=False, linewidth=1.5)
    c3 = plt.Circle((atom_pos[47, 0], atom_pos[38, 1]), 1.45, color='b', fill=False, linewidth=1.5)
    c4 = plt.Circle((atom_pos[46, 0], atom_pos[38, 1]), 1.45, color='b', fill=False, linewidth=1.5)

    for coords in ele_vert:
        coords = np.append(coords, [coords[0]], axis=0)
        xs, ys, zs = zip(*coords)
        ax.plot(xs, ys, c="k")

    if snn_ele is not None:
        for coords in snn_ele:
            coords = np.append(coords, [coords[0]], axis=0)
            xs, ys, zs = zip(*coords)
            ax.plot(xs, ys, c="red")

    # plt.title(f"All elements")
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='20')
    # plt.axis('off')
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.add_patch(c3)
    ax.add_patch(c4)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(fname, dpi=800)
    # plt.clf()


def testing_rotational_alignment():
    fname = '../extra_dataset/extra_9_rot80.vasp'
    lat, abc, xyz, atom = read_structure(fname)

    fnn_ele = all_element_vertices(lat, abc, atom, n_center=30, return_frac=False, cyclic_sort=True,
                                   shift_to_center=True, center_atom='N')
    print("Shape:", fnn_ele.shape)
    new_ele_coords = align_with_template(elem_coords=fnn_ele[26])
    print("before rotating:", fnn_ele[26])
    print("after rotating:", new_ele_coords)
    plot_area_elements([fnn_ele[26], new_ele_coords])

if __name__ == "__main__":
    '''
    Sample usage to generate all area elements from a POSCAR
    ele contains 6 vertices of each area elements (3D array), iterate it yields 6 vertices of each element (2D array)
    e.g.
    [[[A1x1, A1y1, A1z1],
      [A1x2, A1y2, A1z2],
      [A1x3, A1y3, A1z3],
      [A1x4, A1y4, A1z4],
      [A1x5, A1y5, A1z5],
      [A1x6, A1y6, A1z6]],

     [[A2x1, A2y1, A2z1],
      [A2x2, A2y2, A2z2],
      [A2x3, A2y3, A2z3],
      [A2x4, A2y4, A2z4],
      [A2x5, A2y5, A2z5],
      [A2x6, A2y6, A2z6]],

      ...
    ]]]
    '''

    testing_rotational_alignment()
