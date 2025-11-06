import copy
from tqdm import tqdm
import math
import os, json, numpy as np, cv2, random
from collections import Counter
import networkx as nx

np.set_printoptions(threshold=np.inf, linewidth=999999)

def get_label(data):
    # Extract integer labels (<=12)
    integer_labels = [item[1] for item in data if item[1] <= 12]
    # Count occurrences of each integer label
    label_counts = Counter(integer_labels)
    # Select label with maximum frequency
    max_label = max(label_counts, key=label_counts.get)
    return max_label


def get_points_and_pixel_values_inside_polygon(image, polygon_vertices):
    # Compute bounding box of polygon
    min_x = min([vertex[0] for vertex in polygon_vertices])
    max_x = max([vertex[0] for vertex in polygon_vertices])
    min_y = min([vertex[1] for vertex in polygon_vertices])
    max_y = max([vertex[1] for vertex in polygon_vertices])
    # Create blank image matching bounding box size
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    blank_image = np.zeros((height, width), dtype=np.uint8)
    # Create numpy array for polygon points
    polygon_np = np.array([polygon_vertices], dtype=np.int32)
    # Fill polygon mask
    mask = cv2.fillPoly(blank_image, polygon_np - [min_x, min_y], 255)
    # Get all interior points
    points_inside_polygon = np.argwhere(mask == 255)
    # Map points back to original image coordinates and fetch pixel values
    points_and_pixel_values = []
    for point in points_inside_polygon:
        y, x = point
        pixel_value = image[y + min_y][x + min_x]
        points_and_pixel_values.append(((x + min_x, y + min_y), pixel_value))
    return points_and_pixel_values

def count_connected_components(imgggg):
    imgggg[np.where(imgggg >= 14)] = 255
    imgggg[np.where(imgggg <= 13)] = 0
    # Detect connected components
    num_labels, _ = cv2.connectedComponents(imgggg + 1)
    # Return count minus 2 (background & wall counted as components)
    return num_labels - 2


# Function to extract all edges
def extract_edges(coords, adjacency_mat):
    edges = []
    for i in range(len(adjacency_mat)):
        for j in range(i + 1, len(adjacency_mat[i])):
            if adjacency_mat[i][j]:
                edge = (coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
                edges.append(edge)
    return edges
def random_color(variance_threshold):
    color = [0, 0, 0]
    while np.var(color) < variance_threshold:
        color = [random.randint(0, 255) for _ in range(3)]
    return tuple(color)

def draw_edges(image, edges):
    for edge in edges:
        x1, y1, x2, y2 = edge
        color = random_color(variance_threshold=25)
        cv2.line(image, (x1.astype(np.int32), y1.astype(np.int32)), (x2.astype(np.int32), y2.astype(np.int32)), color, 1)
    return image

def get_quadrant(angle):
    # Degrees of the angle falling in each quadrant (two vectors assumed non-collinear)
    if angle[0] < angle[1]: # angle 1
        if 0 <= angle[0] < 90 and 0 <= angle[1] < 90:
            quadrant = (angle[1] - angle[0], 0, 0, 0)
        elif 0 <= angle[0] < 90 and 90 <= angle[1] < 180:
            quadrant = (90 - angle[0], angle[1] - 90, 0, 0)
        elif 0 <= angle[0] < 90 and 180 <= angle[1] < 270:
            quadrant = (90 - angle[0], 90, angle[1] - 180, 0)
        elif 0 <= angle[0] < 90 and 270 <= angle[1] < 360:
            quadrant = (90 - angle[0], 90, 90, angle[1] - 270)
        elif 90 <= angle[0] < 180 and 90 <= angle[1] < 180:
            quadrant = (0, angle[1] - angle[0], 0, 0)
        elif 90 <= angle[0] < 180 and 180 <= angle[1] < 270:
            quadrant = (0, 180 - angle[0], angle[1] - 180, 0)
        elif 90 <= angle[0] < 180 and 270 <= angle[1] < 360:
            quadrant = (0, 180 - angle[0], 90, angle[1] - 270)
        elif 180 <= angle[0] < 270 and 180 <= angle[1] < 270:
            quadrant = (0, 0, angle[1] - angle[0], 0)
        elif 180 <= angle[0] < 270 and 270 <= angle[1] < 360:
            quadrant = (0, 0, 270 - angle[0], angle[1] - 270)
        elif 270 <= angle[0] < 360 and 270 <= angle[1] < 360:
            quadrant = (0, 0, 0, angle[1] - angle[0])
    else: # angle 2
        if 0 <= angle[1] < 90 and 0 <= angle[0] < 90:
            quadrant_ = (angle[0] - angle[1], 0, 0, 0)
        elif 0 <= angle[1] < 90 and 90 <= angle[0] < 180:
            quadrant_ = (90 - angle[1], angle[0] - 90, 0, 0)
        elif 0 <= angle[1] < 90 and 180 <= angle[0] < 270:
            quadrant_ = (90 - angle[1], 90, angle[0] - 180, 0)
        elif 0 <= angle[1] < 90 and 270 <= angle[0] < 360:
            quadrant_ = (90 - angle[1], 90, 90, angle[0] - 270)
        elif 90 <= angle[1] < 180 and 90 <= angle[0] < 180:
            quadrant_ = (0, angle[0] - angle[1], 0, 0)
        elif 90 <= angle[1] < 180 and 180 <= angle[0] < 270:
            quadrant_ = (0, 180 - angle[1], angle[0] - 180, 0)
        elif 90 <= angle[1] < 180 and 270 <= angle[0] < 360:
            quadrant_ = (0, 180 - angle[1], 90, angle[0] - 270)
        elif 180 <= angle[1] < 270 and 180 <= angle[0] < 270:
            quadrant_ = (0, 0, angle[0] - angle[1], 0)
        elif 180 <= angle[1] < 270 and 270 <= angle[0] < 360:
            quadrant_ = (0, 0, 270 - angle[1], angle[0] - 270)
        elif 270 <= angle[1] < 360 and 270 <= angle[0] < 360:
            quadrant_ = (0, 0, 0, angle[0] - angle[1])
        quadrant = (90 - quadrant_[0], 90 - quadrant_[1], 90 - quadrant_[2], 90 - quadrant_[3])
    return quadrant

def poly_area(points): # Compute area of a counterclockwise (concave) polygon; negative if clockwise
    s = 0
    points_count = len(points)
    for i in range(points_count):
        point = points[i]
        point2 = points[(i + 1) % points_count]
        s += (point[0] - point2[0]) * (point[1] + point2[1])
    return s / 2

def rotate_degree_clockwise_from_counter_degree(src_degree, dest_degree):
    delta = src_degree - dest_degree
    return delta if delta >= 0 else 360 + delta

def rotate_degree_counterclockwise_from_counter_degree(src_degree, dest_degree):
    delta = dest_degree - src_degree
    return delta if delta >= 0 else 360 + delta


def x_axis_angle(y):
    # Image coordinate system: (1,0) is 0 degrees; rotate counterclockwise up to 360 degrees
    # print('-------------')
    # print(y)
    y_right_hand = (y[0], -y[1])
    # print(y_right_hand)

    x = (1, 0)
    inner = x[0] * y_right_hand[0] + x[1] * y_right_hand[1]
    # print(inner)
    y_norm2 = (y_right_hand[0] ** 2 + y_right_hand[1] ** 2) ** 0.5
    # print(y_norm2)
    cosxy = inner / y_norm2
    # print(cosxy)
    angle = math.acos(cosxy)
    # print(angle, math.degrees(angle))
    # print('-------------')
    return math.degrees(angle) if y_right_hand[1] >= 0 else 360 - math.degrees(angle)

def get_results_float_with_semantic(best_result):
    preds = best_result[2]
    # All points and edges
    output_points = []
    output_edges = []
    for triplet in preds:
        this_preds = triplet[0]
        last_edges = triplet[1]
        this_edges = triplet[2]
        for this_pred in this_preds:
            point = (this_pred['points'].tolist()[0], this_pred['points'].tolist()[1],
                     this_pred['semantic_left_up'].item(), this_pred['semantic_right_up'].item(),
                     this_pred['semantic_right_down'].item(), this_pred['semantic_left_down'].item())
            output_points.append(point)
        for last_edge in last_edges:
            point1 = (last_edge[0]['points'].tolist()[0], last_edge[0]['points'].tolist()[1],
                     last_edge[0]['semantic_left_up'].item(), last_edge[0]['semantic_right_up'].item(),
                     last_edge[0]['semantic_right_down'].item(), last_edge[0]['semantic_left_down'].item())
            point2 = (last_edge[1]['points'].tolist()[0], last_edge[1]['points'].tolist()[1],
                      last_edge[1]['semantic_left_up'].item(), last_edge[1]['semantic_right_up'].item(),
                      last_edge[1]['semantic_right_down'].item(), last_edge[1]['semantic_left_down'].item())
            edge = (point1, point2)
            output_edges.append(edge)
        for this_edge in this_edges:
            point1 = (this_edge[0]['points'].tolist()[0], this_edge[0]['points'].tolist()[1],
                      this_edge[0]['semantic_left_up'].item(), this_edge[0]['semantic_right_up'].item(),
                      this_edge[0]['semantic_right_down'].item(), this_edge[0]['semantic_left_down'].item())
            point2 = (this_edge[1]['points'].tolist()[0], this_edge[1]['points'].tolist()[1],
                      this_edge[1]['semantic_left_up'].item(), this_edge[1]['semantic_right_up'].item(),
                      this_edge[1]['semantic_right_down'].item(), this_edge[1]['semantic_left_down'].item())
            edge = (point1, point2)
            output_edges.append(edge)
    return output_points, output_edges

def get_cycle_basis_and_semantic(output_points_ori, output_edges_ori):
    output_points = [tuple(p + [99999, 99999, 99999, 99999]) for p in output_points_ori.tolist()]
    output_edges = [tuple([tuple([e[0], e[1]] + [99999, 99999, 99999, 99999]), tuple([e[2], e[3]] + [99999, 99999, 99999, 99999])]) for e in output_edges_ori]

    output_points = copy.deepcopy(output_points)
    output_edges = copy.deepcopy(output_edges)
    # print(output_points)
    # print(output_edges)
    # assert 0
    # Dictionary mapping output points to their index
    d = {}
    for output_point_index, output_point in enumerate(output_points):
        d[output_point] = output_point_index # Cannot handle duplicate points here; keep NMS
    d_rev = {}
    for output_point_index, output_point in enumerate(output_points):
        d_rev[output_point_index] = output_point # Cannot handle duplicate points here; keep NMS
    es = []
    for output_edge in output_edges:
        es.append((d[output_edge[0]], d[output_edge[1]]))
    # print(d)

    G = nx.Graph()
    for e in es:
        G.add_edge(e[0], e[1])

    simple_cycles = []
    simple_cycles_number = []
    simple_cycles_semantics = []
    # print('breakpoint1', simple_cycles)
    bridges = list(nx.bridges(G))
    # Handling virtual edges: delete them from the edge set and process each remaining component separately
    for b in bridges:
        if (d_rev[b[0]], d_rev[b[1]]) in output_edges:
            output_edges.remove((d_rev[b[0]], d_rev[b[1]]))
            es.remove((b[0], b[1]))
            G.remove_edge(b[0], b[1])
        if (d_rev[b[1]], d_rev[b[0]]) in output_edges:
            output_edges.remove((d_rev[b[1]], d_rev[b[0]]))
            es.remove((b[1], b[0]))
            G.remove_edge(b[1], b[0])
    # After removing bridge edges, only isolated nodes or cycles remain; iterate over remaining cycles
    connected_components = list(nx.connected_components(G))
    # print(connected_components)
    for c in connected_components:
        if len(c) == 1:
            pass
        else:
            simple_cycles_c = []
            simple_cycles_number_c = []
            simple_cycle_semantics_c = []
            # print(c) # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
            # Get subset of points and edges for this component
            output_points_c = [p for p in output_points if d[p] in c]
            output_edges_c = [e for e in output_edges if d[e[0]] in c or d[e[1]] in c] # Fixed edge set; not mutated
            output_edges_c_copy_for_traversing = copy.deepcopy(output_edges_c) # Working edge set for traversal; edges removed to reduce complexity
            # print(output_points_c)
            # print(output_edges_c)


            # Method to enumerate all counterclockwise simple cycles in this component:
            # 1. Use index mapping d for component points.
            # 2. For each undirected edge, choose smaller index as start; treat larger as current.
            # 3. Find all neighbor edges of current and compute their outgoing angles.
            # 4. Determine reverse incoming direction from last to current (polar angle in [0,360)).
            # 5. Rotate counterclockwise from that angle to select the last encountered neighbor edge.
            # 6. Advance to its opposite endpoint.
            # 7. Repeat until returning to the second point (cycle closed).
            # 8. Remove traversed edges (pi < p(i+1)) from traversal set to avoid repeats.
            # 9. Continue with remaining edges.

            for edge_c in output_edges_c:
                if edge_c not in output_edges_c_copy_for_traversing:
                    pass
                else:
                    simple_cycle_semantics = []
                    simple_cycle = []
                    simple_cycle_number = []
                    point1 = edge_c[0]
                    point2 = edge_c[1]
                    point1_number = d[point1]
                    point2_number = d[point2]
                    # Initial point
                    initial_point = None
                    initial_point_number = None
                    if point1_number < point2_number:
                        initial_point = point1
                        initial_point_number = point1_number
                    else:
                        initial_point = point2
                        initial_point_number = point2_number
                    simple_cycle.append(initial_point)
                    simple_cycle_number.append(initial_point_number)
                    # Previous point
                    last_point = initial_point
                    last_point_number = initial_point_number
                    # Current point
                    current_point = None
                    current_point_number = None
                    if point1_number < point2_number:
                        current_point = point2
                        current_point_number = point2_number
                    else:
                        current_point = point1
                        current_point_number = point1_number
                    simple_cycle.append(current_point)
                    simple_cycle_number.append(current_point_number)
                    # The point immediately after the initial point (used to detect loop termination)
                    next_initial_point = copy.deepcopy(current_point)
                    next_initial_point_number = copy.deepcopy(current_point_number)
                    # Next point placeholder
                    next_point = None
                    next_point_number = None
                    # Loop until the next point equals the post-initial point (cycle closed)
                    while next_point != next_initial_point:
                        # Collect all neighbor edges of the current point
                        relevant_edges = []
                        for edge in output_edges_c:
                            if edge[0] == current_point or edge[1] == current_point:
                                relevant_edges.append(edge)
                        # Compute outgoing angles of all neighbor edges from the current point
                        relevant_edges_degree = []
                        for relevant_edge in relevant_edges:
                            # Outgoing vector for this neighbor edge
                            vec = None
                            if relevant_edge[0] == current_point:
                                vec = (relevant_edge[1][0] - relevant_edge[0][0], relevant_edge[1][1] - relevant_edge[0][1])
                            elif relevant_edge[1] == current_point:
                                vec = (relevant_edge[0][0] - relevant_edge[1][0], relevant_edge[0][1] - relevant_edge[1][1])
                            else:
                                assert 0
                            # Compute outgoing angle
                            vec_degree = x_axis_angle(vec)
                            relevant_edges_degree.append(vec_degree)
                        # Compute the reverse incoming direction (outgoing direction) from last->current and its angle
                        vec_from_current_point_to_last_point = None
                        vec_from_current_point_to_last_point_degree = None
                        for relevant_edge_ind, relevant_edge in enumerate(relevant_edges):
                            if relevant_edge == (current_point, last_point):
                                vec_from_current_point_to_last_point = (relevant_edge[1][0] - relevant_edge[0][0], relevant_edge[1][1] - relevant_edge[0][1])
                                vec_from_current_point_to_last_point_degree = relevant_edges_degree[relevant_edge_ind]
                                relevant_edges.remove(relevant_edge)
                                relevant_edges_degree.remove(vec_from_current_point_to_last_point_degree)
                            elif relevant_edge == (last_point, current_point):
                                vec_from_current_point_to_last_point = (relevant_edge[0][0] - relevant_edge[1][0], relevant_edge[0][1] - relevant_edge[1][1])
                                vec_from_current_point_to_last_point_degree = relevant_edges_degree[relevant_edge_ind]
                                relevant_edges.remove(relevant_edge)
                                relevant_edges_degree.remove(vec_from_current_point_to_last_point_degree)
                            else:
                                continue
                        # Rotate counterclockwise from this angle, selecting the last encountered neighbor edge.
                        # Record unscanned angular regions (the interior angle) for semantic aggregation.
                        # Use the angular coverage and predicted semantic tags to derive the interior angle semantics.
                        rotate_deltas_counterclockwise = []
                        # Record interior angle region: counterclockwise from previous angle to next angle
                        interior_angles = []
                        for relevant_edge_degree in relevant_edges_degree:
                            rotate_delta = rotate_degree_counterclockwise_from_counter_degree(vec_from_current_point_to_last_point_degree, relevant_edge_degree)
                            rotate_deltas_counterclockwise.append(rotate_delta)
                            interior_angles.append((relevant_edge_degree, vec_from_current_point_to_last_point_degree))
                        # print(rotate_deltas_counterclockwise)
                        # Index of maximum counterclockwise rotation span
                        max_rotate_index = rotate_deltas_counterclockwise.index(max(rotate_deltas_counterclockwise))
                        # Interior angle (counterclockwise span) corresponding to that maximum
                        interior_angle_counterclockwise = interior_angles[max_rotate_index]
                        # Determine semantic region: gather all semantics of current point ordered by quadrants
                        current_point_semantic = [current_point[3], current_point[2], current_point[5], current_point[4]]
                        # Compute how much of each quadrant this counterclockwise span covers.
                        # Method: rotate from smaller to larger degree to get raw coverage.
                        # If smaller degree is the source direction: coverage is direct.
                        # If smaller is the target direction: subtract coverage from 90.
                        interior_angle_counterclockwise_degree_smaller = min(interior_angle_counterclockwise) # Smaller degree value
                        interior_angle_counterclockwise_degree_bigger = max(interior_angle_counterclockwise)  # Larger degree value
                        quadrant_smaller_to_bigger_counterclockwise = get_quadrant((interior_angle_counterclockwise_degree_smaller,
                                                                                    interior_angle_counterclockwise_degree_bigger))
                        # print(quadrant_smaller_to_bigger_counterclockwise)
                        if interior_angle_counterclockwise.index(interior_angle_counterclockwise_degree_smaller) == 0:
                            pass
                        elif interior_angle_counterclockwise.index(interior_angle_counterclockwise_degree_smaller) == 1:
                            quadrant_smaller_to_bigger_counterclockwise = (90 - quadrant_smaller_to_bigger_counterclockwise[0],
                                                                           90 - quadrant_smaller_to_bigger_counterclockwise[1],
                                                                           90 - quadrant_smaller_to_bigger_counterclockwise[2],
                                                                           90 - quadrant_smaller_to_bigger_counterclockwise[3])
                        else:
                            assert 0
                        # Invalidate semantics for quadrants with coverage < 45 degrees
                        current_point_semantic_valid = []
                        for qd, seman in enumerate(current_point_semantic):
                            if quadrant_smaller_to_bigger_counterclockwise[qd] >= 45:
                                current_point_semantic_valid.append(seman)
                            else:
                                current_point_semantic_valid.append(-1)
                        # Store valid interior angle semantics for later voting
                        simple_cycle_semantics.append(current_point_semantic_valid)

                        # Selected neighbor edge with max rotation span
                        max_rotate_edge = relevant_edges[max_rotate_index]
                        # Determine next point (opposite endpoint of selected edge)
                        if max_rotate_edge[0] == current_point:
                            next_point = max_rotate_edge[1]
                            next_point_number = d[next_point]
                        elif max_rotate_edge[1] == current_point:
                            next_point = max_rotate_edge[0]
                            next_point_number = d[next_point]
                        else:
                            assert 0
                        # Update last/current/next pointers and append current point to cycle
                        last_point = current_point
                        last_point_number = current_point_number
                        current_point = next_point
                        current_point_number = next_point_number
                        simple_cycle.append(current_point)
                        simple_cycle_number.append(current_point_number)
                    # Optionally add initial point again (not needed here) before edge removal
                    # simple_cycle.append(initial_point)
                    # simple_cycle_number.append(initial_point_number)
                    # Remove traversed directed edges (pi < p(i+1)) from traversal set to avoid duplicates
                    # print('------------------')
                    # print(simple_cycle)
                    # print(simple_cycle_number)
                    # print('------------------')
                    for point_number_ind, point_number in enumerate(simple_cycle_number):
                        if point_number_ind < len(simple_cycle_number) - 1:
                            edge_number = (point_number, simple_cycle_number[point_number_ind + 1])
                            # print(simple_cycle_number)
                            if edge_number[0] < edge_number[1]:
                                if (d_rev[edge_number[0]], d_rev[edge_number[1]]) in output_edges_c_copy_for_traversing:
                                    output_edges_c_copy_for_traversing.remove((d_rev[edge_number[0]], d_rev[edge_number[1]]))
                                elif (d_rev[edge_number[1]], d_rev[edge_number[0]]) in output_edges_c_copy_for_traversing:
                                    output_edges_c_copy_for_traversing.remove((d_rev[edge_number[1]], d_rev[edge_number[0]]))
                    # Remove duplicated last point; polygon area computation does not need closure here
                    simple_cycle.pop(-1)
                    simple_cycle_number.pop(-1)
                    # Keep cycle if counterclockwise area positive (clockwise => outermost / discard for semantics)
                    polygon_counterclockwise = [(int(p[0]), -int(p[1])) for p in simple_cycle]
                    polygon_counterclockwise.pop(-1)
                    # print('poly_area(polygon_counterclockwise)', poly_area(polygon_counterclockwise))
                    if poly_area(polygon_counterclockwise) > 0:
                    # if 1:
                        simple_cycles_c.append(simple_cycle)
                        simple_cycles_number_c.append(simple_cycle_number)
                        # Perform fractional voting (skip outermost cycle) to decide semantic label for this cycle
                        semantic_result = {}
                        for semantic_label in range(99998, 100000):
                            semantic_result[semantic_label] = 0
                        for everypoint_semantic in simple_cycle_semantics:
                            everypoint_semantic = [s for s in everypoint_semantic if s != -1]
                            for label in everypoint_semantic:
                                semantic_result[label] += 1 / len(everypoint_semantic)
                        # print(semantic_result)
                        # If top votes tie choose uniformly at random among them (labels 11 and 12 excluded)
                        this_cycle_semantic = sorted(semantic_result.items(), key=lambda d: d[1], reverse=True)
                        # print(this_cycle_semantic)
                        this_cycle_result = None
                        if this_cycle_semantic[0][1] > this_cycle_semantic[1][1]:
                            # Unique highest vote decides label
                            this_cycle_result = this_cycle_semantic[0][0]
                        else:
                            # Randomly sample among labels with equal highest vote count
                            this_cycle_results = [i[0] for i in this_cycle_semantic if i[1] == this_cycle_semantic[0][1]]
                            this_cycle_result = this_cycle_results[random.randint(0, len(this_cycle_results) - 1)]
                        # print(this_cycle_result)
                        simple_cycle_semantics_c.append(this_cycle_result)
                    if poly_area(polygon_counterclockwise) < 0: # Negative area => outermost enclosing cycle
                        # if 1:
                        simple_cycles_c.append(simple_cycle + ['max'])
                        simple_cycles_number_c.append(simple_cycle_number)
                        # Fractional voting again (outermost cycle semantics still computed for completeness)
                        semantic_result = {}
                        for semantic_label in range(99998, 100000):
                            semantic_result[semantic_label] = 0
                        for everypoint_semantic in simple_cycle_semantics:
                            everypoint_semantic = [s for s in everypoint_semantic if s != -1]
                            for label in everypoint_semantic:
                                semantic_result[label] += 1 / len(everypoint_semantic)
                        # print(semantic_result)
                        # If top votes tie choose uniformly at random (labels 11 and 12 excluded)
                        this_cycle_semantic = sorted(semantic_result.items(), key=lambda d: d[1], reverse=True)
                        # print(this_cycle_semantic)
                        this_cycle_result = None
                        if this_cycle_semantic[0][1] > this_cycle_semantic[1][1]:
                            # Unique highest vote decides label
                            this_cycle_result = this_cycle_semantic[0][0]
                        else:
                            # Randomly sample among labels with equal highest vote count
                            this_cycle_results = [i[0] for i in this_cycle_semantic if i[1] == this_cycle_semantic[0][1]]
                            this_cycle_result = this_cycle_results[random.randint(0, len(this_cycle_results) - 1)]
                        # print(this_cycle_result)
                        simple_cycle_semantics_c.append(this_cycle_result)

            simple_cycles.extend(simple_cycles_c)
            simple_cycles_number.extend(simple_cycles_number_c)
            simple_cycles_semantics.extend(simple_cycle_semantics_c)



    # print([[(int(j[0]), int(j[1])) for j in i] for i in simple_cycles])

    # print(len(simple_cycles_number))
    # print(simple_cycles_semantics)

    return d_rev, simple_cycles, simple_cycles_semantics

def get_cycle_basis_and_semantic_3_semansimplified(output_points_ori, output_edges_ori):
    output_points = [tuple(p + [99999, 99999, 99999, 99999]) for p in output_points_ori.tolist()]
    output_edges = [
        tuple([tuple([e[0], e[1]] + [99999, 99999, 99999, 99999]), tuple([e[2], e[3]] + [99999, 99999, 99999, 99999])])
        for e in output_edges_ori]

    output_points = copy.deepcopy(output_points)
    output_edges = copy.deepcopy(output_edges)
    # Dictionary relating indices to output points
    d = {}
    for output_point_index, output_point in enumerate(output_points):
        d[output_point] = output_point_index  # Cannot handle duplicate points here; keep NMS
    d_rev = {}
    for output_point_index, output_point in enumerate(output_points):
        d_rev[output_point_index] = output_point  # Cannot handle duplicate points here; keep NMS
    es = []
    for output_edge in output_edges:
        es.append((d[output_edge[0]], d[output_edge[1]]))


    G = nx.Graph()
    for e in es:
        G.add_edge(e[0], e[1])
        G.add_edge(e[1], e[0])


    simple_cycles = []
    simple_cycles_number = []
    simple_cycles_semantics = []
    # print('breakpoint1', simple_cycles)
    bridges = list(nx.bridges(G))
    # Handling virtual (bridge) edges: remove them from the edge set and treat remaining components independently
    for b in bridges:
        if (d_rev[b[0]], d_rev[b[1]]) in output_edges:
            output_edges.remove((d_rev[b[0]], d_rev[b[1]]))
            es.remove((b[0], b[1]))
            if G.has_edge(b[0], b[1]):
                G.remove_edge(b[0], b[1])
        if (d_rev[b[1]], d_rev[b[0]]) in output_edges:
            output_edges.remove((d_rev[b[1]], d_rev[b[0]]))
            es.remove((b[1], b[0]))
            if G.has_edge(b[1], b[0]):
                G.remove_edge(b[1], b[0])
    # After removing bridges, remaining connected components contain either isolated nodes or cycles; enumerate cycles
    connected_components = list(nx.connected_components(G))


    for c in connected_components:
        if len(c) == 1:
            pass
        else:
            simple_cycles_c = []
            simple_cycles_number_c = []
            simple_cycle_semantics_c = []
            # print(c) # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
            # Collect points and edges belonging to this connected component
            output_points_c = [p for p in output_points if d[p] in c]
            output_edges_c = [e for e in output_edges if d[e[0]] in c or d[e[1]] in c]  # Fixed edge set; not mutated
            output_edges_c_copy_for_traversing = copy.deepcopy(output_edges_c)  # Working edge set for traversal; edges removed to reduce complexity
            # print(output_points_c)
            # print(output_edges_c)

            # Method to enumerate all counterclockwise simple cycles in this component:
            # 1. Use index mapping d for component points.
            # 2. For each undirected edge, choose smaller index as start; treat larger as current.
            # 3. Find all neighbor edges of current and compute their outgoing angles.
            # 4. Determine reverse incoming direction from last to current (polar angle in [0,360)).
            # 5. Rotate counterclockwise from that angle to select the last encountered neighbor edge.
            # 6. Advance to its opposite endpoint.
            # 7. Repeat until returning to the second point (cycle closed).
            # 8. Remove traversed edges (pi < p(i+1)) from traversal set to avoid repeats.
            # 9. Continue with remaining edges.

            for edge_c in output_edges_c:
                if edge_c not in output_edges_c_copy_for_traversing:
                    pass
                else:
                    try:
                        simple_cycle_semantics = []
                        simple_cycle = []
                        simple_cycle_number = []
                        point1 = edge_c[0]
                        point2 = edge_c[1]
                        point1_number = d[point1]
                        point2_number = d[point2]
                        # Initial point
                        initial_point = None
                        initial_point_number = None
                        if point1_number < point2_number:
                            initial_point = point1
                            initial_point_number = point1_number
                        else:
                            initial_point = point2
                            initial_point_number = point2_number
                        simple_cycle.append(initial_point)
                        simple_cycle_number.append(initial_point_number)
                        # Previous point
                        last_point = initial_point
                        last_point_number = initial_point_number
                        # Current point
                        current_point = None
                        current_point_number = None
                        if point1_number < point2_number:
                            current_point = point2
                            current_point_number = point2_number
                        else:
                            current_point = point1
                            current_point_number = point1_number
                        simple_cycle.append(current_point)
                        simple_cycle_number.append(current_point_number)
                        # The point immediately after the initial point (used to detect while-loop termination)
                        next_initial_point = copy.deepcopy(current_point)
                        next_initial_point_number = copy.deepcopy(current_point_number)
                        # Placeholder for next point reference
                        next_point = None
                        next_point_number = None
                        # Loop terminates when next point equals the post-initial point
                        while_count = 0
                        while next_point != next_initial_point and while_count < 100:
                            # Collect all neighbor edges of the current point
                            relevant_edges = []
                            for edge in output_edges_c:
                                if (edge[0] == current_point or edge[1] == current_point) and (not (edge[0] == current_point and edge[1] == current_point)):
                                    relevant_edges.append(edge)
                            # Compute outgoing angles for each neighbor edge of current point
                            relevant_edges_degree = []
                            for relevant_edge in relevant_edges:
                                # Outgoing vector for this neighbor edge
                                vec = None
                                if relevant_edge[0] == current_point:
                                    vec = (
                                    relevant_edge[1][0] - relevant_edge[0][0], relevant_edge[1][1] - relevant_edge[0][1])
                                elif relevant_edge[1] == current_point:
                                    vec = (
                                    relevant_edge[0][0] - relevant_edge[1][0], relevant_edge[0][1] - relevant_edge[1][1])
                                else:
                                    assert 0
                                # Compute outgoing angle
                                vec_degree = x_axis_angle(vec)
                                relevant_edges_degree.append(vec_degree)
                            # Compute reverse incoming direction (outgoing direction) from last->current and its angle
                            vec_from_current_point_to_last_point = None
                            vec_from_current_point_to_last_point_degree = None
                            for relevant_edge_ind, relevant_edge in enumerate(relevant_edges):
                                if relevant_edge == (current_point, last_point):
                                    vec_from_current_point_to_last_point = (
                                    relevant_edge[1][0] - relevant_edge[0][0], relevant_edge[1][1] - relevant_edge[0][1])
                                    vec_from_current_point_to_last_point_degree = relevant_edges_degree[relevant_edge_ind]
                                    relevant_edges.remove(relevant_edge)
                                    relevant_edges_degree.remove(vec_from_current_point_to_last_point_degree)
                                elif relevant_edge == (last_point, current_point):
                                    vec_from_current_point_to_last_point = (
                                    relevant_edge[0][0] - relevant_edge[1][0], relevant_edge[0][1] - relevant_edge[1][1])
                                    vec_from_current_point_to_last_point_degree = relevant_edges_degree[relevant_edge_ind]
                                    relevant_edges.remove(relevant_edge)
                                    relevant_edges_degree.remove(vec_from_current_point_to_last_point_degree)
                                else:
                                    continue
                            # Rotate counterclockwise from this angle to the last encountered neighbor edge of current point
                            # Record the unscanned angular interval (the interior angle region)
                            # Here we replace interior-angle semantics with full semantics (no filtering)
                            rotate_deltas_counterclockwise = []
                            # Record interior angle region: counterclockwise from previous angle to next angle
                            interior_angles = []
                            for relevant_edge_degree in relevant_edges_degree:
                                rotate_delta = rotate_degree_counterclockwise_from_counter_degree(
                                    vec_from_current_point_to_last_point_degree, relevant_edge_degree)
                                rotate_deltas_counterclockwise.append(rotate_delta)
                                interior_angles.append((relevant_edge_degree, vec_from_current_point_to_last_point_degree))
                            # print(rotate_deltas_counterclockwise)
                            # Index of maximum counterclockwise rotation span
                            max_rotate_index = rotate_deltas_counterclockwise.index(max(rotate_deltas_counterclockwise))
                            # Interior angle (counterclockwise span) corresponding to that maximum
                            interior_angle_counterclockwise = interior_angles[max_rotate_index]
                            # Determine semantic region for interior angle
                            # Gather all semantics at current point ordered by quadrants
                            # current_point_semantic = [current_point[3], current_point[2], current_point[5],
                            #                           current_point[4], ]
                            current_point_semantic = [current_point[3], current_point[2], current_point[5],
                                                      current_point[4], current_point[6], current_point[7],
                                                      current_point[8]]
                            # Compute how many degrees of each quadrant the counterclockwise span covers
                            # Method: rotate from smaller degree to larger degree; span covers intervening quadrants
                            # If smaller degree is the source direction, the coverage is direct;
                            # If smaller degree is the target direction, subtract each covered quadrant span from 90
                            interior_angle_counterclockwise_degree_smaller = min(interior_angle_counterclockwise)  # Smaller degree value
                            interior_angle_counterclockwise_degree_bigger = max(interior_angle_counterclockwise)  # Larger degree value
                            quadrant_smaller_to_bigger_counterclockwise = get_quadrant(
                                (interior_angle_counterclockwise_degree_smaller,
                                 interior_angle_counterclockwise_degree_bigger))
                            # print(quadrant_smaller_to_bigger_counterclockwise)
                            if interior_angle_counterclockwise.index(interior_angle_counterclockwise_degree_smaller) == 0:
                                pass
                            elif interior_angle_counterclockwise.index(interior_angle_counterclockwise_degree_smaller) == 1:
                                quadrant_smaller_to_bigger_counterclockwise = (
                                90 - quadrant_smaller_to_bigger_counterclockwise[0],
                                90 - quadrant_smaller_to_bigger_counterclockwise[1],
                                90 - quadrant_smaller_to_bigger_counterclockwise[2],
                                90 - quadrant_smaller_to_bigger_counterclockwise[3])
                            else:
                                assert 0
                            # Never invalidate (set to -1) any semantic in this variant
                            current_point_semantic_valid = []
                            for qd, seman in enumerate(current_point_semantic):
                                if 1:
                                    current_point_semantic_valid.append(seman)
                                else:
                                    current_point_semantic_valid.append(-1)
                            # Aggregate all semantics (no filtering) for voting
                            simple_cycle_semantics.append(current_point_semantic_valid)

                            # Selected neighbor edge with max rotation span
                            max_rotate_edge = relevant_edges[max_rotate_index]
                            # Determine next point (opposite endpoint of selected edge)
                            if max_rotate_edge[0] == current_point:
                                next_point = max_rotate_edge[1]
                                next_point_number = d[next_point]
                            elif max_rotate_edge[1] == current_point:
                                next_point = max_rotate_edge[0]
                                next_point_number = d[next_point]
                            else:
                                assert 0
                            # Update last/current/next references and append current point to cycle
                            last_point = current_point
                            last_point_number = current_point_number
                            current_point = next_point
                            current_point_number = next_point_number
                            simple_cycle.append(current_point)
                            simple_cycle_number.append(current_point_number)
                            while_count += 1
                        if len(simple_cycle) > 80:
                            continue
                        # Optionally append initial point again before edge removal (not required here)
                        # simple_cycle.append(initial_point)
                        # simple_cycle_number.append(initial_point_number)
                        # Remove traversed directed edges (pi < p(i+1)) from traversal set to avoid duplicates
                        # print('------------------')
                        # print(simple_cycle)
                        # print(simple_cycle_number)
                        # print('------------------')
                        for point_number_ind, point_number in enumerate(simple_cycle_number):
                            if point_number_ind < len(simple_cycle_number) - 1:
                                edge_number = (point_number, simple_cycle_number[point_number_ind + 1])
                                # print(simple_cycle_number)
                                if edge_number[0] < edge_number[1]:
                                    if (d_rev[edge_number[0]], d_rev[edge_number[1]]) in output_edges_c_copy_for_traversing:
                                        output_edges_c_copy_for_traversing.remove(
                                            (d_rev[edge_number[0]], d_rev[edge_number[1]]))
                                    elif (
                                    d_rev[edge_number[1]], d_rev[edge_number[0]]) in output_edges_c_copy_for_traversing:
                                        output_edges_c_copy_for_traversing.remove(
                                            (d_rev[edge_number[1]], d_rev[edge_number[0]]))
                        # Area computation does not require explicit closure (last point duplicates first)
                        simple_cycle.pop(-1)
                        simple_cycle_number.pop(-1)
                        # Keep only cycles with positive (counterclockwise) area; negative => outermost enclosing loop
                        polygon_counterclockwise = [(int(p[0]), -int(p[1])) for p in simple_cycle]
                        polygon_counterclockwise.pop(-1)
                        # print('poly_area(polygon_counterclockwise)', poly_area(polygon_counterclockwise))
                        if poly_area(polygon_counterclockwise) > 0:
                            simple_cycles_c.append(simple_cycle)
                            simple_cycles_number_c.append(simple_cycle_number)
                            # Determine the common maximal semantic (skip the largest enclosing cycle); compute and record this simple_cycle's semantic label distribution
                            semantic_result = {}
                            for semantic_label in range(0, 7):
                                semantic_result[semantic_label] = 0
                            for everypoint_semantic in simple_cycle_semantics:
                                for _ in range(0, 7):
                                    if _ in everypoint_semantic:
                                        semantic_result[_] += 1
                            del semantic_result[6]

                            # print(semantic_result)
                            # If the highest vote counts tie choose uniformly at random among them (note label 13 is ignored if present)
                            this_cycle_semantic = sorted(semantic_result.items(), key=lambda d: d[1], reverse=True)
                            # print(this_cycle_semantic)
                            if this_cycle_semantic[0][1] > this_cycle_semantic[1][1]:
                                # Use the single highest vote directly
                                this_cycle_result = this_cycle_semantic[0][0]
                            else:
                                # Among all labels with the highest vote count apply priority: cabinet 2 > bathroom 4 > kitchen 3 > bedroom 1 > balcony 5 > living room 0
                                this_cycle_results = [i[0] for i in this_cycle_semantic if
                                                      i[1] == this_cycle_semantic[0][1]]
                                if 2 in this_cycle_results:
                                    this_cycle_result = 2
                                elif 4 in this_cycle_results:
                                    this_cycle_result = 4
                                elif 3 in this_cycle_results:
                                    this_cycle_result = 3
                                elif 1 in this_cycle_results:
                                    this_cycle_result = 1
                                elif 5 in this_cycle_results:
                                    this_cycle_result = 5
                                else:
                                    this_cycle_result = 0
                            # print(this_cycle_result)
                            simple_cycle_semantics_c.append(this_cycle_result)
                    except:
                        pass

            simple_cycles.extend(simple_cycles_c)
            simple_cycles_number.extend(simple_cycles_number_c)
            simple_cycles_semantics.extend(simple_cycle_semantics_c)

    # print([[(int(j[0]), int(j[1])) for j in i] for i in simple_cycles])

    # print(len(simple_cycles_number))
    # print(simple_cycles_semantics)

    return d_rev, simple_cycles, simple_cycles_semantics



def deep_compare(a, b):
    '''Return True if two nested data structures (dict/list/tuple/ndarray/scalars) are exactly identical.'''
    # If types differ return False immediately
    if type(a) != type(b):
        return False

    # If dict: compare keys and then recurse on each value
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(deep_compare(a[key], b[key]) for key in a)

    # If list or tuple: compare length then each element
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(deep_compare(item1, item2) for item1, item2 in zip(a, b))

    # If numpy array: use np.array_equal
    elif isinstance(a, np.ndarray):
        return np.array_equal(a, b)

    # Other scalar types compare directly (int, float, str, etc.)
    else:
        return a == b

def check_subdir_file_counts(base_dir):
    subdirs = ['train', 'val', 'test']
    for subdir in subdirs:
        full_path = os.path.join(base_dir, subdir)
        file_list = [f for f in os.listdir(full_path)
                     if os.path.isfile(os.path.join(full_path, f))]
    print('file count ' + str(len(file_list)))


# Record accumulator
ccccc = []


# Previously extracted structure map
structure_graphs = np.load('./rplandata/Data/structure_graphs.npy', allow_pickle=True).item()
image_path = r'rplandata/Data/floorplan_dataset'
# print(len(os.listdir(image_path)))

os.mkdir('./rplandata/Data/rplang-v3-withsemantics')
os.mkdir('./rplandata/Data/rplang-v3-withsemantics/train')
os.mkdir('./rplandata/Data/rplang-v3-withsemantics/val')
os.mkdir('./rplandata/Data/rplang-v3-withsemantics/test')

from tiny_graph import train as tr
from tiny_graph import val as va
from tiny_graph import test as te

train_fnids = [int(fnid[:-4]) for fnid in tr]
val_fnids = [int(fnid[:-4]) for fnid in va]
test_fnids = [int(fnid[:-4]) for fnid in te]


# Iterate over 71,763 file_ids; perform dataset split afterwards
for file_id, structure_graph in tqdm(structure_graphs.items()):
    # if file_id == 10030:
    if 1:
        g = {}
        # print('file_id', file_id)
        g['file_id'] = file_id # √
        
        '''original corner data'''
        corners = structure_graph['corners']
        # print('corners', corners)
        g['corners'] = corners # √
        
        adjacency_matrix = structure_graph['adjacency_matrix']
        # print('adjacency_matrix', adjacency_matrix)
        g['adjacency_matrix'] = adjacency_matrix # √
        
        adjacency_list = structure_graph['adjacency_list']
        # print('adjacency_list', adjacency_list)
        g['adjacency_list'] = adjacency_list # √
        
        '''convert to ndarray'''
        corners_np = np.array([list(_) for _ in corners], dtype=np.float64)
        # print('corners_np', corners_np)
        g['corners_np'] = corners_np # √
        
        adjacency_matrix_np = np.array(adjacency_matrix, dtype=np.uint8)
        # print('adjacency_matrix_np', adjacency_matrix_np)
        g['adjacency_matrix_np'] = adjacency_matrix_np
        
        adjacency_list_np = np.array(adjacency_list, dtype=np.uint8)
        # print('adjacency_list_np', adjacency_list_np)
        g['adjacency_list_np'] = adjacency_list_np # √
        
        '''normalization (coords: [0, 255] -> [-1, 1])
           if rescale roi(bounding box) to [-1, 1], layouts' large/tiny area will be damaged,
           we want to generate both tiny and large (rational area) layout,
           all rois are already in center(if not in center, learning roi biases on canvas is irrational)'''
        corner_list_np_normalized = (corners_np - 128) / 128
        # print('corner_list_np_normalized', corner_list_np_normalized)
        g['corner_list_np_normalized'] = corner_list_np_normalized # √
        
        '''padding and attn mask generating.
           1 means compute and 0 means padding.
           padding to 53 because max corner number is 53; (this is rational!! more paddings make no sense.)
           we don't use 100 because 100*100 edges are too large, about 4 times to 53*53'''
        padding_to_number = 53
        
        '''after-padding corner lists'''
        corner_list_np_normalized_padding = np.zeros((padding_to_number, 2), dtype=np.float64)
        corner_list_np_normalized_padding[:len(corner_list_np_normalized), :] = corner_list_np_normalized
        # print('corner_list_np_normalized_padding', corner_list_np_normalized_padding)
        g['corner_list_np_normalized_padding'] = corner_list_np_normalized_padding # √
        
        ''' padding mask, only real corners (<=> not padding/virtual corners) should compute loss
            we compute it in advance, to save time'''
        padding_mask = np.zeros((padding_to_number, 1), dtype=np.uint8)
        padding_mask[:len(corner_list_np_normalized), :] = 1
        # print('padding_mask', padding_mask)
        g['padding_mask'] = padding_mask # √
        
        '''global matrix, (53, 53), each pair of nodes except padding nodes (attention type 1: every)'''
        global_matrix_np_padding = np.zeros((padding_to_number, padding_to_number), dtype=np.uint8)
        global_matrix_np_padding[:len(corner_list_np_normalized), :len(corner_list_np_normalized)] = 1
        # print('global_matrix_np_padding', global_matrix_np_padding)
        g['global_matrix_np_padding'] = global_matrix_np_padding # √
        
        '''adjacency matrix, (53, 53), only edge exists == 1 (attention type 2: adjacent)'''
        adjacency_matrix_np_padding = np.zeros((padding_to_number, padding_to_number), dtype=np.uint8)
        adjacency_matrix_np_padding[:len(adjacency_matrix_np), :len(adjacency_matrix_np)] = adjacency_matrix_np
        # print('adjacency_matrix_np_padding', adjacency_matrix_np_padding)
        g['adjacency_matrix_np_padding'] = adjacency_matrix_np_padding # √
        
        '''to prepare edge data. we need 53*53 edges, each is [(x1, y1), (x2, y2), 0/1]
        (in accordance with convention in matrix theory, (x1, y1) for row, (x2, y2) for column)
        if padding corner, (xi, yi) can be any value ((0, 0) for convenience) as they contribute 0 for attn.'''
        edge_coord1 = np.repeat(corner_list_np_normalized_padding[:, None, :], padding_to_number, axis=1)
        edge_coord2 = np.repeat(corner_list_np_normalized_padding[None, :, :], padding_to_number, axis=0)
        edge_coords = np.concatenate((edge_coord1, edge_coord2), axis=2).reshape(-1, 4)
        # print('edge_coords', edge_coords)
        g['edge_coords'] = edge_coords # √
        
        '''edges'''
        edges = adjacency_matrix_np_padding[:, :, None].reshape(-1, 1)
        # print('edges', edges)
        g['edges'] = edges # √






        '''Begin extracting semantics field and corner_list_np_normalized_padding_withsemantics field'''
        # read image
        image = cv2.imread(os.path.join(image_path, str(file_id) + '.png'), cv2.IMREAD_UNCHANGED) # 4 channels
        
        '''Randomly sampled four-channel pixels
            Pixel 1: [ 0 13  0  0]
            Pixel 2: [  3   0   0 255]
            Pixel 3: [  3   0   0 255]
            Pixel 4: [  3   0   0 255]
            Pixel 5: [ 0 13  0  0]
            Pixel 6: [  4   7   0 255]
            Pixel 7: [ 0 13  0  0]
            Pixel 8: [ 0 13  0  0]
            Pixel 9: [  3   0   0 255]
            Pixel 10: [ 0 13  0  0]
            ...
        label.xlsx  channels order: 3 2 1 4
        '''

        # Code cleanup: unify variable names
        graph = g
        corners_sss = graph['corners_np']
        adjacency_matrix_sss = graph['adjacency_matrix_np']
        assert np.all(adjacency_matrix_sss == adjacency_matrix_sss.T)
        # Extract all edges
        edges_list = extract_edges(corners_sss, adjacency_matrix_sss)
        assert len(edges_list) == np.sum(np.triu(adjacency_matrix_sss))
    
        # # Try visualizing these edges (random colors on 3-channel bin_img)
        channel_2 = copy.deepcopy(image[:, :, 1]) # Extract second channel (index 1)
    # thresholded_channel_2 = np.where(channel_2 >= 14, 255, 0) # Set pixels >=14 to 255 else 0
    # three_channel_img = np.stack((thresholded_channel_2, thresholded_channel_2, thresholded_channel_2), axis=-1).astype(np.uint8) # Replicate thresholded_channel_2 into 3-channel image

        d_rev, simple_cycles_, simple_cycles_semantics = get_cycle_basis_and_semantic(corners_sss, edges_list)
        # d_rev, simple_cycles_, simple_cycles_semantics = get_cycle_basis_and_semantic_3_semansimplified(corners_sss, edges_list)
        
        simple_cycles = []
        for sc_ in simple_cycles_:
            if sc_[-1] != 'max':
                simple_cycles.append([[int(_) for _ in list(t)[:2]] for t in sc_])
            else:
                simple_cycles.append([[int(_) for _ in list(t)[:2]] for t in sc_[:-1]] + ['max'])
        # for sc in simple_cycles:
        #     print(sc)
    
        '''Verify polygon extraction count: compare connected component count in bin_imgs channel with number of extracted polygons; if mismatch we may need to discard some samples.'''
        # Count connected components in bin_imgs channel
        true_polygon_number = count_connected_components(channel_2)
        # Number of extracted polygons
        r2g_obtain_polygon_number = len(simple_cycles)
        # if not true_polygon_number == r2g_obtain_polygon_number:
    #     print(str(sample_id)) # Confirmed r2g polygon extraction is completely correct
    
        # Next: determine semantic label for each polygon; note the largest cycle is directly assigned label 13.
        polygon_semantics = []
        for polygon in simple_cycles:
            if polygon[-1] != 'max':
                points_and_pixel_values = get_points_and_pixel_values_inside_polygon(copy.deepcopy(image[:, :, 1]), polygon)
                lbl = get_label(points_and_pixel_values)
                polygon_semantics.append((polygon[:-1], lbl))
            else:
                polygon_semantics.append((polygon[:-2], 13))
        # print(polygon_semantics)
        # assert 0
        # Corner dictionary: assign each corner a 14-dim (0-13) vector initialized to zeros counting room occurrences
        seman_d = {}
        for corner_sss in corners_sss:
            corner_tupleint_sss = tuple(corner_sss.astype(np.int32).tolist())
            # print(corner_sss, corner_tupleint_sss)
            seman_d[corner_tupleint_sss] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for polygon_semantic in polygon_semantics:
            polygon = polygon_semantic[0]
            semantic = polygon_semantic[1]
            # print(polygon)
            # print(semantic)
            for point in polygon:
                # seman_d[((tuple(point)[0] - 128) / 128, (tuple(point)[1] - 128) / 128)][semantic] += 1
                seman_d[tuple(point)][semantic] += 1
        normalized_seman_d = {}
        for k, v in seman_d.items():
            normalized_seman_d[((k[0] - 128) / 128, (k[1] - 128) / 128)] = v



        corner_list = graph['corner_list_np_normalized_padding']
        new_semantics = normalized_seman_d
        # Create new empty array shape (53, 16)
        result = np.zeros((53, 16), dtype=corner_list.dtype)
        # Iterate through 'corner_list'
        for idx, coord in enumerate(corner_list):
            # Convert coordinate tuple to original precision
            coord_tuple = (coord[0], coord[1])
            # print(coord_tuple)
            # print(new_semantics)
            # Check if coordinate exists in 'semantics' dict
            if coord_tuple in new_semantics:
                vector = new_semantics[coord_tuple]
            else:
                if idx < len(new_semantics):
                    assert 0
                vector = [0] * 14  # If absent, create all-zero vector
            # Concatenate coordinate with semantics vector
            result[idx] = np.concatenate((coord, vector))
        # print(result)
    

    
        ''' write'''
        new_graph = copy.deepcopy(graph)
        # new_graph['semantics'] = seman_d
        new_graph['semantics'] = normalized_seman_d
        new_graph['corner_list_np_normalized_padding_withsemantics'] = result
        # print(new_graph)

        if file_id in train_fnids:
            np.save('./rplandata/Data/rplang-v3-withsemantics/train/' + str(file_id) + '.npy', new_graph)
        elif file_id in val_fnids:
            np.save('./rplandata/Data/rplang-v3-withsemantics/val/' + str(file_id) + '.npy', new_graph)
        elif file_id in test_fnids:
            np.save('./rplandata/Data/rplang-v3-withsemantics/test/' + str(file_id) + '.npy', new_graph)
        else:
            assert 0
# Check file counts in the three subdirectories under rplandata/Data/rplang-v3-withsemantics
check_subdir_file_counts('./rplandata/Data/rplang-v3-withsemantics')
