import numpy as np
import basic_image_operations as bio

def cut_boxes(im_bw):
    """
    Locates effect boxes and process boxes on the b&w
    screen shot of the Cures screen.
    """
    # find triangles
    kernel = np.load('./kernels/triangle.npy')
    th = 950
    triangle_centers = bio.find_shapes(im_bw, kernel, th)
    triangle_centers.sort()

    # group triangles into columns
    columns_of_triangles = []
    columns_of_triangles.append([triangle_centers[0]])
    for triangle in triangle_centers[1:]:
        if np.abs(columns_of_triangles[-1][0][0] - triangle[0]) < 50:
            columns_of_triangles[-1].append(triangle)
        else:
            columns_of_triangles.append([triangle])

    drug_box_height = 63
    process_box_height = 125
    box_width = 290
    padding = 15
    last_padding = 22

    drug_box_gap = drug_box_height + 2 * padding
    process_box_gap = process_box_height + 2 * padding

    columns_of_box_identities = []  # 'drug' or 'process'
    columns_of_box_coordinates = []  # (x1, x2, y1, y2)
    for idx, col in enumerate(columns_of_triangles):
        if len(col) < 2:
            print('Warning: missing triangles in column ' + str(idx))
            break

        box_identities = []
        box_coordinates = []
        columns_of_box_identities.append(box_identities)
        columns_of_box_coordinates.append(box_coordinates)
        for i in range(len(col) - 1):

            # get box
            y1 = col[i][1] + padding
            y2 = col[i + 1][1] - padding
            x1 = col[i][0] - box_width // 2
            x2 = col[i][0] + box_width // 2
            box_coordinates.append((x1, x2, y1, y2))

            gap = col[i + 1][1] - col[i][1]
            if abs(gap - drug_box_gap) < abs(gap - process_box_gap):
                box_identities.append('cure')
            else:
                box_identities.append('process')

        # first box identity revision
        if box_identities[0] == 'process':
            box_identities.insert(0, 'cure')
        else:
            box_identities.insert(0, 'process')

        # last box identity revision
        if box_identities[-1] == 'process':
            box_identities.append('cure')
        else:
            box_identities.append('process')

        # first box coordinate revision
        if box_identities[0] == 'cure':
            box_height = drug_box_height
        elif box_identities[0] == 'process':
            box_height = process_box_height
        x, y = col[0]
        y1 = y - padding - drug_box_height
        y2 = y - padding
        x1 = x - box_width // 2
        x2 = x + box_width // 2
        box_coordinates.insert(0, (x1, x2, y1, y2))

        # last box coordinate revision
        if box_identities[-1] == 'cure':
            box_height = drug_box_height
        elif box_identities[-1] == 'process':
            box_height = process_box_height
        x, y = col[-1]
        y1 = y + last_padding
        y2 = y + last_padding + drug_box_height
        x1 = x - box_width // 2
        x2 = x + box_width // 2
        box_coordinates.append((x1, x2, y1, y2))

    # cut from image
    columns_of_boxes = []
    for box_coordinates in columns_of_box_coordinates:
        boxes = []
        columns_of_boxes.append(boxes)
        for box_coord in box_coordinates:
            x1, x2, y1, y2 = box_coord
            boxes.append(im_bw[int(y1):int(y2), int(x1):int(x2)])

    return columns_of_boxes, columns_of_box_identities
