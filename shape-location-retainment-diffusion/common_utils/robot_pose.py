import numpy as np

def read_pose_from_file(path):
    pose_array_list = []

    # Read the content of the file
    with open(path, 'r') as file:
        for line in file:
            values = line.strip().split()  # Split the line into columns
        
            if len(values) >= 6:
                pose_array = np.array(values[:6], dtype=float)  # Convert to NumPy array
                pose_array_list.append(pose_array)
    
#     print(pose_array_list[0])

#     print(f'==============={pose_array_list[0] - pose_array_list[1]}')

    return pose_array_list