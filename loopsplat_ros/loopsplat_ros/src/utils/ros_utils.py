import numpy as np
import torch
from std_msgs.msg import Float32MultiArray, Int32MultiArray, MultiArrayDimension

def convert_ros_multi_array_message_to_numpy(ros_multiarray_msg):

    num_dims = len(ros_multiarray_msg.layout.dim)
    if num_dims == 0:
        return None

    # Extract the dimensions from the layout
    dim0 = ros_multiarray_msg.layout.dim[0].size
    dim1 = ros_multiarray_msg.layout.dim[1].size
    if num_dims == 2:
        dims = (dim0, dim1)
    else:
        dim2 = ros_multiarray_msg.layout.dim[2].size
        dims = (dim0, dim1, dim2)

    # Convert the flat array back into a 2D numpy array
    if isinstance(ros_multiarray_msg, Float32MultiArray):
        data = np.array(ros_multiarray_msg.data, dtype=np.float32).reshape(dims)
    elif isinstance(ros_multiarray_msg, Int32MultiArray):
        data = np.array(ros_multiarray_msg.data, dtype=np.int32).reshape(dims)

    return data

def convert_ros_multi_array_message_to_tensor(ros_multiarray_msg, device):

    num_dims = len(ros_multiarray_msg.layout.dim)
    if num_dims == 0:
        return None

    # Extract the dimensions from the layout
    dim0 = ros_multiarray_msg.layout.dim[0].size
    dim1 = ros_multiarray_msg.layout.dim[1].size
    if num_dims == 2:
        dims = (dim0, dim1)
    else:
        dim2 = ros_multiarray_msg.layout.dim[2].size
        dims = (dim0, dim1, dim2)

    # Convert the flat array back into a 2D numpy array
    if isinstance(ros_multiarray_msg, Float32MultiArray):
        data = torch.reshape(torch.tensor(ros_multiarray_msg.data, dtype=torch.float32, device=device), dims)
    elif isinstance(ros_multiarray_msg, Int32MultiArray):
        data = torch.reshape(torch.tensor(ros_multiarray_msg.data, dtype=torch.int32, device=device), dims)

    return data

def convert_ros_array_message_to_tensor(ros_array_msg, device):

    if all(isinstance(i, int) for i in ros_array_msg):
        data = torch.tensor(ros_array_msg, dtype=torch.int32, device=device)
    elif all(isinstance(i, float) for i in ros_array_msg):
        data = torch.tensor(ros_array_msg, dtype=torch.float32, device=device)

    return data

def convert_tensor_to_ros_message(tensor_msg):
    if tensor_msg.dtype == torch.float32 or tensor_msg.dtype == torch.float64:
        ros_multiarray_msg = Float32MultiArray()
    elif tensor_msg.dtype == torch.int32 or tensor_msg.dtype == torch.int64:
        ros_multiarray_msg = Int32MultiArray()

    # If empty tensor
    if tensor_msg.shape[0] == 0:
        return ros_multiarray_msg

    num_dims = len(tensor_msg.size())

    if num_dims == 2:
        # Define the layout of the array
        dim0 = MultiArrayDimension()
        dim0.label = "dim0"
        dim0.size = tensor_msg.shape[0]
        dim0.stride = tensor_msg.shape[1] # num of columns per row

        dim1 = MultiArrayDimension()
        dim1.label = "dim1"
        dim1.size = tensor_msg.shape[1]
        dim1.stride = 1
        ros_multiarray_msg.layout.dim = [dim0, dim1]

    elif num_dims == 3:
        # Define the layout of the array
        dim0 = MultiArrayDimension()
        dim0.label = "dim0"
        dim0.size = tensor_msg.shape[0]
        dim0.stride = tensor_msg.shape[1]*tensor_msg.shape[2] # num of columns per row

        dim1 = MultiArrayDimension()
        dim1.label = "dim1"
        dim1.size = tensor_msg.shape[1]
        dim1.stride = tensor_msg.shape[2]

        dim2 = MultiArrayDimension()
        dim2.label = "dim1"
        dim2.size = tensor_msg.shape[2]
        dim2.stride = 1
        ros_multiarray_msg.layout.dim = [dim0, dim1, dim2]
    elif num_dims > 3:
        print("#Dimensions is > 3")

    ros_multiarray_msg.layout.data_offset = 0

    # Flatten the data and assign it to the message
    ros_multiarray_msg.data = tensor_msg.flatten().tolist()

    return ros_multiarray_msg


def convert_numpy_array_to_ros_message(np_arr):
    if np_arr.dtype == np.float32 or np_arr.dtype == np.float64:
        ros_multiarray_msg = Float32MultiArray()
    elif np_arr.dtype == np.int32 or np_arr.dtype == np.int32:
        ros_multiarray_msg = Int32MultiArray()

    # If empty tensor
    if np_arr.shape[0] == 0:
        return ros_multiarray_msg

    num_dims = np_arr.ndim

    if num_dims == 2:
        # Define the layout of the array
        dim0 = MultiArrayDimension()
        dim0.label = "dim0"
        dim0.size = np_arr.shape[0]
        dim0.stride = np_arr.shape[1] # num of columns per row

        dim1 = MultiArrayDimension()
        dim1.label = "dim1"
        dim1.size = np_arr.shape[1]
        dim1.stride = 1
        ros_multiarray_msg.layout.dim = [dim0, dim1]

    elif num_dims == 3:
        # Define the layout of the array
        dim0 = MultiArrayDimension()
        dim0.label = "dim0"
        dim0.size = np_arr.shape[0]
        dim0.stride = np_arr.shape[1]*np_arr.shape[2] # num of columns per row

        dim1 = MultiArrayDimension()
        dim1.label = "dim1"
        dim1.size = np_arr.shape[1]
        dim1.stride = np_arr.shape[2]

        dim2 = MultiArrayDimension()
        dim2.label = "dim1"
        dim2.size = np_arr.shape[2]
        dim2.stride = 1
        ros_multiarray_msg.layout.dim = [dim0, dim1, dim2]
    elif num_dims > 3:
        print("#Dimensions is > 3")

    ros_multiarray_msg.layout.data_offset = 0

    # Flatten the data and assign it to the message
    ros_multiarray_msg.data = np_arr.flatten().tolist()

    return ros_multiarray_msg