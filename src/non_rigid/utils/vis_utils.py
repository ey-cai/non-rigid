import rpad.visualize_3d.plots as rvpl
import gif
import numpy as np
import plotly.graph_objects as go
import rpad.visualize_3d.primitives as rvpr
import torch

import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image

class FlowNetAnimation:
    def __init__(self):
        self.num_frames = 0
        self.traces = {}
        self.gif_frames = []

    def add_trace(self, pos, flow_pos, flows, flowcolor):
        pcd = rvpl.pointcloud(pos, downsample=1)
        try:
            ts = rvpl._flow_traces(
                flow_pos[0],
                flows[0],
                # 0.1,
                1,
                scene="scene1",
                flowcolor=flowcolor,
                name="pred_flow",
            )
            self.traces[self.num_frames] = [pcd]
            for t in ts:
                self.traces[self.num_frames].append(t)
        except IndexError as e:
            print(f'Failed to add trace for frame {self.num_frames}. Error: {e}')
            return

        # self.traces[self.num_frames] = pcd
        self.num_frames += 1
        if self.num_frames == 1 or self.num_frames == 0:
            self.pos = pos
            self.ts = ts

    @gif.frame
    def add_trace_gif(self, pos, flow_pos, flows, flowcolor):
        f = go.Figure()
        f.add_trace(rvpr.pointcloud(pos, downsample=1, scene="scene1"))
        ts = rvpl._flow_traces(
            flow_pos[0],
            flows[0],
            scene="scene1",
            flowcolor=flowcolor,
            name="pred_flow",
        )
        for t in ts:
            f.add_trace(t)
        f.update_layout(scene1=rvpl._3d_scene(pos))
        return f

    def append_gif_frame(self, f):
        self.gif_frames.append(f)

    def show_animation(self):
        self.fig = go.Figure(
            frames=[
                go.Frame(data=self.traces[k], name=str(k))
                for k in range(self.num_frames)
            ]
        )

    def frame_args(self, duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    def set_sliders(self):
        self.sliders = [
            {
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], self.frame_args(0)],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(self.fig.frames)
                ],
            }
        ]
        return self.sliders

    @gif.frame
    def save_gif(self, dir):
        gif.save(self.gif_frames, dir, duration=100)

    def animate(self):
        self.show_animation()
        if self.num_frames == 0:
            return None
        k = np.random.permutation(np.arange(self.pos.shape[0]))[:500]
        self.fig.add_trace(rvpr.pointcloud(self.pos[k], downsample=1))
        for t in self.ts:
            self.fig.add_trace(t)
        # Layout
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.5, y=0, z=1.5),
        )
        self.fig.update_layout(
            title="Flow Prediction",
            scene_camera=camera,
            width=900,
            height=900,
            template="plotly_white",
            scene1=rvpl._3d_scene(self.pos),#, domain_scale=1.5),
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [None, self.frame_args(50)],
                            "label": "&#9654;",  # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], self.frame_args(0)],
                            "label": "&#9724;",  # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=self.set_sliders(),
        )
        # self.fig.show()
        return self.fig

def get_color(tensor_list, color_list, axis=False):
    """
    @param tensor_list: list of tensors of shape (num_points, 3)
    @param color_list: list of strings of color names that should be within color_scheme.keys(), eg, 'red', 'blue' 
    @return points_color_stacked: numpy array of shape (num_points*len(tensor_list), 6)
    """

    color_scheme = {}
    color_scheme['blue'] = np.array([68, 119, 170])
    color_scheme['cyan'] = np.array([102, 204, 238])
    color_scheme['green'] = np.array([34, 136, 51])
    color_scheme['yellow'] = np.array([204, 187, 68])
    color_scheme['red'] = np.array([238, 102, 119])
    color_scheme['purple'] = np.array([170, 51, 119])
    color_scheme['orange'] = np.array([238, 119, 51])

    stacked_list = []
    assert len(tensor_list) == len(
        color_list), 'len(tensor_list) should match len(color_list)'
    for i in range(len(tensor_list)):
        tensor = tensor_list[i].detach().cpu().numpy()
        color = color_list[i]
        assert len(
            tensor.shape) == 2, 'tensor should be of shape of (num_points, 3)'
        assert tensor.shape[-1] == 3, 'tensor.shape[-1] should be of shape 3, for point coordinates'
        assert color in color_scheme.keys(
        ), 'passed in color {} is not in the available color scheme, go to utils/color_utils.py to add any'.format(color)

        color_tensor = torch.from_numpy(
            color_scheme[color]).unsqueeze(0)  # 1,3
        N = tensor.shape[0]
        color_tensor = torch.repeat_interleave(
            color_tensor, N, dim=0).detach().cpu().numpy()  # N,3

        points_color = np.concatenate(
            (tensor, color_tensor), axis=-1)  # num_points, 6

        stacked_list.append(points_color)
    points_color_stacked = np.concatenate(
        stacked_list, axis=0)  # num_points*len(tensor_list), 6
    if axis:
        axis_pts = create_axis()
        points_color_stacked = np.concatenate(
            [points_color_stacked, axis_pts], axis=0)

    return points_color_stacked


def create_axis(length=1.0, num_points=100):
    pts = np.linspace(0, length, num_points)
    x_axis_pts = np.stack(
        [pts, np.zeros(num_points), np.zeros(num_points)], axis=1)
    y_axis_pts = np.stack(
        [np.zeros(num_points), pts, np.zeros(num_points)], axis=1)
    z_axis_pts = np.stack(
        [np.zeros(num_points), np.zeros(num_points), pts], axis=1)

    x_axis_clr = np.tile([255, 0, 0], [num_points, 1])
    y_axis_clr = np.tile([0, 255, 0], [num_points, 1])
    z_axis_clr = np.tile([0, 0, 255], [num_points, 1])

    x_axis = np.concatenate([x_axis_pts, x_axis_clr], axis=1)
    y_axis = np.concatenate([y_axis_pts, y_axis_clr], axis=1)
    z_axis = np.concatenate([z_axis_pts, z_axis_clr], axis=1)

    pts = np.concatenate([x_axis, y_axis, z_axis], axis=0)

    return pts


def toDisplay(x, target_dim = None):
    while(target_dim is not None and x.dim() > target_dim):
        x = x[0]
    return x.detach().cpu().numpy()


def plot_multi_np(plist):
    """
    Args: plist, list of numpy arrays of shape, (1,num_points,3)
    """
    colors = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#e377c2',  # raspberry yogurt pink
        '#8c564b',  # chestnut brown
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf',  # blue-teal
    ] * ((len(plist) // 10) + 1)
    skip = 1
    go_data = []
    for i in range(len(plist)):
        p_dp = toDisplay(torch.from_numpy(plist[i]))
        plot = go.Scatter3d(x=p_dp[::skip,0], y=p_dp[::skip,1], z=p_dp[::skip,2], 
                     mode='markers', marker=dict(size=2, color=colors[i],
                     symbol='circle'))
        go_data.append(plot)
 
    layout = go.Layout(
        scene=dict(
            aspectmode='data',
        ),
        height=800,
        width=800,
    )

    fig = go.Figure(data=go_data, layout=layout)
    fig.show()
    return fig



CAM_PROJECTION = (1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, -1.0000200271606445, -1.0,
                    0.0, 0.0, -0.02000020071864128, 0.0)
CAM_VIEW = (0.9396926164627075, 0.14454397559165955, -0.3099755346775055, 0.0, 
            -0.342020183801651, 0.3971312642097473, -0.8516507148742676, 0.0, 
            7.450580596923828e-09, 0.9063077569007874, 0.4226182699203491, 0.0, 
            0.5810889005661011, -4.983892917633057, -22.852874755859375, 1.0)

CAM_WIDTH = 500
CAM_HEIGHT = 500

def camera_project(point,
                   projectionMatrix=CAM_PROJECTION,
                   viewMatrix=CAM_VIEW,
                   height=CAM_HEIGHT,
                   width=CAM_WIDTH):
    """
    Projects a world point in homogeneous coordinates to pixel coordinates
    Args
        point: np.array of shape (N, 3); indicates the desired point to project
    Output
        (x, y): np.array of shape (N, 2); pixel coordinates of the projected point
    """
    point = np.concatenate([point, np.ones_like(point[:, :1])], axis=-1) # N x 4

    # reshape to get homogeneus transform
    persp_m = np.array(projectionMatrix).reshape((4, 4)).T
    view_m = np.array(viewMatrix).reshape((4, 4)).T

    # Perspective proj matrix
    world_pix_tran = persp_m @ view_m @ point.T
    world_pix_tran = world_pix_tran / world_pix_tran[-1]  # divide by w
    world_pix_tran[:3] = (world_pix_tran[:3] + 1) / 2
    x, y = world_pix_tran[0] * width, (1 - world_pix_tran[1]) * height
    x, y = np.floor(x).astype(int), np.floor(y).astype(int)

    return np.stack([x, y], axis=1)

def plot_diffusion(img, results, viewmat, color_key):
    # https://stackoverflow.com/questions/34768717/matplotlib-unable-to-save-image-in-same-resolution-as-original-image
    diffusion_frames = []

    for res in results:
        res_cam = camera_project(res, viewMatrix=viewmat)

        dpi = rcParams['figure.dpi']
        img = np.array(img)
        height, width, _ = img.shape
        figsize = width / dpi, height / dpi

        # creating figure of exact image size
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(img)

        # clipping points that are outside of camera frame
        filter = (res_cam[:, 0] >= 0) & (res_cam[:, 0] < height) & (res_cam[:, 1] >= 0) & (res_cam[:, 1] < width)
        res_cam = res_cam[filter]
        color_key_cam = color_key[filter]

        # plotting points
        ax.scatter(res_cam[:, 0], res_cam[:, 1], cmap="inferno", 
                   c=color_key_cam, s=5, marker='o')
        fig.canvas.draw()
        frame = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close(fig)
        diffusion_frames.append(frame)
    return diffusion_frames