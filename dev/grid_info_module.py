import numpy as np

class grid_info():
    def __init__(self, n_grid, x_min, x_max, y_min, y_max, z_min, z_max):
        assert x_min < x_max
        assert y_min < y_max
        assert z_min < z_max
        self.n_grid = n_grid
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        # 1D Array
        self.x_rng = np.linspace(x_min, x_max, n_grid)
        self.delta_x = self.x_rng[1]-self.x_rng[0]
        self.y_rng = np.linspace(y_min, y_max, n_grid)
        self.delta_y = self.y_rng[1]-self.y_rng[0]
        self.z_rng = np.linspace(z_min, z_max, n_grid)
        self.delta_z = self.z_rng[1]-self.z_rng[0]
        #
        # 3D Array
        self.XYZ = np.meshgrid(self.x_rng, self.y_rng, self.z_rng, indexing='ij') # Use this like X, Y,
        # Z = my_grid.XYZ to lo