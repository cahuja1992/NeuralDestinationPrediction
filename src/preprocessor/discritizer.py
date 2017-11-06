import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def find_direct_path(start, end):
    x1, y1 = start
    x2, y2 = end

    dy = y2 - y1
    dx = x2 - x1

    # Check if the connecting line is steep
    steep = abs(dy) > abs(dx)
    if steep:
        # If so, rotate the line
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap the start and end points if necessary
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate the differentials
    dy = y2 - y1
    dx = x2 - x1

    # Calculate the error
    error = int(dx / 2.0)

    # Are we moving left or right?
    ystep = 1 if y1 < y2 else -1

    # Iterate over the possible points between start and end
    y = y1
    path = []
    for x in range(x1, x2 + 1):
        # Bresenham's algorithm
        point = (y, x) if steep else (x, y)
        path.append(point)
        error -= abs(dy)

        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the start and end points were swapped
    if swapped:
        path.reverse()

    return path


def trip_to_grid(coords, lon_bins, lon_step, lat_bins, lat_step):
    # Initialize list with grid coordinates of the trip
    grid_trip = []

    for coord in coords:
        # Determine the cell for each pair of coordinates
        cell_lon = np.max(np.where(coord[0] > lon_bins))
        cell_lat = np.max(np.where(coord[1] > lat_bins))
        new_cell = (cell_lon, cell_lat)

        # Add the cell coordinates to the list if it is empty
        if len(grid_trip) == 0:
            grid_trip.append(new_cell)
            continue

        # If the new cell is the same as the old cell, do not add it
        # if new_cell == grid_trip[-1]:
        #  continue

        # Otherwise, add the cell to the list and interpolate the cells in
        # between, if necessary.
        if abs(new_cell[0] - grid_trip[-1][0]) >= 2 or abs(new_cell[1] - grid_trip[-1][1]) >= 2:
            # Compute the direct path between the gap start and end points
            gap_path = find_direct_path(start=grid_trip[-1], end=new_cell)

            # Add the path to the coordinate list
            grid_trip.extend(gap_path[1:])
        else:
            grid_trip.append(new_cell)

    return Grid(grid_trip, lon_bins, lon_step, lat_bins, lat_step)


def plot_grid_representation(coords, grid, plot_grid_lines=False):
    # Get the handle of the figure
    fig, ax = plt.subplots()

    # Plot the GPS coordinates of the trip
    ax.plot(coords[:, 0], coords[:, 1], color="black")
    ax.scatter(coords[:, 0], coords[:, 1], s=10, color="black")

    # Plot the grid representation
    ax = grid.PlotGrid(ax, plot_grid_lines)

    return fig, ax


class Grid:
    def __init__(self, grid, lon_grid, lon_step, lat_grid, lat_step):
        self.grid = grid
        self.lon_grid = lon_grid
        self.lon_step = lon_step
        self.lat_grid = lat_grid
        self.lat_step = lat_step

    def grid_to_string(self):
        return str(self.grid)

    def grid_to_array(self):
        N = len(self.grid)
        grid_array = np.zeros((N, N))

        for cell in self.grid:
            grid_array[(cell[0] - 1, cell[1] - 1)] += 1

        return grid_array

    def PlotGrid(self, ax, plot_grid_lines=False):

        # If there is no figure given to the function
        # if fig is None or ax is None:
        #  fig, ax = plt.subplots()

        # For each cell in the grid representation, we plot a shaded grey box
        for cell in self.grid:
            # Compute the coordinates of the cell
            lon_min = self.lon_grid[0] + cell[0] * self.lon_step
            lon_max = lon_min + self.lon_step

            lat_min = self.lat_grid[0] + cell[1] * self.lat_step
            lat_max = lat_min + self.lat_step

            # Plot the shaded cell
            ax.fill([lon_min, lon_max, lon_max, lon_min], [lat_min, lat_min, lat_max, lat_max], 'k', alpha=0.05)

        if plot_grid_lines:
            # Hack so we get the same x and y axis ranges
            x_range = ax.get_xlim()
            y_range = ax.get_ylim()

            # We plot the grid lines as well
            ax.set_yticks(self.lat_grid, minor=False)
            ax.set_xticks(self.lon_grid, minor=False)

            # Show the gridlines
            ax.yaxis.grid(True)
            ax.xaxis.grid(True)

            # Reset the ranges for the x and y axis
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)

        return ax

    def to_img(self):
        I = np.full([N, M], 255, dtype='int32')
        for x in range(N):
            for y in range(M):
                for lon, lat in self.grid:
                    if x == lon and y == lat:
                        I[x, y] = 255 // 2
                    if x == self.grid[-1][0] and y == self.grid[-1][1]:
                        I[x, y] = 0

        return I.flatten()


if __name__ == "__main__":

    # Get the location of the data file
    filepath = "test_cleaned.csv"
    filepath_clean = "test_binarized_trips.csv"

    # Define the bins for the grid
    # Note: 1 degree of longitude/latitude is approximately 111.38 km
    N = 100
    M = 75

    # From visual inspection of porto map. We are only focusing on the city centre
    # 15-07-14: We continue with this bounding box as of now
    lon_vals = [-8.73, -8.5]
    lat_vals = [41.1, 41.25]

    lon_bins, lon_step = np.linspace(lon_vals[0], lon_vals[1], N, retstep=True)
    lat_bins, lat_step = np.linspace(lat_vals[0], lat_vals[1], M, retstep=True)

    # Use pandas read_csv function to read the data in chunks
    data_chunks = pd.read_csv(filepath_or_buffer=filepath,
                              sep=",",
                              chunksize=10000,
                              converters={'POLYLINE': lambda x: json.loads(x)})

    # Define a function that transforms the POLYLINE data
    to_grid = lambda polyline: trip_to_grid(np.array(polyline), lon_bins, lon_step, lat_bins, lat_step)

    for idx, chunk in enumerate(data_chunks):
        # Compute the grid representations of these trips
        chunk["GRID_POLYLINE"] = chunk["POLYLINE"].map(to_grid)

        chunk["IMAGE"] = chunk["GRID_POLYLINE"].map(lambda x: x.to_img())

        chunk["GRID_POLYLINE"] = chunk["GRID_POLYLINE"].map(lambda x: x.grid_to_string())

        # Append data to a csv file
        if idx == 0:
            chunk.to_csv(filepath_clean, header=True)
        else:
            chunk.to_csv(filepath_clean, mode="a", header=False)