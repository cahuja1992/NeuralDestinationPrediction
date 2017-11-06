import numpy as np
import pandas as pd
import datetime
import json


def haversine(p1, p2):
    # convert decimal degrees to radians
    if p1.ndim == 1:
        p1 = p1.reshape(-1, 2)
    if p2.ndim == 1:
        p2 = p2.reshape(-1, 2)

    lon1, lat1, lon2, lat2 = map(np.radians, [p1[:, 0], p1[:, 1], p2[:, 0], p2[:, 1]])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r


def remove_outliers(trip, threshold_speed=120):
    c = threshold_speed / 3600 * 15  # km per 15s threshold

    # Initialize the list with the cleaned coordinates
    new_trip = [trip[0]]

    for point in trip[1:]:
        # Compare the distance between the current point and the first previous
        # non-outlier point
        if haversine(point, new_trip[-1]) < c:
            new_trip.append(point)

    return np.array(new_trip)


def is_outlier(trip, threshold_speed=120):
    # Define a conversion for the threshold comparison
    c = threshold_speed / 3600 * 15  # km per 15s threshold

    if type(is_outlier) != np.ndarray:
        try:
            trip = np.array(trip)
        except:
            print("Fuck")
            return True

    # Compute distances
    distances = haversine(trip[:-1], trip[1:])

    if np.any(distances >= c):
        return True
    else:
        return False


def is_trip_in_grid(trip, lon_vals, lat_vals):
    if len(trip) is not 0:
        in_lon_grid = np.all(trip[:, 0] >= lon_vals[0]) and np.all(trip[:, 0] <= lon_vals[1])
        in_lat_grid = np.all(trip[:, 1] >= lat_vals[0]) and np.all(trip[:, 1] <= lat_vals[1])
    else:
        return False

    return in_lon_grid and in_lat_grid


if __name__ == "__main__":

    # Get the location of the data file
    filepath = "test.csv"
    filepath_clean = "test_cleaned.csv"

    # Use pandas read_csv function to read the data in chunks
    data_chunks = pd.read_csv(filepath_or_buffer=filepath,
                              sep=",",
                              chunksize=10000,
                              converters={'POLYLINE': lambda x: json.loads(x)})

    # From visual inspection of porto map. We are only focusing on the city centre
    lon_vals = (-8.73, -8.5)
    lat_vals = (41.1, 41.25)

    # Define a lambda function that can be passed to pd.map
    is_in_grid = lambda x: is_trip_in_grid(np.array(x), lon_vals, lat_vals)

    for idx, chunk in enumerate(data_chunks):
        # Remove the points that have MISSING_DATA = TRUE
        missing = (chunk["MISSING_DATA"] == True)
        # Check if each trip is contained in the grid representation
        in_grid = chunk["POLYLINE"].map(is_in_grid)
        # Remove the trips that have outlier points
        outliers = chunk["POLYLINE"].map(is_outlier)
        # Remove trips that only contain a starting point
        nrpoints = chunk["POLYLINE"].map(len)

        # Concatenate these if statements together
        remove = missing | outliers | ~ in_grid | (nrpoints <= 1)

        # Remove these rows from the dataframe
        chunk = chunk[-remove].reset_index(drop=True)

        # Store start and end points of trips for later use
        chunk["START_POINT"] = chunk["POLYLINE"].apply(lambda x: x[0])
        chunk["END_POINT"] = chunk["POLYLINE"].apply(lambda x: x[-1])

        # Store UNIX timestamp as POSIXct
        chunk["TIMESTAMP"] = chunk["TIMESTAMP"].apply(datetime.datetime.utcfromtimestamp)

        # Get hour of the day and day of the week from timestamp
        chunk["HOUR"] = chunk["TIMESTAMP"].dt.hour
        chunk["WDAY"] = chunk["TIMESTAMP"].dt.weekday

        # Calculate length of trip
        chunk["DURATION"] = chunk["POLYLINE"].apply(lambda x: (len(x) - 1) * 15)

        # Back transform POLYLINE
        chunk["POLYLINE"] = chunk["POLYLINE"].apply(json.dumps)

        # Append chunk to file
        if idx == 0:
            chunk.to_csv(filepath_clean, header=True)
        else:
            chunk.to_csv(filepath_clean, mode="a", header=False)