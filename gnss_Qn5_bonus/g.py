import folium
from folium import plugins

#### DO NOT CHANGE THIS PART

#Converts NMEA to Decimal (Map format)
def nmea_to_dec(value, direction):
    if not value: return 0.0
    split = 2 if direction in ['N', 'S'] else 3
    deg = float(value[:split])
    min = float(value[split:])
    res = deg + (min / 60.0)
    return -res if direction in ['S', 'W'] else res

#Parses the NMEA file
path_coords = []
base_coords = []
qualities = []

with open('moving_rover.nmea', 'r') as f:
    for line in f:
        if line.startswith('$GPGGA'):
            bits = line.split(',')
            lat = nmea_to_dec(bits[2], bits[3])
            lon = nmea_to_dec(bits[4], bits[5])
            quality = bits[6] # 1=GPS, 2=DGPS
            
            path_coords.append([lat, lon])
            qualities.append(quality)
with open('fixed_rover.nmea', 'r') as f1:
    for line in f1:
        if line.startswith('$GPGGA'):
            bits = line.split(',')
            lat = nmea_to_dec(bits[2], bits[3])
            lon = nmea_to_dec(bits[4], bits[5])
            base_coords.append([lat, lon])
m = folium.Map(location=path_coords[0], zoom_start=19, tiles='OpenStreetMap')

start_lat = path_coords[0][0]
start_lon = path_coords[0][1]

##### ADD YOUR CODE HERE

# -------- Differential GNSS Correction --------

# -------- Differential GNSS Correction --------

# Estimate true base position (average of stationary receiver readings)
true_base_lat = sum(p[0] for p in base_coords) / len(base_coords)
true_base_lon = sum(p[1] for p in base_coords) / len(base_coords)

corrected_coords = []

for i in range(min(len(path_coords), len(base_coords))):

    rover_lat, rover_lon = path_coords[i]
    base_lat, base_lon = base_coords[i]

    # error observed at base station
    err_lat = base_lat - true_base_lat
    err_lon = base_lon - true_base_lon

    # corrected rover position
    corr_lat = rover_lat - err_lat
    corr_lon = rover_lon - err_lon

    corrected_coords.append([corr_lat, corr_lon])

# replace path with corrected path
path_coords[:] = corrected_coords


# -------- Mark all points --------

# Start marker
folium.Marker(
    location=path_coords[0],
    popup="Start (Gajendra Circle)",
    icon=folium.Icon(color="red", icon="play")
).add_to(m)

# End marker
folium.Marker(
    location=path_coords[-1],
    popup="Destination (Swimming Pool)",
    icon=folium.Icon(color="black", icon="flag")
).add_to(m)

# Intermediate markers (green pins, no label)
for pt in path_coords[1:-1]:
    folium.Marker(
        location=pt,
        icon=folium.Icon(color="green", icon="info-sign")
    ).add_to(m)

##### END OF YOUR CODE


##### DO NOT CHANGE THIS, IT IS FOR VISUALIZATION
plugins.AntPath(
    locations=path_coords, 
    dash_array=[1, 10], 
    delay=1000, 
    color='blue', 
    weight=5,
    paused=False,      
    reverse=False,     
    loop=False         
).add_to(m)
m.save("my_rover_map.html")
print("Success! Open 'my_rover_map.html' in your browser.")