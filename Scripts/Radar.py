print("loading libraries")
import pyart
import fsspec
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
import numpy as np
import os
import sys
import random
warnings.filterwarnings("ignore")
fs = fsspec.filesystem("s3", anon=True) 
import cartopy.io.shapereader as shpreader
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import math
import matplotlib.ticker as mticker
from matplotlib.transforms import Bbox


plt.rcParams['axes.facecolor'] = '#0e1111'
plt.rcParams['figure.facecolor'] = '#232b2b'
plt.rcParams['text.color'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams["figure.figsize"] = (12, 8)



# Load in the shapefiles
reader1 = shpreader.Reader("/path/to/shp")
reader2 = shpreader.Reader("/path/to/shp")
reader3 = shpreader.Reader("/path/to/shp")
reader4 = shpreader.Reader("/path/to/shp")

interstates = list(reader1.geometries())
INTERSTATES = cfeature.ShapelyFeature(interstates, ccrs.PlateCarree())


counties = list(reader2.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

borders = list(reader3.geometries())
BORDERS = cfeature.ShapelyFeature(borders, ccrs.PlateCarree())

lakes = list(reader4.geometries())
LAKES = cfeature.ShapelyFeature(lakes, ccrs.PlateCarree())




# Retrieve town names from the .csv file

def get_places(xmin, xmax, ymin, ymax):
    with open('/path/to/csv', 'r', encoding='windows-1252') as fp:
        content = fp.read().splitlines()
        places = []
        for line in content:
            test = line.split(',')
            place = test[0]
            lat = test[-2][:6]
            lon = test[-1][:7]
            try:
                float(lon)
                if float(lon) > xmin and float(lon) < xmax:
                    if float(lat) > ymin and float(lat) < ymax:
                        places.append([place, lon, lat])
            except Exception:
                pass

    return places

dx = 1
dy = 1
# Load in your nexrad L2 file
NEXRADLevel2File = ("path/to/nexrad/data/or/otherwise")
radar = pyart.io.read_nexrad_archive(NEXRADLevel2File)

rda_lon = radar.longitude['data'][0]
rda_lat = radar.latitude['data'][0]
xmin = rda_lon - dx + 0.25
xmax = rda_lon + dx - 0.25
ymin = rda_lat - dy + 0.25
ymax = rda_lat + dy - 0.25
locations = get_places(xmin, xmax, ymin, ymax)
places = get_places(xmin, xmax, ymin, ymax)





fig = plt.figure()

 # Load in the radar coordinates, working on an automatic import to eliminate manual input :) 
radar_lat = radar.latitude['data'] = np.array([37.654])
radar_lon = radar.longitude['data'] = np.array([-97.443])
radar_name = radar.metadata['instrument_name']

vcp = radar.metadata['vcp_pattern']

# Apply a nice gatefilter and implement the region dealias algorithm
gatefilter = pyart.filters.GateFilter(radar)
gatefilter.exclude_transition()
gatefilter.exclude_invalid("velocity")
gatefilter.exclude_outside("reflectivity", 0, 80)
gatefilter.exclude_outside("reflectivity", 0, 80)
gatefilter.exclude_below('reflectivity', 0)

dealias_data = pyart.correct.dealias_region_based(radar, gatefilter=gatefilter)
radar.add_field("corrected_velocity", dealias_data)
sweep = 0
radar_name = radar.metadata['instrument_name']

index_at_start = radar.sweep_start_ray_index['data'][sweep]
time_at_start_of_radar = pyart.io.cfradial.netCDF4.num2date(radar.time['data'][index_at_start], 
radar.time['units'])
formatted_date = time_at_start_of_radar.strftime('%m-%d-%Y %H:%MZ')

print("Grabbing places, dealiasing data...")


# Get slices for a NEXRAD files various moments and variables 
slice_indices = radar.get_slice(sweep)
max_ref = radar.fields['reflectivity']['data'][slice_indices].max()
elev_angle = (round(radar.elevation['data'][slice_indices].mean(), 2))

max_vel = radar.fields['corrected_velocity']['data'][slice_indices].max()
elev_angle = (round(radar.elevation['data'][slice_indices].mean(), 2)) 

max_cc = radar.fields['cross_correlation_ratio']['data'][slice_indices].min()
elev_angle = (round(radar.elevation['data'][slice_indices].mean(), 2)) 

max_zdr = radar.fields['differential_reflectivity']['data'][slice_indices].max()
elev_angle = (round(radar.elevation['data'][slice_indices].mean(), 2)) 
deg_sign = u'\N{DEGREE SIGN}'


# Begin the plot!
ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree())
ax1.add_feature(INTERSTATES, facecolor='none', edgecolor='yellow')
ax1.add_feature(COUNTIES, facecolor='none', edgecolor='red') 
ax1.add_feature(BORDERS, facecolor='none', edgecolor='white')
ax1.add_feature(LAKES, facecolor='blue', edgecolor='blue')
display = pyart.graph.RadarMapDisplay(radar)
ref_map = display.plot_ppi_map('reflectivity',
                               sweep=0,
                               vmin=-10,
                               vmax=70,
                               ax=ax1,
                               cmap='pyart_ChaseSpectral',
                               colorbar_label='', title='', colorbar_flag=False, lat_lines=[0], lon_lines=[0], 
                               lat_0=rda_lat, lon_0=rda_lon)
# Grab some coordinates to create the plot limits
plt.xlim()
plt.ylim()

for p in range(0, len(locations)):
    place = locations[p][0]
    lat = float(locations[p][2])
    lon = float(locations[p][1])
    plt.plot(lon, lat, '+', color='white', markersize=10, transform=ccrs.PlateCarree(), zorder=10)
    text_offset = 0.03  
    if lon < ax1.get_xlim()[1] and lon > ax1.get_xlim()[0] and lat < ax1.get_ylim()[1] and lat > ax1.get_ylim()[0]:
        plt.text(lon + text_offset, lat, place, horizontalalignment='left', verticalalignment='center', transform=ccrs.PlateCarree())

display.plot_range_ring(50., linestyle='dashed', color='gainsboro', lw=1)
display.plot_range_ring(150., linestyle='solid', color='gainsboro', lw=1)
display.plot_range_ring(250., linestyle='dashdot', color='gainsboro', lw=1)
display.plot_range_ring(350., linestyle='dashed', color='gainsboro', lw=1)



ax1.text(0.05, 0.95, 'Level-II\nRadar: ' + radar_name + f'\nVCP: {vcp}', fontsize=11, fontweight='bold', transform=ax1.transAxes, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5, color='black'))
ax1.text(0.5, 0.95, str(formatted_date), fontsize=11, fontweight='bold', transform=ax1.transAxes, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.5, color='black'))
ax1.text(0.95, 0.95, 'Max: ' + str(max_ref) + ' dBz' + '\nElev Angle: ' + str(elev_angle) + deg_sign, fontsize=11, fontweight='bold', transform=ax1.transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5, color='black'))
ax1.text(0.82, 0.05, 'Reflectivity', fontsize=11, fontweight='bold', transform=ax1.transAxes, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5, color='black'))

cax = fig.add_axes([ax1.get_position().x0, ax1.get_position().y0 - 0.05, ax1.get_position().width, 0.05])
display.plot_colorbar(mappable=None, field=None, label='($Z{e}$)', orient='horizontal', cax=cax, ax=ax1, fig=None, ticks=[-10, 0, 10, 20, 30, 40, 50, 60, 70,], ticklabs=None)
cax.xaxis.label.set_color('white')

print("Creating fancy titles...")

resol = '10m'
land = cfeature.NaturalEarthFeature('physical', 'land', scale=resol, facecolor=cfeature.COLORS['land'])
ax1.add_feature(land, facecolor='#5A4B41', zorder=0)


dtor = math.pi/180.0
max_range = 350
maxrange_meters = max_range * 1000.
meters_to_lat = 1. / 111177.
meters_to_lon =  1. / (111177. * math.cos(radar_lat * dtor))

for azi in range(0, 360, 45):  # 45 degree intervals
    azimuth = 90. - azi
    dazimuth = azimuth * dtor
    lon_maxrange = radar_lon + math.cos(dazimuth) * meters_to_lon * maxrange_meters
    lat_maxrange = radar_lat + math.sin(dazimuth) * meters_to_lat * maxrange_meters
    display.plot_line_geo([radar_lon, lon_maxrange], [radar_lat, lat_maxrange], linestyle='-', lw=0.5, color='gainsboro')


max_values = [50, 150, 350]     

xlim = plt.xlim()
ylim = plt.ylim()

azim = 90. - 30
dazim = azim * dtor
for i in max_values:
    max_m = i * 1000.
    lon_max = radar_lon + math.cos(dazim) * meters_to_lon * max_m
    lat_max = radar_lat + math.sin(dazim) * meters_to_lat * max_m
    if xlim[0] <= lon_max <= xlim[1] and ylim[0] <= lat_max <= ylim[1]:
     display.plot_point(lon_max, lat_max, symbol='None', label_text=' '+str(i)+'km')

display.plot_point(radar.longitude['data'][0], radar.latitude['data'][0],color='k',label_text=radar_name, symbol='*', markersize=10)

# Do some RNG to prevent file overwriting 
output_dir = ("/desired/path/to/folder/you/want/to/export/your/images/to")
def generate_and_save_random_png(output_dir):
    # Generate a random filename for the PNG file
    random_filename = f'PYART_{random.randint(1000, 9999)}.png'

    output_path = os.path.join(output_dir, random_filename)
    plt.savefig(output_path, dpi=400)
    

    print(f"Plotting Data.......Saving PNG file; Finised!: {output_dir}")

output_directory = "/desired/path/you/want"

generate_and_save_random_png(output_directory)

plt.show()
