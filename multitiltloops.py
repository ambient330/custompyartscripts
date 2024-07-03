import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
import pyart, glob
from scipy import constants
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.colors as colors
import matplotlib.colors as mcolors


## pick and choose RGB hexcodes to make a custom colormap (this one below is NWS Norman's colortable/the WDSS_II table ##

hex_colors = ['#F6F6F6', '#E7E7E7', '#D7D7D7', '#AF9FBB', '#952CAF', '#B3007C', '#990000', '#B40000', '#C71900', '#C47700', '#BBA501', '#9DAF02', '#017D01', '#02AA02', '#25BF25', '#88A788', '#766F7C', '#5E506A', '#50445B', '#43394C', '#352E3C', '#28222E', '#1B171F', '#0C0B0E', '#030203']

rgb_colors = [mcolors.hex2color(hex_code) for hex_code in hex_colors]

reversed_rgb_colors = rgb_colors[::-1]

custom_colormap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", reversed_rgb_colors)

plt.rcParams['axes.facecolor'] = '#0e1111'
plt.rcParams['figure.facecolor'] = '#232b2b'
plt.rcParams['text.color'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams["figure.figsize"] = (12, 8)

reader1 = shpreader.Reader("/mnt/c/Users/micha/Desktop/tl_2022_us_primaryroads/tl_2022_us_primaryroads.shp")
reader2 = shpreader.Reader("/mnt/c/users/micha/desktop/GPK.SHP.cb_2013_us_county_500k_wgs84/cb_2013_us_county_500k_wgs84.shp")
reader3 = shpreader.Reader("/mnt/c/users/micha/desktop/borders/InternationalBorders.shp")
reader4 = shpreader.Reader("/mnt/c/users/micha/desktop/lakes/ne_10m_lakes.shp")

interstates = list(reader1.geometries())
INTERSTATES = cfeature.ShapelyFeature(interstates, ccrs.PlateCarree())


counties = list(reader2.geometries())
COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())

borders = list(reader3.geometries())
BORDERS = cfeature.ShapelyFeature(borders, ccrs.PlateCarree())

lakes = list(reader4.geometries())
LAKES = cfeature.ShapelyFeature(lakes, ccrs.PlateCarree())




fig = plt.figure(figsize=(14, 8))

def plot_radar(radar, display, sweep_idx):
    plt.clf()

    z_sweep = sweep_idx
    v_sweep = sweep_idx + 1
    name = radar.metadata['instrument_name']
    angle = radar.fixed_angle['data'][0]
    
    sweep = 0

    ts = pyart.graph.common.generate_radar_time_sweep(radar, z_sweep).strftime('%m/%d/%Y %H:%M:%S') + 'Z'
    
    slice_indices = radar.get_slice(sweep)
    max_ref = radar.fields['reflectivity']['data'][slice_indices].max()

    
    dx = 1
    dy = 1
    vcp = radar.metadata['vcp_pattern']
    rda_lon = radar.longitude['data'][0]
    rda_lat = radar.latitude['data'][0]
    xmin = rda_lon - dx + 0.25
    xmax = rda_lon + dx - 0.25
    ymin = rda_lat - dy + 0.25
    ymax = rda_lat + dy - 0.25
    deg_sign = u'\N{DEGREE SIGN}'
    elev_angle = (round(radar.elevation['data'][slice_indices].mean(), 2))

    ax = plt.axes(projection=ccrs.PlateCarree())
    


    display.plot_ppi_map('reflectivity', z_sweep,
                 vmin=-33, vmax=92, 
                 colorbar_label='(Zh)', 
                 cmap=custom_colormap,
                 title=None,
                 ax=ax, colorbar_flag=False, lat_lines=[0], lon_lines=[0], 
                 lat_0=rda_lat, lon_0=rda_lon)
                 
                 
                 


    # Set limits
    plt.xlim()
    plt.ylim()

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(INTERSTATES, facecolor='none', edgecolor='yellow')
    ax.add_feature(COUNTIES, facecolor='none', edgecolor='red') 
    ax.add_feature(BORDERS, facecolor='none', edgecolor='black')
    ax.add_feature(LAKES, facecolor='blue', edgecolor='blue', zorder=1)

    cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.09, ax.get_position().width, 0.05])
    display.plot_colorbar(mappable=None, field=None, label=None, orient='horizontal', cax=cax, ax=ax, fig=None, ticks=[-33, -28, -23, -18, -13, -8, -3, 2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 77, 82, 87, 91], ticklabs=None, extend='both')
    display.plot_point(radar.longitude['data'][0], radar.latitude['data'][0],color='k',label_text=name, symbol='*', markersize=10)
    
    ax.set_title('Reflectivity - Max: ' + str(max_ref) + 'dBZ', fontsize=11, loc='left', fontweight='bold')
    ax.set_title(f'{name} Low Z {ts}', fontsize=11, loc='left', fontweight='bold')
    ax.set_title(f'VCP: {vcp}', fontsize=11, loc='center', fontweight='bold') 
    resol = '10m'
    land = cfeature.NaturalEarthFeature('physical', 'land', scale=resol, facecolor=cfeature.COLORS['land'])
    ax.add_feature(land, facecolor='#1B292E', zorder=0)
    #gdf.plot()



def animate():
    frame = 0
    files = glob.glob('/mnt/c/Users/micha/downloads/ok/*.gz') 
    for file in files:
        radar = pyart.io.read_nexrad_archive(file)
        
        print(file)

        elev = radar.fixed_angle['data'][:]
        sweep_idx = np.where(np.logical_and(elev >= elev_min, elev <= elev_max))[0]
        display = pyart.graph.RadarMapDisplay(radar)

        for sweep in sweep_idx:
            plot_radar(radar, display, sweep)

            if frame < 1:
                frame += 1

            yield

        del radar, display
elev_min = 0.5 # min elevation you want to plot, max elevation below. 
elev_max = 1.8


fig = plt.figure()
anim = FuncAnimation(fig, lambda artists: artists, frames=animate())
anim.save('animation.mp4', writer='ffmpeg', fps=10, dpi=144)
