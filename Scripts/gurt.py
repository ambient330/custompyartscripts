#Version 2 Graphic Utility Radar Toolkit (GURT)
#@multidpppler 

import pyart
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import os
import argparse
from datetime import datetime
import matplotlib.colors as colors
from copy import deepcopy
from scipy import interpolate
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.use('Agg')

plt.rcParams["font.weight"] = 'bold'
plt.rcParams['axes.facecolor'] = 'black' #191970



 
def create_custom_cmap(hex_colors):
    rgb_colors = [mcolors.hex2color(hex_code) for hex_code in hex_colors]
    return mcolors.LinearSegmentedColormap.from_list("custom_cmap", rgb_colors[::-1])

zdr_cmap = create_custom_cmap(['#922E97', '#B26CB6'
, '#C998CB', '#E2C7E3', '#FEF9FB', '#F8BEDB', '#EF77B2', 
'#CF3B58', '#B00000', '#C80603', '#DE1A0B', '#EE8836',
 '#FEF861', '#5ADE64', '#3FE2CF', '#2474B4', '#0B0D9C', 
 '#D1D1D9', '#7F6CA2', '#453B58', '#292335', '#060507'])
cc_cmap = create_custom_cmap(['#8B1E4D', '#E41000',
 '#FC7F00', '#FFB600', '#FFFB00', '#BCE906', '#87D70B', 
 '#61ED6E', '#719CD2', '#5151E8', '#2929D1', '#0A0ABD', 
 '#0C0CAC', '#0D0D9C', '#0F0F8C', '#1C1C9E', '#2D2D84', 
 '#404068', '#454561', '#4F4F4F'])
kdp_cmap = create_custom_cmap(['#C361F9', '#6F329A', 
'#160234', '#624264', '#B18596', '#FAC4C5', '#FF7B00', 
'#FFBC00', '#FEFF00', '#84DA1A', '#16BA31', '#3ADB94', 
'#60FEF6', '#74C7D1', '#8987A2', '#9B507A', '#EA77B8', 
'#CE5B93', '#B03D6A', '#921F42', '#75021B', '#62000E', 
'#4B0101', '#4B2828', '#4B4A4A', '#5F5F5F', '#757575'])
sw_cmap = create_custom_cmap(['#02A0C8', '#2CA7C6', 
'#53AEC5', '#78B4C3', '#9FBBC1', '#C1C1C1', '#DCDCDC',
 '#E6E6E6', '#F2F2F2', '#FFFD01', '#FDC60F', '#FDB313',
 '#FC991A', '#F7742D', '#EF6341', '#E54F5B', '#DE406D',
 '#B73192', '#7D26BD', '#31148A', '#1A0855'])
phi_cmap = create_custom_cmap([
'#FE8BFE', '#ED7CE3', '#D569BE', '#C0579E', '#AB457D',
 '#98345F', '#83233F', '#6D111E', '#5A0001', '#882300',
 '#B04100', '#DA6100', '#FF7F00', '#FFBF00', '#FFFE00', 
 '#9EBC00', '#1C6E00', '#10AE00', '#0FB100', '#00F404', 
 '#058730', '#0A1A5C', '#71758A', '#D2D2B5', '#E9E9DA', 
 '#F7F7F1'])
hawkeye_cmap = create_custom_cmap([
'#FEFEFE', '#FFFFFF', '#DCDCDC', '#FD158B', '#F13C3A', 
'#F48474', '#D5BC20', '#D4A919', '#C16022', '#8E2453', 
'#8D3B60', '#0F0A99', '#7A65EC', '#3CAC71', '#076658', 
'#318025', '#4E6F2F', '#386421', '#0B5C09'])
vel_cmap = create_custom_cmap([    '#040202', '#2F1913', '#552E23', '#794232', '#A55B43', '#B67A36', '#C99C28',
    '#DCBF1A', '#EFE10C', '#FEFD01', '#FE9F25', '#FF4F1D', '#C72913', '#911C0D', 
    '#580E06', '#210000',   '#6D6C6C',    '#082303', '#185616', '#2D902E', '#3EC142',
    '#4FF580', '#44E7F1', '#3AC4F3', '#39C2F3', '#2F9EF5', '#2273F8', '#164BFA',
    '#0C28FD', '#0000FD', '#0000C0', '#00007D', '#00004B'])
ref_cmap = create_custom_cmap([
'#7E7A79', '#8D8988', '#9B9793', '#ACA8A5', '#B7B3B0', '#C6C2BF', '#D3D2D0', '#E0E0DE', '#F0EFED', '#FEFFFE', 
'#FFFFFF', '#7E017B', '#850179', '#89007B', '#8D007B', '#93077B', '#9A1281', '#A01F83', '#A52384', '#B13289', '#B2368E',
 '#9D001B', '#A9001A', '#B50121', '#CA0226', '#D50323', '#E90127', '#E11923', '#E01F22', '#E82023', '#F02124', '#EB7A15',
 '#EC860C', '#ED9705', '#F1A400', '#EEB603', '#F5C50C', '#F6D611', '#F6E714', '#F3F21E', '#FFFE1E', '#007D26', '#01952B', 
 '#019E31', '#00A937', '#00AF35', '#06AF34', '#1BB230', '#1CB433', '#28B62F', '#30B42F', '#30B630', '#39B730', 
 '#3AB630', '#43B92D', '#41107B', '#41107B', '#41107B', '#422282', '#4B388F', '#4D4597', '#5558A2', '#5867AD', '#5E73B5', 
 '#5E81C0', '#5E94CC', '#5EA4DA', '#54B5E8', '#5CC0F1', '#69C2DE', '#C1BDBC', '#ABA7A4', '#95918E', '#7E7A79', '#676362', 
 '#514D4C', '#383637', '#0D0E10', '#030303'])
 
 
 
 
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from scipy import interpolate

def mybgcolor():
    """Returns background color - adjust based on your needs"""
    return [0.10, 0.00, 0.30]  # white background

def fleximap(n, control_points):
    """
    Create a colormap by interpolating between control points
    n: number of colors in output colormap
    control_points: array of [position, r, g, b] values
    """
    positions = control_points[:, 0]
    colors = control_points[:, 1:4]
    
    f_r = interpolate.interp1d(positions, colors[:, 0], kind='linear')
    f_g = interpolate.interp1d(positions, colors[:, 1], kind='linear')
    f_b = interpolate.interp1d(positions, colors[:, 2], kind='linear')
    
    new_positions = np.linspace(0, 1, n)
    
    new_r = f_r(new_positions)
    new_g = f_g(new_positions)
    new_b = f_b(new_positions)
    
    colormap = np.column_stack((new_r, new_g, new_b))
    
    return colormap

def carbmap(num=256):
    """
    Generate a carbon-style colormap
    num: number of colors in the colormap (default: 256)
    Returns: numpy array of shape (num, 3) with RGB values
    """
    if num < 16:
        print(f'Cannot generate map with {num} elements')
        return np.array([])
    
    num = num - 1
    
    pt = np.array([
        [0.000,                  *mybgcolor()],
        [1/num,                  0.10, 0.00, 0.30],  # dark blue
        [0.125,                  0.33, 0.06, 0.73],  # blue-purple
        [round(0.25*num-1)/num,  0.50, 0.45, 1.00],  # light blue
        [round(0.25*num)/num,    0.00, 0.40, 0.00],  # green
        [0.325,                  0.00, 0.65, 0.00],  # mid-green
        [round(0.475*num-1)/num, 0.70, 0.90, 0.70],  # light green
        [round(0.475*num)/num,   0.70, 0.70, 0.70],  # gray
        [0.575,                  1.00, 1.00, 0.00],  # light yellow
        [0.675,                  0.95, 0.60, 0.10],  # yellowish orange
        [round(0.825*num)/num,   0.50, 0.30, 0.25],  # brown
        [round(0.825*num+1)/num, 1.00, 0.30, 0.51],  # pink/red
        [1.000,                  0.60, 0.00, 0.05]   # dark red
    ])
    
    x = fleximap(num + 1, pt)
    return x

def create_carbmap_colormap(n_colors=256):
    """
    Create a matplotlib colormap from carbmap function
    Returns: matplotlib LinearSegmentedColormap object
    """
    colors = carbmap(n_colors)
    cmap = LinearSegmentedColormap.from_list('carbmap', colors)
    return cmap

carbmap_cmap = create_carbmap_colormap()
try:
    import matplotlib
    matplotlib.colormaps.register(carbmap_cmap, name='carbmap')
except AttributeError:
    plt.register_cmap(name='carbmap', cmap=carbmap_cmap)

def create_custom_cmap(hex_colors):
    rgb_colors = [mcolors.hex2color(hex_code) for hex_code in hex_colors]
    return mcolors.LinearSegmentedColormap.from_list("custom_cmap", rgb_colors[::-1])

from scipy.interpolate import interp1d

def fleximap(num, control_points):
    """
    MATLAB fleximap equivalent - interpolates colors between control points
    control_points: array where each row is [position, R, G, B]
    """
    positions = control_points[:, 0]
    colors = control_points[:, 1:4]
    
    r_interp = interp1d(positions, colors[:, 0], kind='linear', bounds_error=False, fill_value='extrapolate')
    g_interp = interp1d(positions, colors[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')
    b_interp = interp1d(positions, colors[:, 2], kind='linear', bounds_error=False, fill_value='extrapolate')
    
    x = np.linspace(0, 1, num)
    
    r_vals = r_interp(x)
    g_vals = g_interp(x)
    b_vals = b_interp(x)
    
    rgb_array = np.column_stack([r_vals, g_vals, b_vals])
    rgb_array = np.clip(rgb_array, 0, 1)
    
    return rgb_array

def create_rgmap_colormap(num=256):
    """Red-Green colormap with white center"""
    pt = np.array([
        [0,    0.00, 0.20, 0.00],  # Dark green
        [0.30, 0.00, 0.80, 0.00],  # Bright green
        [0.50, 0.85, 0.85, 0.85],  # Light gray/white
        [0.70, 0.80, 0.00, 0.00],  # Bright red
        [1.00, 0.20, 0.00, 0.00]   # Dark red
    ])
    colors = fleximap(num, pt)
    return LinearSegmentedColormap.from_list('rgmap', colors)

def create_rgmapwe_colormap(num=256):
    """Red-Green colormap with blue-cyan-yellow extended range"""
    anchors = np.floor(np.array([0, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 1.0]) * num) + \
              np.array([1, 0, 1, -1, 0, 1, 2, 0, 1, 0])
    anchors = (anchors - 1) / (num - 1)
    
    colors = np.array([
        [0.00, 0.00, 1.00],  # Blue
        [0.00, 1.00, 1.00],  # Cyan
        [0.00, 1.00, 0.00],  # Green
        [0.00, 0.40, 0.00],  # Dark green
        [0.45, 0.60, 0.45],  # Light green-gray
        [0.60, 0.40, 0.40],  # Light red-gray
        [0.45, 0.00, 0.00],  # Dark red
        [1.00, 0.00, 0.00],  # Red
        [1.00, 0.40, 0.00],  # Orange-red
        [1.00, 1.00, 0.30]   # Yellow
    ])
    
    pt = np.column_stack([anchors, colors])
    colors_array = fleximap(num, pt)
    return LinearSegmentedColormap.from_list('rgmapwe', colors_array)

rgmap_cmap = create_rgmap_colormap()
rgmapwe_cmap = create_rgmapwe_colormap()

try:
    # For matplotlib >= 3.5
    import matplotlib
    matplotlib.colormaps.register(rgmap_cmap, name='rgmap')
    matplotlib.colormaps.register(rgmapwe_cmap, name='rgmapwe')
except AttributeError:
   
    print("Warning: Could not register colormaps. Use the cmap objects directly.")
    
    
    
    
    
    
    
    
    
    
    
    
    
    



 
 
fields = [ 
        ("DBZ", r'$Z_{h}$ (dBz)', 25, 60, ref_cmap), 
        ("VEL", r'$V_{r}$ (m/s)', -25, 25, carbmap_cmap),
        ("DP", r'$K_{dp}$ (deg/km)', -2, 12, kdp_cmap)]
        
    #
    #("ZDR", r'$Z_{dr}$ (dB)', -2, 8, zdr_cmap),
    #("RHOHV", r'$\rho_{HV}$ (unitless)', 0, 1.0, cc_cmap),    
    #("DP", r'$K_{dp}$ (deg/km)', -2, 12, kdp_cmap),
    #('SW", r'$\sigma_{2}$ (m/s)', 0, 15, 'magma')
    #("differential_phase", r'$\phi_{dp}$ (deg)', 0, 360, "pyart_Carbone42")]























import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def calculate_kdp(radar, force_recalculate=False, resolution_multiplier=0):
    """
    Calculate KDP field from radar data using an improved method with filtering.
    Only calculates KDP for bins with reflectivity greater than or equal to 20 dBZ.
    
    Parameters:
    -----------
    radar : pyart.core.Radar
        Radar object
    force_recalculate : bool
        If True, recalculates KDP even if it already exists in the radar data
    resolution_multiplier : int
        Factor by which to increase the range resolution (e.g., 2 = double resolution)
        Default is 1 (no change in resolution)
        
    Returns:
    --------
    radar : pyart.core.Radar
        Radar object with KDP field added
    """
    PHIDP_NAMES = ['PHIDP', 'PHI', 'differential_phase', 'PH']
    REFLECTIVITY_NAMES = ['REFLECTIVITY', 'reflectivity', 'DBZ', 'dbz', 'DBMHC']
    
    if 'DP' in radar.fields and not force_recalculate:
        print("KDP field already exists in radar data")
        return radar
    
    phidp_field_name = None
    for name in PHIDP_NAMES:
        if name in radar.fields:
            phidp_field_name = name
            print(f"Found PhiDP field: {name}")
            break
    
    if phidp_field_name is None:
        print("WARNING: No PhiDP field found in radar data. Checked fields:", PHIDP_NAMES)
        print("KDP will not be calculated")
        kdp_field = {
            'data': np.zeros_like(list(radar.fields.values())[0]['data']),
            'units': 'degrees/km',
            'long_name': 'Specific differential phase',
            'standard_name': 'KDP',
            '_FillValue': -9999.0,
        }
        radar.add_field('DP', kdp_field, replace_existing=True)
        return radar
    
    refl_field_name = None
    for name in REFLECTIVITY_NAMES:
        if name in radar.fields:
            refl_field_name = name
            print(f"Found reflectivity field: {name}")
            break
    
    if refl_field_name is None:
        print("WARNING: No reflectivity field found. Checked fields:", REFLECTIVITY_NAMES)
        print("No reflectivity threshold used. Prepare for shitty clutter")
        refl_mask = np.zeros_like(radar.fields[phidp_field_name]['data'].data, dtype=bool)
    else:
        refl_data = radar.fields[refl_field_name]['data']
        refl_mask = refl_data < 20
    
    print(f"Calculating KDP with resolution multiplier: {resolution_multiplier}x...")
    
    phidp_raw = radar.fields[phidp_field_name]['data'].copy()
    
    r = radar.range['data']  
    dr = (r[1] - r[0]) / 1000.0 
    
    if resolution_multiplier > 1:
        r_highres = np.linspace(r[0], r[-1], len(r) * resolution_multiplier)
        dr_highres = (r_highres[1] - r_highres[0]) / 1000.0
        
        phidp_highres = np.zeros((phidp_raw.shape[0], len(r_highres)))
        refl_mask_highres = np.zeros((refl_mask.shape[0], len(r_highres)), dtype=bool)
        
        print(f"Interpolating to high resolution: {len(r)} -> {len(r_highres)} gates")
    else:
        dr_highres = dr
    
    if resolution_multiplier > 1:
        kdp_data = np.zeros_like(phidp_highres)
    else:
        kdp_data = np.zeros_like(phidp_raw)
    
    if hasattr(phidp_raw, 'mask'):
        mask = phidp_raw.mask.copy()
    else:
        fill_value = radar.fields[phidp_field_name].get('_FillValue', -9999.0)
        mask = np.isclose(phidp_raw, fill_value) | np.isnan(phidp_raw)
    
    combined_mask = mask | refl_mask
    
    phidp = phidp_raw.copy()
    if hasattr(phidp, 'mask'):
        phidp = phidp.filled(np.nan)
    
    for i in range(phidp.shape[0]):
        valid_indices = ~np.isnan(phidp[i, :]) & ~combined_mask[i, :]
        if np.sum(valid_indices) > 0:
            phidp[i, valid_indices] = np.unwrap(np.deg2rad(phidp[i, valid_indices])) * 180.0 / np.pi
    
    if resolution_multiplier > 1:
        for i in range(phidp.shape[0]):
            valid_indices = ~np.isnan(phidp[i, :]) & ~combined_mask[i, :]
            if np.sum(valid_indices) > 2:  

                f_phidp = interp1d(r[valid_indices], phidp[i, valid_indices], 
                                   kind='linear', bounds_error=False, fill_value=np.nan)
                phidp_highres[i, :] = f_phidp(r_highres)
                
                f_mask = interp1d(r, combined_mask[i, :].astype(float), 
                                  kind='nearest', bounds_error=False, fill_value=1)
                refl_mask_highres[i, :] = f_mask(r_highres) > 0.5
        
        phidp = phidp_highres
        combined_mask = refl_mask_highres
        dr = dr_highres
    
    window_size = 9  # Window size for smoothing (adjust as needed)
    
    for i in range(phidp.shape[0]):
        valid = ~np.isnan(phidp[i, :]) & ~combined_mask[i, :]
        if np.sum(valid) > window_size:
            kernel = np.ones(window_size) / window_size
            padded = np.pad(phidp[i, valid], (window_size//2, window_size//2), mode='edge')
            smoothed = np.convolve(padded, kernel, mode='valid')
            phidp[i, valid] = smoothed
            
            
            kdp_ray = np.zeros_like(phidp[i, :])
            
            half_win = 4  
            for j in range(half_win, len(phidp[i, :]) - half_win):
                if valid[j-half_win] and valid[j+half_win]:

                    kdp_ray[j] = (phidp[i, j+half_win] - phidp[i, j-half_win]) / (2 * 2 * half_win * dr)
            
            valid_kdp = ~np.isnan(kdp_ray) & (kdp_ray != 0)
            if np.sum(valid_kdp) > window_size:
                padded_kdp = np.pad(kdp_ray[valid_kdp], (window_size//2, window_size//2), mode='edge')
                smoothed_kdp = np.convolve(padded_kdp, kernel, mode='valid')
                kdp_ray[valid_kdp] = smoothed_kdp
            

            kdp_ray = np.clip(kdp_ray, -2.0, 12)
            
            # Store result
            kdp_data[i, :] = kdp_ray
    
    if resolution_multiplier > 1:
        kdp_original_res = np.zeros_like(phidp_raw)
        combined_mask_original = mask | refl_mask
        
        for i in range(kdp_data.shape[0]):
            valid_highres = ~np.isnan(kdp_data[i, :]) & ~combined_mask[i, :]
            if np.sum(valid_highres) > 2:
                f_kdp = interp1d(r_highres[valid_highres], kdp_data[i, valid_highres],
                                kind='linear', bounds_error=False, fill_value=np.nan)
                kdp_original_res[i, :] = f_kdp(r)
        
        kdp_data = kdp_original_res
        combined_mask = combined_mask_original
    
    kdp_field = {
        'data': np.ma.masked_array(kdp_data, mask=combined_mask),
        'units': 'degrees/km',
        'long_name': 'Specific differential phase',
        'standard_name': 'KDP',
        '_FillValue': -9999.0,
    }
    
    radar.add_field('DP', kdp_field, replace_existing=True)
    
    print("KDP calculation complete")
    return radar



def dealias_velocity(radar, force_recalculate=False, use_gatefilter=True, 
                    min_refl=30):
    """
    Apply region-based velocity dealiasing algorithm to radar data with gatefilter
    
    Parameters:
    -----------
    radar : pyart.core.Radar
        Radar object containing velocity data
    force_recalculate : bool
        If True, recalculate dealiased velocity even if it already exists
    use_gatefilter : bool
        If True, apply gatefilter to exclude poor quality gates
    min_refl : float
        Minimum reflectivity threshold for gatefilter
    
    Returns:
    --------
    radar : pyart.core.Radar
        Radar object with dealiased velocity field added
    """
    velocity_fields = ['VEL', 'velocity', 'VU', 'VC', 'V1', 'VE', 'VEL_F', 'VF']
    reflectivity_fields = ['reflectivity', 'DBZ', 'REF', 'DZ', 'DBZHC_F', 'DBMHC', 'SNR', 'SN']
    
    if 'VELD' in radar.fields and not force_recalculate:
        return radar
    
    vel_field_name = None
    for field in velocity_fields:
        if field in radar.fields:
            vel_field_name = field
            break
    
    if vel_field_name is None:
        print(f"No velocity field found. Searched for: {velocity_fields}")
        return radar
    
    print(f"Using velocity field: {vel_field_name}")
    
    try:
        gatefilter = None
        if use_gatefilter:
            gatefilter = pyart.filters.GateFilter(radar)
            
            refl_field_name = None
            for field in reflectivity_fields:
                if field in radar.fields:
                    refl_field_name = field
                    break
            
            if refl_field_name is not None:
                gatefilter.exclude_below(refl_field_name, min_refl)
                print(f"Applied reflectivity filter: excluding gates with {refl_field_name} < {min_refl} dB")
            else:
                print(f"No reflectivity field found. Searched for: {reflectivity_fields}")
            
            total_gates = gatefilter.gate_excluded.size
            excluded_gates = np.sum(gatefilter.gate_excluded)
            print(f"GateFilter created - excluding {excluded_gates} of {total_gates} gates")
        
        nyquist_vel = radar.instrument_parameters.get('nyquist_velocity', None)
        
        if nyquist_vel is None:
            print("Nyquist velocity not found in radar metadata, estimating from data...")
            vel_field = radar.fields[vel_field_name]['data']
            nyquist_velocity = float(np.nanmax(np.abs(vel_field)))
        else:
            if isinstance(nyquist_vel, dict) and 'data' in nyquist_vel:
                if isinstance(nyquist_vel['data'], np.ndarray):
                    nyquist_velocity = float(nyquist_vel['data'][0])
                else:
                    nyquist_velocity = float(nyquist_vel['data'])
            else:
                nyquist_velocity = float(nyquist_vel)
            
        print(f"Using Nyquist velocity: {nyquist_velocity} m/s")
        
        dealiased_vel = pyart.correct.dealias_region_based(
            radar,
            vel_field=vel_field_name,
            nyquist_vel=nyquist_velocity,
            gatefilter=gatefilter,
            centered=True,
            skip_between_rays=0,
            skip_along_ray=0
        )
        
        radar.add_field('VELD', dealiased_vel)
        print("Velocity dealiasing completed successfully")
        
    except Exception as e:
        print(f"Error in velocity dealiasing: {e}")
        if vel_field_name in radar.fields:
            vel_shape = radar.fields[vel_field_name]['data'].shape
            empty_data = np.ma.array(
                np.ones(vel_shape) * np.nan,
                mask=np.ones(vel_shape, dtype=bool)
            )
            empty_field = {
                'data': empty_data,
                'units': 'm/s',
                'standard_name': 'VELD',
                'long_name': 'Dealiased Doppler velocity',
                '_FillValue': np.nan,
            }
            radar.add_field('VELD', empty_field)
            print("Added empty dealiased_velocity field due to error")
    
    return radar
    

def add_preset_colorbars(fig, selected_fields, radar, panels, tick_color='white', tick_label_color=None, 
                        bbox_color='black', bbox_alpha=0.7, bbox_pad=0.3):
    """
    Add colorbars with optimized placement based on the number of panels.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object
    selected_fields : list
        List of field tuples (field_name, title, vmin, vmax, cmap)
    radar : pyart.core.Radar
        Radar object containing the fields
    panels : int
        Number of panels (1, 2, or 3)
    tick_color : str or tuple, optional
        Color for the colorbar tick marks (default: 'white')
    tick_label_color : str or tuple, optional
        Color for the colorbar tick labels (default: same as tick_color)
    bbox_color : str or tuple, optional
        Color for the bounding box around tick labels (default: 'black')
    bbox_alpha : float, optional
        Alpha (transparency) for the bounding box (default: 0.7)
    bbox_pad : float, optional
        Padding around the text in the bounding box (default: 0.3)
    """
    if tick_label_color is None:
        tick_label_color = tick_color
    
    if panels == 1:
        rows, cols = 1, 1
        bottom_offset = 0.01
        row_offset = 0.02
        height = 0.03
    elif panels == 2:
        rows, cols = 1, 2
        bottom_offset = 0.01
        row_offset = 0.02
        height = 0.03
    elif panels == 3:
        rows, cols = 1, 3
        bottom_offset = 0.01
        row_offset = 0.02
        height = 0.03
    else:
        raise ValueError("panels must be 1, 2, or 3")
    
    for idx, (field_name, title, vmin, vmax, cmap) in enumerate(selected_fields):
        if field_name not in radar.fields:
            continue
            
        col = idx % cols
        row = idx // cols
        
        ax_pos = fig.axes[idx].get_position()
        
        cax = fig.add_axes([
            ax_pos.x0,                      # Left position
            bottom_offset + (row * row_offset),  # Bottom position (varies by panel count)
            ax_pos.width,                   # Width
            height                          # Height (varies by panel count)
        ])
        
        norm = plt.Normalize(vmin, vmax)
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=cax, orientation='horizontal')
        
        import numpy as np
        tick_range = vmax - vmin
        margin = tick_range * 0.05  # 5% margin from each end
        tick_locations = np.linspace(vmin + margin, vmax - margin, 5)  # 5 ticks between margins
        cb.set_ticks(tick_locations)
        
        cb.ax.tick_params(labelsize=9, pad=2, color=tick_color, labelcolor=tick_label_color)
        cb.outline.set_linewidth(0.5)
        cb.ax.xaxis.set_label_position('top')
        cb.ax.xaxis.tick_top()  # Move ticks to top as well
        
        bbox_props = dict(boxstyle=f"square,pad={bbox_pad}", 
                         facecolor=bbox_color, 
                         alpha=bbox_alpha,
                         edgecolor='none')
        
        for label in cb.ax.get_xticklabels():
            label.set_bbox(bbox_props)

from matplotlib.patches import Circle
import matplotlib.patches as patches

def add_range_rings_and_azimuth_lines(ax, max_range=150, range_interval=5, 
                                    azimuth_interval=20, radar_lat=None, radar_lon=None,
                                    ring_color='white', ring_alpha=1, ring_linewidth=0.8,
                                    azimuth_color='white', azimuth_alpha=1, azimuth_linewidth=0.8,
                                    show_range_labels=True, show_azimuth_labels=True,
                                    label_color='white', label_fontsize=10,
                                    tickskm=None,
                                    xmin=None, xmax=None, ymin=None, ymax=None):
    """
    Add range rings and azimuth lines to a radar PPI plot.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes object to add the rings and lines to
    max_range : float, default=150
        Maximum range for the rings in km
    range_interval : float, default=30
        Interval between range rings in km
    azimuth_interval : float, default=30
        Interval between azimuth lines in degrees
    radar_lat : float, optional
        Radar latitude (for future geographic projections)
    radar_lon : float, optional
        Radar longitude (for future geographic projections)
    ring_color : str, default='white'
        Color of the range rings
    ring_alpha : float, default=0.6
        Transparency of the range rings
    ring_linewidth : float, default=0.8
        Line width of the range rings
    azimuth_color : str, default='white'
        Color of the azimuth lines
    azimuth_alpha : float, default=0.6
        Transparency of the azimuth lines
    azimuth_linewidth : float, default=0.8
        Line width of the azimuth lines
    show_range_labels : bool, default=True
        Whether to show range labels on the rings
    show_azimuth_labels : bool, default=True
        Whether to show azimuth labels
    label_color : str, default='white'
        Color of the labels
    label_fontsize : int, default=10
        Font size of the labels
    tickskm : float, optional
        Interval for tick marks in km. If None, no tick marks are added.
    xmin, xmax, ymin, ymax : float, optional
        Explicit bounds for tick mark placement. If provided, these override
        the automatic bounds detection from axis limits.
    """
    
    # Get current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if tickskm is not None:
        if xmin is not None and xmax is not None and ymin is not None and ymax is not None:
            x_min, x_max = xmin, xmax
            y_min, y_max = ymin, ymax
        else:
            x_min, x_max = xlim
            y_min, y_max = ylim
        

        x_start = np.ceil(x_min / tickskm) * tickskm
        x_end = np.floor(x_max / tickskm) * tickskm
        y_start = np.ceil(y_min / tickskm) * tickskm
        y_end = np.floor(y_max / tickskm) * tickskm
        
        x_ticks = np.arange(x_start, x_end + tickskm, tickskm)
        y_ticks = np.arange(y_start, y_end + tickskm, tickskm)
        
        X_ticks, Y_ticks = np.meshgrid(x_ticks, y_ticks)
        
        x_positions = X_ticks.flatten()
        y_positions = Y_ticks.flatten()
        
        if len(x_positions) > 0:

            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
    
            tick_size = min(x_range, y_range) * 0.01  
    
            for x, y in zip(x_positions, y_positions):
                ax.plot([x - tick_size, x + tick_size], [y, y], 
                       color='white', linewidth=1, alpha=1)
                ax.plot([x, x], [y - tick_size, y + tick_size], 
                       color='white', linewidth=1, alpha=1)
        
 
        x_offset = (x_max - x_min) * 0.02  
        for y_tick in y_ticks:
            if y_min <= y_tick <= y_max:
                label = f'{y_tick:.1f}' if y_tick % 1 != 0 else f'{int(y_tick)}'
                ax.text(x_min + x_offset, y_tick, label, 
                       ha='left', va='center', 
                       color='white', 
                       fontsize=8,
                       fontweight='bold')
        

        y_offset = (y_max - y_min) * 0.02  
        for x_tick in x_ticks:
            if x_min <= x_tick <= x_max:
                label = f'{x_tick:.1f}' if x_tick % 1 != 0 else f'{int(x_tick)}'
                ax.text(x_tick, y_max - y_offset, label, 
                       ha='center', va='top', 
                       color='white', 
                       fontsize=8,
                       fontweight='bold')
    
    # Add range rings
    ranges = np.arange(range_interval, max_range + range_interval, range_interval)
    for range_km in ranges:
        # Only add ring if it's within the plot bounds
        if range_km <= max(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1])):
            circle = Circle((0, 0), range_km, 
                          fill=False, 
                          color=ring_color, 
                          alpha=ring_alpha, 
                          linewidth=ring_linewidth,
                          linestyle='--')
            ax.add_patch(circle)
            
            # Add range labels
            if show_range_labels:
                # Position label at the top of the circle
                ax.text(0, range_km, f'{range_km}', 
                       ha='center', va='bottom', 
                       color=label_color, 
                       fontsize=label_fontsize,
                       fontweight='bold')
    
    azimuths = np.arange(0, 360, azimuth_interval)
    for azimuth in azimuths:
      
        angle_rad = np.radians(90 - azimuth)
        
        end_range = max(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1]))
        x_end = end_range * np.cos(angle_rad)
        y_end = end_range * np.sin(angle_rad)
        
        ax.plot([0, x_end], [0, y_end], 
                color=azimuth_color, 
                alpha=azimuth_alpha, 
                linewidth=azimuth_linewidth,
                linestyle='--')
        
        if show_azimuth_labels:
            label_distance = max_range * 0.85
            if label_distance > max(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1])):
                label_distance = max(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1])) * 0.9
                
            x_label = label_distance * np.cos(angle_rad)
            y_label = label_distance * np.sin(angle_rad)
            
            ax.text(x_label, y_label, f'{int(azimuth)}°', 
                   ha='center', va='center', 
                   color=label_color, 
                   fontweight='bold')

def plot_radar(radar_file, panels=3, max_range=150, xmin=None, xmax=None, ymin=None, ymax=None, 
               output_folder="/mnt/c/users/micha/downloads", calc_kdp=True, force_kdp=False, 
               field_list=None, dealias=True, force_dealias=False, sweep=1,
               add_grid=True, range_interval=5, azimuth_interval=20, tickskm=None):
    """
    Plot radar data with specified number of panels.
    
    New parameters:
    ---------------
    add_grid : bool, default=True
        Whether to add range rings and azimuth lines
    range_interval : float, default=30
        Interval between range rings in km
    azimuth_interval : float, default=30
        Interval between azimuth lines in degrees
    tickskm : float, optional
        Interval for tick marks in km. If None, no tick marks are added.
    """
    radar = pyart.io.read(radar_file, linear_interp=False)
    
    if sweep < 0 or sweep >= radar.nsweeps:
        print(f"WARNING: Sweep {sweep} not available. Radar has {radar.nsweeps} sweeps (0-{radar.nsweeps-1})")
        print(f"Using sweep 0 instead")
        sweep = 0
    
    if calc_kdp:
        radar = calculate_kdp(radar, force_recalculate=force_kdp)
    elif not calc_kdp and 'DP' not in radar.fields:
        print(f"WARNING: KDP calculation disabled for {radar_file}")
        print("Skipping KDP")
    
    if dealias:
        radar = dealias_velocity(radar, force_recalculate=force_dealias)
    elif not dealias and 'VELD' not in radar.fields:
        print(f"WARNING: Velocity dealiasing disabled for {radar_file}")
        print("Skipping velocity dealiasing")
    
    radar_name = radar.metadata.get('instrument_name', 'Unknown Radar')
    
    time_units = radar.time['units']
    time_str = time_units.split('since')[-1].strip()
    try:
        scan_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        time_formatted = scan_time.strftime('%Y-%m-%d %H:%M:%S UTC')
    except:
        time_formatted = time_str
    
    if panels == 1:
        rows, cols = 1, 1
        selected_fields = fields[0:1]  # Just reflectivity
        figsize = (8, 8)
        bottom_margin = 0.15
    elif panels == 2:
        rows, cols = 1, 2
        selected_fields = fields[0:2]  # Reflectivity and velocity
        figsize = (12, 6)
        bottom_margin = 0.15
    elif panels == 3:
        rows, cols = 1, 3
        selected_fields = fields[0:3]  # Reflectivity, velocity, and KDP
        figsize = (18, 6)  # Wider to accommodate 3 panels
        bottom_margin = 0.15
    else:
        raise ValueError("panels must be 1, 2, or 3")
    
    if field_list is not None:
        selected_fields = field_list
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    tilt_angle = radar.fixed_angle['data'][sweep]  # Use the selected sweep

    solo3_title = f"{time_formatted} {radar_name} {tilt_angle:.1f}°"
    
    # Set layout parameters based on panel count
    if panels == 1:
        bottom = 0.01
        left = 0.055
        right = 0.97
        title_x_pos = 0.29
        fontsize = 12
    elif panels == 2:
        bottom = 0.01
        left = 0.022
        right = 0.97
        title_x_pos = 0.26
        fontsize = 8
    elif panels == 3:
        bottom = 0.01
        left = 0.011
        right = 0.97
        title_x_pos = 0.26
        fontsize = 8
    else:
        bottom = 0.01
        left = 0.033
        right = 0.97
        title_x_pos = 0.29
        fontsize = 12
    
    gs_main = plt.GridSpec(rows, cols, figure=fig, 
                         height_ratios=[1] * rows, 
                         hspace=0.04, #space between panels vertically
                         wspace=0.03, #space between panels horizontally
                         bottom=bottom,  #reduced from bottom_margin to minimize whitespace
                         top=0.95,  #more room since no suptitle
                         left=left,
                         right=right)
    
    display = pyart.graph.RadarDisplay(radar)
    
    for idx, (field_name, title, vmin, vmax, cmap) in enumerate(selected_fields):
        row = idx // cols
        col = idx % cols
        ax = fig.add_subplot(gs_main[row, col])
        
        try:
            if field_name == 'DP' and not calc_kdp and field_name not in radar.fields:
                ax.text(0.5, 0.5, "KDP calculation disabled", 
                        ha='center', va='center', transform=ax.transAxes)
                continue
                
            if field_name == 'VELD' and not dealias and field_name not in radar.fields:
                ax.text(0.5, 0.5, "Velocity dealiasing disabled", 
                        ha='center', va='center', transform=ax.transAxes)
                continue
                
            display.plot_ppi(
                field_name,
                sweep=sweep,
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                colorbar_flag=False,
                title=title,
                axislabels=(f'X-Distance from {radar_name} (km)', '(km)'),
                axislabels_flag=True, edges=False, 
                filter_transitions=False)
            
            x_min = xmin if xmin is not None else -max_range
            x_max = xmax if xmax is not None else max_range
            y_min = ymin if ymin is not None else -max_range
            y_max = ymax if ymax is not None else max_range
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            ax.tick_params(direction='in', length=4, width=1, colors='white')
            ax.tick_params(which='minor', direction='in', length=2, width=0.5, colors='white')
            
            ax.tick_params(top=True, bottom=True, left=True, right=True)
            ax.tick_params(labeltop=False, labelbottom=True, labelleft=True, labelright=False)
            
            range_span = max(x_max - x_min, y_max - y_min)
            
            if range_span <= 10:
                major_interval = 2
            elif range_span <= 20:
                major_interval = 5
            elif range_span <= 40:
                major_interval = 10
            elif range_span <= 100:
                major_interval = 20
            elif range_span <= 300:
                major_interval = 50
            else:
                major_interval = 100
                
            x_tick_start = np.ceil(x_min / major_interval) * major_interval
            x_tick_end = np.floor(x_max / major_interval) * major_interval
            y_tick_start = np.ceil(y_min / major_interval) * major_interval
            y_tick_end = np.floor(y_max / major_interval) * major_interval
            
            x_major_ticks = np.arange(x_tick_start, x_tick_end + major_interval, major_interval)
            y_major_ticks = np.arange(y_tick_start, y_tick_end + major_interval, major_interval)
            
            x_major_ticks = x_major_ticks[(x_major_ticks >= x_min) & (x_major_ticks <= x_max)]
            y_major_ticks = y_major_ticks[(y_major_ticks >= y_min) & (y_major_ticks <= y_max)]
            
            ax.set_xticks(x_major_ticks)
            ax.set_yticks(y_major_ticks)
            

            if add_grid:
                add_range_rings_and_azimuth_lines(
                    ax, 
                    max_range=max_range,
                    range_interval=range_interval,
                    azimuth_interval=azimuth_interval,
                    ring_color='white',
                    ring_alpha=1,
                    ring_linewidth=0.8,
                    azimuth_color='white',
                    azimuth_alpha=1,
                    azimuth_linewidth=0.8,
                    show_range_labels=True,
                    show_azimuth_labels=True,
                    label_color='white',
                    label_fontsize=8,
                    tickskm=tickskm,
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax
                )
                
                for special_range in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
                    if special_range <= max_range and special_range not in range(range_interval, max_range + range_interval, range_interval):
                        circle = Circle((0, 0), special_range, 
                                      fill=False, 
                                      color='white', 
                                      alpha=0.6, 
                                      linewidth=0.8,
                                      linestyle='--')
                        ax.add_patch(circle)
                        
                        ax.text(0, special_range, f'{special_range}', 
                               ha='center', va='bottom', 
                               color='white', 
                               fontsize=8,
                               fontweight='bold')
            
            if col != 0:
                ax.set_yticklabels([])
                ax.set_ylabel('')
            
            if row != rows - 1:
                ax.set_xticklabels([])
                ax.set_xlabel('')
                
            # Reduce tick padding
            ax.tick_params(pad=2)
                
        except KeyError:
            ax.text(0.5, 0.5, f"Field '{field_name}' not found", 
                    ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title(title, fontsize=12, pad=4, fontweight='bold')
        
        ax.text(title_x_pos, 0.995, solo3_title, 
                transform=ax.transAxes, 
                fontsize=fontsize, 
                fontweight='bold',
                color='black',
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='square,pad=0.3', 
                         facecolor='white', 
                         alpha=1,
                         edgecolor='none'))
    
    add_preset_colorbars(fig, selected_fields, radar, panels)
    
    datetime_str = time_str.replace(' ', '_').replace(':', '')
    radar_name_clean = radar_name.replace(' ', '_')
    
    output_filename = os.path.join(output_folder, f"{radar_name_clean}_{datetime_str}_{panels}.png")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    plt.savefig(output_filename, dpi=500, facecolor=fig.get_facecolor())
    print(f"Saved figure as {output_filename}")
    plt.close()
    
    return output_filename

def process_folder(folder_path, panels=3, max_range=150, xmin=None, xmax=None, ymin=None, ymax=None, output_folder="/mnt/c/users/micha/downloads", calc_kdp=True, force_kdp=False, field_list=None, file_extensions=None, dealias=True, force_dealias=False, sweep=1, tickskm=None):
    """
    Process all radar files in a folder using concurrent processing.
    """
    import concurrent.futures
    from functools import partial
    
    if file_extensions is None:
        file_extensions = ['.nc', '.h5', '.buf', '.raw', '.gz', '_V06', '.msg31', '.0', '.ar2v', '.RAW']
    
    os.makedirs(output_folder, exist_ok=True)
    
    radar_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in file_extensions):
                radar_files.append(os.path.join(root, file))
    
    if not radar_files:
        print(f"No radar files found in {folder_path} with extensions {file_extensions}")
        return
    
    print(f"Found {len(radar_files)} radar files to process")
    
    process_file = partial(
        plot_radar,
        panels=panels,
        max_range=max_range,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        output_folder=output_folder,
        calc_kdp=calc_kdp,
        force_kdp=force_kdp,
        field_list=field_list,
        dealias=dealias,
        force_dealias=force_dealias,
        sweep=sweep,
        tickskm=tickskm
    )
    

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, file): file for file in radar_files}
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                output_file = future.result()
                completed += 1
                print(f"Progress: {completed}/{len(radar_files)} files processed")
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    print(f"All files have been processed. Output images saved to {output_folder}")

if __name__ == "__main__":
    import argparse
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pyart
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Plot radar data with customizable panel layouts')
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file', help='Path to a single radar file')
    input_group.add_argument('--folder', help='Path to a folder containing radar files')
    
    parser.add_argument('--panels', type=int, choices=[1, 2, 3], default=2, 
                        help='Number of panels to display (1, 2, or 3)')
    parser.add_argument('--max_range', type=float, default=150,
                        help='Maximum range to display in km (for both X and Y axes if not specified separately)')
    parser.add_argument('--xmin', type=float, default=None,
                        help='Minimum X axis value to display in km')
    parser.add_argument('--xmax', type=float, default=None,
                        help='Maximum X axis value to display in km')
    parser.add_argument('--ymin', type=float, default=None,
                        help='Minimum Y axis value to display in km')
    parser.add_argument('--ymax', type=float, default=None,
                        help='Maximum Y axis value to display in km')
    parser.add_argument('--output_folder', default="/mnt/c/users/micha/downloads",
                        help='Folder to save the output image(s)')
    parser.add_argument('--sweep', type=int, default=1,
                        help='Sweep number to plot (default: 1). Use 0 for lowest tilt angle.')
    parser.add_argument('--no-kdp', action='store_false', dest='calc_kdp',
                        help='Disable KDP calculation (use existing KDP data if available)')
    parser.add_argument('--force-kdp', action='store_true',
                        help='Force recalculation of KDP even if it already exists')
    parser.add_argument('--no-dealias', action='store_false', dest='dealias',
                        help='Disable velocity dealiasing (use existing dealiased velocity data if available)')
    parser.add_argument('--force-dealias', action='store_true',
                        help='Force recalculation of dealiased velocity even if it already exists')
    parser.add_argument('--fields', nargs='+', type=str, default=None,
                        help='Custom list of field names to plot (must match radar fields)')
    parser.add_argument('--extensions', nargs='+', type=str, default=['.nc', '.h5', '.buf', '.raw', '.gz', '_V06', '.msg31', '.0', '.ar2v',  '.RAW'],
                        help='File extensions to process when scanning a folder (default: .nc .h5 .buf .raw _V06 .msg31 .0)')
    parser.add_argument('--tickskm', type=float, default=None,
                        help='Interval for tick marks in km (e.g., --tickskm 30). If not specified, no tick marks are added.')
    
    
    args = parser.parse_args()
    
    custom_fields = None
    if args.fields:
        field_map = {f[0]: f for f in fields}
        custom_fields = []
        for field_name in args.fields:
            if field_name in field_map:
                custom_fields.append(field_map[field_name])
            else:
                print(f"WARNING: Field '{field_name}' not found in predefined field list.")
                print(f"Available fields: {', '.join(field_map.keys())}")
        
        if not custom_fields:
            print("ERROR: No valid fields specified. Using default fields.")
            custom_fields = None
    
    if args.file:
        print(f"Processing single radar file: {args.file}")
        plot_radar(
            args.file, 
            args.panels, 
            args.max_range,
            args.xmin,
            args.xmax,
            args.ymin,
            args.ymax, 
            args.output_folder,
            calc_kdp=args.calc_kdp,
            force_kdp=args.force_kdp,
            field_list=custom_fields,
            dealias=args.dealias,
            force_dealias=args.force_dealias,
            sweep=args.sweep,
            tickskm=args.tickskm
        )
    else:
        print(f"Processing all radar files in folder: {args.folder}")
        process_folder(
            args.folder,
            args.panels,
            args.max_range,
            args.xmin,
            args.xmax,
            args.ymin,
            args.ymax,
            args.output_folder,
            calc_kdp=args.calc_kdp,
            force_kdp=args.force_kdp,
            field_list=custom_fields,
            file_extensions=args.extensions,
            dealias=args.dealias,
            force_dealias=args.force_dealias,
            sweep=args.sweep,
            tickskm=args.tickskm
        )