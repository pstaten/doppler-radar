from numpy.ma import array as ma_array
from numpy.ma import masked as ma_masked
from numpy import array as np_array
from numpy import deg2rad as np_deg2rad
from numpy import sin as np_sin
from numpy import cos as np_cos
from numpy import interp as np_interp
from numpy import arange as np_arange
from numpy import isnan as np_isnan
from numpy import newaxis as np_newaxis
from cartopy.crs import LambertConformal as ccrs_LambertConformal
import cartopy.feature as cfeature
from matplotlib import cm, colors, rcParams
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from metpy.plots import add_timestamp, ctables
import metpy.plots as mpplots

def raw_to_masked_float(var, data):
    # Values come back signed. If the _Unsigned attribute is set, we need to convert
    # from the range [-127, 128] to [0, 255].
    if var._Unsigned:
        data = data & 255

    # Mask missing points
    data = ma_array(data, mask=data==0)

    # Convert to float using the scale and offset
    return data * var.scale_factor + var.add_offset

def polar_to_cartesian(az, rng):
    az_rad = np_deg2rad(az)[:, None]
    x = rng * np_sin(az_rad)
    y = rng * np_cos(az_rad)
    return x, y

# We import MetPy and use it to get the colortable and value mapping information for the NWS Reflectivity data.

def new_map(fig, lon, lat):
    # Create projection centered on the radar. This allows us to use x
    # and y relative to the radar.
    proj = ccrs_LambertConformal(central_longitude=lon, central_latitude=lat)

    # New axes with the specified projection
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], projection=proj)

    # Add coastlines and states
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=2)
    ax.add_feature(cfeature.STATES.with_scale('50m'))
    
    return ax

def inter_from_256(x):
    return np_interp(x=x,xp=[0,255],fp=[0,1])

def get_custom_cmaps():
    zdr_cdict = {
        'red':((0.00,  inter_from_256(0),  inter_from_256(0)), # black, black
               (0.25,inter_from_256(144),inter_from_256(144)), # purple
               (0.42,inter_from_256(160), inter_from_256(57)), # lavender, dark blue
               (0.46,inter_from_256(157),inter_from_256(157)), # aqua
               (0.50,inter_from_256(164),inter_from_256(164)), # chartreuse
               (0.52,inter_from_256(237),inter_from_256(237)), # yellow
               (0.56,inter_from_256(237),inter_from_256(237)),
               (0.62,inter_from_256(222),inter_from_256(222)),
               (0.70,inter_from_256(171),inter_from_256(171)),
               (0.80,inter_from_256(219),inter_from_256(219)),
               (0.85,inter_from_256(255),inter_from_256(255)),
               (1.00,inter_from_256(123),inter_from_256(123))),
        'green':((0.0,  inter_from_256(0),  inter_from_256(0)),
                (0.25,inter_from_256(123),inter_from_256(123)),
                (0.42,inter_from_256(162), inter_from_256(63)),
                (0.46,inter_from_256(228),inter_from_256(228)),
                (0.50,inter_from_256(212),inter_from_256(212)),
                (0.52,inter_from_256(228),inter_from_256(228)),
                (0.56,inter_from_256(143),inter_from_256(143)),
                (0.62, inter_from_256(48), inter_from_256(48)),
                (0.70, inter_from_256(41), inter_from_256(41)),
                (0.80,inter_from_256(123),inter_from_256(123)),
                (0.85,inter_from_256(255),inter_from_256(255)),
                (1.00, inter_from_256(70), inter_from_256(70))),
        'blue':((0.00,  inter_from_256(0),  inter_from_256(0)),
                (0.25,inter_from_256(184),inter_from_256(184)),
                (0.42,inter_from_256(198),inter_from_256(160)),
                (0.46,inter_from_256(214),inter_from_256(214)),
                (0.50, inter_from_256(84), inter_from_256(84)),
                (0.52,inter_from_256(111),inter_from_256(111)),
                (0.56, inter_from_256(22), inter_from_256(22)),
                (0.62, inter_from_256(26), inter_from_256(26)),
                (0.70, inter_from_256(39), inter_from_256(39)),
                (0.80,inter_from_256(168),inter_from_256(168)),
                (0.85,inter_from_256(255),inter_from_256(255)),
                (1.00,inter_from_256(131),inter_from_256(131))),
    }
    ccp_cdict = {
        'red':((0.00, inter_from_256(97), inter_from_256(97)),
               (0.30, inter_from_256(91), inter_from_256(55)),
               (0.70,inter_from_256(131),inter_from_256(131)),
               (0.72,inter_from_256(124),inter_from_256(124)),
               (0.80,inter_from_256(161),inter_from_256(161)),
               (0.88,inter_from_256(249),inter_from_256(249)),
               (1.00,inter_from_256(152),inter_from_256(97))),
        'green':((0.0, inter_from_256(97), inter_from_256(97)),
                (0.30, inter_from_256(91), inter_from_256(62)),
                (0.70,inter_from_256(158),inter_from_256(158)),
                (0.72,inter_from_256(199),inter_from_256(199)),
                (0.80,inter_from_256(208),inter_from_256(208)),
                (0.88,inter_from_256(227),inter_from_256(227)),
                (1.00, inter_from_256(38), inter_from_256(97))),
        'blue':((0.00, inter_from_256(97), inter_from_256(97)),
                (0.30, inter_from_256(91),inter_from_256(135)),
                (0.70,inter_from_256(195),inter_from_256(195)),
                (0.72,inter_from_256(184),inter_from_256(184)),
                (0.80, inter_from_256(32), inter_from_256(32)),
                (0.88, inter_from_256(51), inter_from_256(51)),
                (1.00, inter_from_256(69), inter_from_256(97))),
    }
    phi_cdict = {
        'red':((0.000, inter_from_256(97), inter_from_256(167)), # gray | light lavender
               (0.167, inter_from_256(100), inter_from_256(81)), # purple | sky
               (0.333,inter_from_256(55),inter_from_256(113)), # indigo | chartreuse
               (0.500,inter_from_256(47),inter_from_256(250)), # green | yellow
               (0.667,inter_from_256(239),inter_from_256(222)), # orange | red
               (0.833,inter_from_256(155),inter_from_256(182)), # maroon | pink
               (1.00,inter_from_256(141),inter_from_256(97))), # plum | gray
        'green':((0.00, inter_from_256(97), inter_from_256(191)),
                (0.167, inter_from_256(105), inter_from_256(144)),
                (0.333,inter_from_256(62),inter_from_256(198)),
                (0.500,inter_from_256(130),inter_from_256(229)),
                (0.667,inter_from_256(127),inter_from_256(44)),
                (0.833,inter_from_256(41),inter_from_256(112)),
                (1.00, inter_from_256(53), inter_from_256(97))),
        'blue':((0.000, inter_from_256(97), inter_from_256(200)),
                (0.167, inter_from_256(180),inter_from_256(196)),
                (0.333,inter_from_256(136),inter_from_256(11)),
                (0.500,inter_from_256(50),inter_from_256(60)),
                (0.667, inter_from_256(33), inter_from_256(33)),
                (0.833, inter_from_256(36), inter_from_256(172)),
                (1.00, inter_from_256(153), inter_from_256(97))),
    }

    zdr_cmap = colors.LinearSegmentedColormap('zdr_cmap',segmentdata=zdr_cdict)
    ccp_cmap = colors.LinearSegmentedColormap('ccp_cmap',segmentdata=ccp_cdict)
    phi_cmap = colors.LinearSegmentedColormap('phi_cmap',segmentdata=phi_cdict)

    return(zdr_cmap, ccp_cmap, phi_cmap)


def plot_six_panel(f, title='a super duper pretty six-panel'):
    print('Subsetting Data...')
    '''With the file comes a lot of data, including multiple elevations and products.
    In the next block, we'll pull out the specific data we want to plot.'''

    # I think each sweep corresponds to a different zenith angle,
    # but not all variables are available for all the different sweeps,
    # and it seem the elevation is not consistent within a given sweep,
    # so I'm not sure.
    print('(There are {} sweeps in this file. If you get zero sweeps, try downloading a data file before or after this one.)'.format(len(f.sweeps)))
    sweep0 = 0 # used for REF, RHO, PHI, ZDR, or CFP (not used here)
    sweep1 = 1 # used for VEL, SW

    # First item in ray is header, which has azimuth angle
    az0 = np_array([ray[0].az_angle for ray in f.sweeps[sweep0]])
    az1 = np_array([ray[0].az_angle for ray in f.sweeps[sweep1]])

    # Read in just the information we need to plot each of these fields for one RHI scan
    ref_desc = r'reflectivity (Z)'
    ref_hdr = f.sweeps[sweep0][0][4][b'REF'][0]
    ref_range = np_arange(ref_hdr.num_gates) * ref_hdr.gate_width + ref_hdr.first_gate
    ref = np_array([ray[4][b'REF'][1] for ray in f.sweeps[sweep0]])

    rho_desc = r'copolar correlation coefficient ($\rho_{hv}$)'
    rho_hdr = f.sweeps[sweep0][0][4][b'RHO'][0]
    rho_range = (np_arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
    rho = np_array([ray[4][b'RHO'][1] for ray in f.sweeps[sweep0]])

    zdr_desc = r'differential reflectivity (Z$_{DR}$)'
    zdr_hdr = f.sweeps[sweep0][0][4][b'ZDR'][0]
    zdr_range = (np_arange(zdr_hdr.num_gates + 1) - 0.5) * zdr_hdr.gate_width + zdr_hdr.first_gate
    zdr = np_array([ray[4][b'ZDR'][1] for ray in f.sweeps[sweep0]])

    phi_desc = r'MYSTERY VARIABLE!!!!!'
    # Don't look too closely at these next few lines of code. They'll spoil the answer.
    phi_hdr = f.sweeps[sweep0][0][4][b'PHI'][0]
    phi_range = (np_arange(phi_hdr.num_gates + 1) - 0.5) * phi_hdr.gate_width + phi_hdr.first_gate
    phi = np_array([ray[4][b'PHI'][1] for ray in f.sweeps[sweep0]])

    vel_desc = r'radial velocity ($v_r$)'
    vel_hdr = f.sweeps[sweep1][0][4][b'VEL'][0]
    vel_range = (np_arange(vel_hdr.num_gates + 1) - 0.5) * vel_hdr.gate_width + vel_hdr.first_gate
    vel = np_array([ray[4][b'VEL'][1] for ray in f.sweeps[sweep1]])

    sw_desc = r'spectrum width ($\sigma_v$)'
    sw_hdr = f.sweeps[sweep1][0][4][b'SW'][0]
    sw_range = (np_arange(sw_hdr.num_gates + 1) - 0.5) * sw_hdr.gate_width + sw_hdr.first_gate
    sw = np_array([ray[4][b'SW'][1] for ray in f.sweeps[sweep1]])

    # Get the NWS reflectivity colortable from MetPy
    ref_norm, ref_cmap = ctables.registry.get_with_steps('NWSReflectivity', 5, 5)
    vel_norm, vel_cmap = ctables.registry.get_with_steps('NWSVelocity', 5, 5)

    # Get the color tables I made just for you. You're welcome!
    zdr_cmap, rho_cmap, phi_cmap = get_custom_cmaps()

    print('Plotting the data...')
    # Plot the data! Yay!
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    for ialpha, var_data, var_range, icolors, lbl, ax, desc in zip(
            ('(a) ', '(b) ', '(c) ', '(d) ', '(e) ', '(f) '),
            (ref, rho, zdr, phi, vel, sw),
            (ref_range, rho_range, zdr_range, phi_range, vel_range, sw_range),
            (ref_cmap, rho_cmap, zdr_cmap, phi_cmap, vel_cmap, 'viridis'),
            ('dBZ', '', 'dBZ', 'units of mystery', 'm/s' , r'm/s'),
            axes.flatten(),
            (ref_desc, rho_desc, zdr_desc, phi_desc, vel_desc, sw_desc)):
        # Turn into an array, then mask
        data = ma_array(var_data)
        data[np_isnan(data)] = ma_masked

        # Convert azimuth,range to x,y
        if lbl in ['VEL', 'SW']:
            xlocs = var_range * np_sin(np_deg2rad(az1[:, np_newaxis]))
            ylocs = var_range * np_cos(np_deg2rad(az1[:, np_newaxis]))
        else:
            xlocs = var_range * np_sin(np_deg2rad(az0[:, np_newaxis]))
            ylocs = var_range * np_cos(np_deg2rad(az0[:, np_newaxis]))

        # Define norm for reflectivity.
        # If you understand what the norms are for colorbars, please tell me.
        # I'm too lazy to google it right now.
        # What!? I've been doing a lot of googling today, okay? =)
        norm = ref_norm if icolors == ref_cmap else None

        # Plot the data
        a = ax.pcolormesh(xlocs, ylocs, data, cmap=icolors, norm=norm)

        # I've never done this before, but I'm stoked so see this code
        # so I can insert colortables to the right of individual panels
        # without crowding neighboring plots!
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        #fig.colorbar(a, cax=cax, orientation='vertical', label=lbl)
        fig.colorbar(a, cax=cax, label=lbl)

        # Are you still reading this?
        #
        ax.set_aspect('equal', 'datalim')
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_title(ialpha + desc)
        add_timestamp(ax, f.dt, y=0.02, high_contrast=False)
    plt.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.show()

def animate_radar_loop(cat, vel=False):
    ds = cat.datasets[0]
    data = ds.remote_access()
    sLon, sLat = data.StationLongitude, data.StationLatitude
    if vel:
        var_norm, var_cmap = mpplots.ctables.registry.get_with_steps('NWSVelocity', -80, 10)
    else:
        var_norm, var_cmap = mpplots.ctables.registry.get_with_steps('NWSReflectivity', 5, 5)


    # Create the figure with the lat and lon limits from the radar station
    plt.close() # close anything before so there's no weird overlapping
    fig = plt.figure(figsize=(10, 7.5))
    ax = new_map(fig, sLon, sLat)
    ax.set_extent([sLon-3, sLon+4, sLat-2, sLat+2])
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAND.with_scale('50m'))

    meshes = []
    '''Now we can loop over the collection of returned datasets and plot them.
    As we plot, we collect the returned plot objects so that we can use them
    to make an animated plot. We also add a timestamp for each plot.'''
    for ds_name in cat.datasets:
        # After looping over the list of sorted datasets, pull the actual Dataset object out
        # of our list of items and access over CDMRemote
        data = cat.datasets[ds_name].remote_access()

        # Pull out the data of interest
        sweep = 0
        if vel:
            rng = data.variables['distanceV_HI'][:]
            az = data.variables['azimuthV_HI'][sweep]
            plot_var = data.variables['RadialVelocity_HI']
        else:
            rng = data.variables['distanceR_HI'][:]
            az = data.variables['azimuthR_HI'][sweep]
            plot_var = data.variables['Reflectivity_HI']

        # Convert data to float and coordinates to Cartesian
        toplot = raw_to_masked_float(plot_var, plot_var[sweep])
        x, y = polar_to_cartesian(az, rng)

        # Plot the data and the timestamp
        mesh = ax.pcolormesh(x, y, toplot, cmap=var_cmap, norm=var_norm, zorder=0)
        text = ax.text(0.7, 0.02, data.time_coverage_start, transform=ax.transAxes,
                       fontdict={'size':16})
        
        # Collect the things we've plotted so we can animate
        meshes.append((mesh, text))

    '''Using matplotlib, we can take a collection of ``Artists``
    that have been plotted and turn them into an animation.
    Using the FFMpeg utility, this animation can be converted
    to HTML5 video viewable in the notebook.'''
    # Set up matplotlib to do the conversion to HTML5 video
    rcParams['animation.html'] = 'html5'

    # return to create an animation
    return(fig, meshes)
