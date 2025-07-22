import sqlite3
import numpy as np
from PIL import Image
import tifffile as tifffile
import os
from datetime import datetime, timedelta

def crxreader(experiment_file, **kwargs):
    """
    Reads and processes data from a CellReporterXpress experiment SQLite database.

    Parameters:
        experiment_file (str): Path to the experiment file.
        verbose (int, optional): Verbosity level for output logs (default: 0).
        channel (int, optional): Channel number to process (default: 1).
        well (str, optional): Identifier for the specific well to process (default: None).
        tile (str or int, optional): Specific tile to process or 'all' for all tiles (default: None).
        level (int, optional): Image pyramid level to process (default: 0).
        time_point (int, optional): TimePoint value corresponding to TimeSeriesElementId (default: 0).
        save_as (str, optional): Path to save the processed image(s) (default: None).
        tiff_compression (str, optional): TIFF compression type; options are 'none', 'lzw', or 'deflate' (default: 'deflate').
        info (dict, optional): Preloaded dictionary for experiment metadata (default: None).
        show_well_matrix (bool, optional): If True, display a matrix of wells used for each timepoint (default: False).

    Returns:
        dict: Metadata about the experiment (if no well or tile is specified).
        numpy.ndarray: Processed image data (if a well or tile is specified and valid).

    Usage Examples:
        # Example 1: Extract metadata from an experiment file
        info = crxreader('path/to/experiment.db')

        # Example 2: Process and save a specific well and tile as a TIFF image
        imdata = crxreader('path/to/experiment.db', well='B2', tile=1, save_as='output.tif', info=info)

        # Example 3: Process an image at a specific level and save it
        imdata = crxreader('path/to/experiment.db', well='A1', level=3, save_as='output_level3.tif')

    Notes:
        - Ensure the experiment file exists and is accessible.
        - Supported image formats for saving are '.tif' and '.png'.
        - The function supports processing single tiles or all tiles for a well.
        - The metadata includes well and channel details, experiment information, and image acquisition parameters.

    License:
        This code is distributed under the MIT License.
    """

    # Set default values
    verbose = kwargs.get('verbose', 0)
    channel = kwargs.get('channel', 1)
    well = kwargs.get('well', None)
    tile = kwargs.get('tile', None)
    level = kwargs.get('level', 0)
    time_point = kwargs.get('time_point', 0)  # Added TimePoint parameter
    save_as = kwargs.get('save_as', None)
    tiff_compression = kwargs.get('tiff_compression', 'deflate')
    info = kwargs.get('info', None)
    show_well_matrix = kwargs.get('show_well_matrix', False)  # Added parameter to display well matrix

    well = remove_leading_zero(well)  # Remove leading zero from well name

    # Check if the file exists
    is_ok = False
    if os.path.isfile(experiment_file):
        if not info:
            info = {}
            info['experiment_file'] = experiment_file
            filepath = os.path.dirname(experiment_file)
            info['images_file'] = None  # To be determined dynamically
            do_disp('Reading Info from CellReporterXpress experiment.db file', verbose)
            try:
                # Connect to the SQLite database
                with sqlite3.connect(experiment_file) as conn:
                    cur = conn.cursor()

                    # Check for TimeSeriesElementId values
                    cur.execute("SELECT DISTINCT TimeSeriesElementId FROM SourceImageBase")
                    time_series_ids = [row[0] for row in cur.fetchall()]

                    if len(time_series_ids) == 1 and time_series_ids[0] == 0:
                        # Only one TimeSeriesElementId (0), ignore time_point
                        time_point = 0
                        info['images_file'] = os.path.join(filepath, 'images-0.db')
                    else:
                        # Determine the appropriate images-x.db file for the given TimePoint
                        if time_point not in time_series_ids:
                            raise ValueError(f"Invalid TimePoint: {time_point}. Available values: {time_series_ids}")
                        info['images_file'] = os.path.join(filepath, f'images-{time_point}.db')

                    # Fetch additional metadata
                    cur.execute('SELECT DateCreated, Creator, Name FROM ExperimentBase')
                    rs = cur.fetchone()
                    dt = convert_dotnet_ticks_to_datetime(int(rs[0]))
                    info.update({
                        'name': rs[2],
                        'creator': rs[1],
                        'dt': dt,
                    })

                    # Fetch additional info required to read well info
                    cur.execute('SELECT SensorSizeYPixels, SensorSizeXPixels, Objective, PixelSizeUm, SensorBitness, SitesX, SitesY FROM AcquisitionExp, AutomaticZonesParametersExp')
                    rsdI = cur.fetchone()
                    cur.execute('SELECT Emission, Excitation, Dye, channelNumber, ColorName FROM ImagechannelExp')
                    rsdC = cur.fetchall()

                    # Process and store well information
                    info['well_info'] = read_well_info(rsdI, rsdC)

                    # Fetch well data
                    cur.execute('SELECT Name, ZoneIndex FROM well WHERE HasImages = 1')
                    wells = cur.fetchall()
                    info['numwells'] = len(wells)
                    info['wells'] = {well[0]: well[1] for well in wells}
                is_ok = True
            except sqlite3.Error as e:
                do_disp(f'Error Reading Info: {str(e)}', verbose)
        else:
            is_ok = True
            do_disp('Using given Info!', verbose)
    else:
        do_disp('Error: File not found!', verbose)

    if show_well_matrix and is_ok:
        display_well_matrix(info, experiment_file)
        return info

    if not well and not tile and is_ok:
        return info

    # Read image data if well is specified
    if well and is_ok:
        if well in info['wells']:
            zone_index = info['wells'][well]
            with sqlite3.connect(info['experiment_file']) as conn:
                cur = conn.cursor()
                cur.execute('''
                    SELECT CoordX, CoordY, SizeX, SizeY, BitsPerPixel, ImageIndex, channelId 
                    FROM SourceImageBase
                    WHERE ZoneIndex = ? AND level = ? AND TimeSeriesElementId = ?
                    ORDER BY CoordX ASC, CoordY ASC
                ''', (zone_index, level, time_point))
                zd = cur.fetchall()
            conn.close()

            # Filter the data for the specified channel
            zd = [d for d in zd if d[6] == channel - 1]
            if zd:
                # Compute the dimensions of the Well image
                a=np.asarray(zd)
                xmax= a[a[:,1]==0,2].sum()
                ymax= a[a[:,0]==0,3].sum()
                # Initialize the image data array
                imdata = np.zeros((ymax, xmax), dtype=np.uint16)

                # Read binary data from the image file
                with open(info['images_file'], 'rb') as fid:
                    for data in zd:
                        fid.seek(data[5])
                        # The following reads and inserts a sub-image into the larger image array
                        sub_image_data = np.fromfile(fid, dtype=np.uint16, count=data[2]*data[3])
                        sub_image_data = sub_image_data.reshape((data[3],data[2]))
                        imdata[data[1]:data[1]+data[3],data[0]:data[0]+data[2]] = sub_image_data

                if not tile:
                    do_save(imdata, save_as, tile, channel, level, well, tiff_compression, verbose, info['well_info'])
                    do_disp('Ready Reading Well',verbose)
                    return imdata
                elif not level : # If tile is specified and full level is requested
                    if isinstance(tile, str):
                        if tile.lower() == 'all':
                            outdata = []
                            tx = 0
                            ty = 1
                            for tile in range(info['well_info']['tiles']):
                                tx += 1
                                if tx > info['well_info']['tilex']:
                                    tx = 1
                                    ty += 1
                                txs = (tx-1) * info['well_info']['xs']
                                tys = (ty-1) * info['well_info']['ys']
                                xs = txs + info['well_info']['xs']
                                ys = tys + info['well_info']['ys']
                                outdata.append(imdata[tys:ys,txs:xs])
                            do_save(outdata, save_as, tile, channel, level, well, tiff_compression, verbose, info['well_info'])
                            do_disp(f'Ready Reading all tiles from well: {well}',verbose)
                        else:
                            do_disp(f'Error Reading Well, tile {tile} does not exist',verbose)
                            return None
                    else: # if single tile is specified
                        if tile > 0 and tile <= info['well_info']['tiles']:
                            # Calculate tile position
                            tx = (tile - 1) % info['well_info']['tilex']
                            ty = (tile - 1) // info['well_info']['tilex']
                            xs = tx * info['well_info']['xs']
                            ys = ty * info['well_info']['ys']
                            xse = xs + info['well_info']['xs'] 
                            yse = ys + info['well_info']['ys']                       

                            # Find the bounding coordinates for the tile
                            zdx = sorted(set(row[0] for row in zd))
                            zdy = sorted(set(row[1] for row in zd))
                            xmin = max(x for x in zdx if x <= xs)
                            xmax = min((x for x in zdx if x >= xse), default=xse)
                            ymin = max(y for y in zdy if y <= ys)
                            ymax = min((y for y in zdy if y >= yse), default=yse)

                            # Filter zd for the current tile
                            zd = [row for row in zd if xmin <= row[0] <= xmax and ymin <= row[1] <= ymax]

                            if xmax > xmin and ymax > ymin:
                                imdata = np.zeros((ymax - ymin,xmax - xmin), dtype=np.uint16)

                                # Read and populate image data
                                with open(info['images_file'], 'rb') as fid:
                                    for row in zd:
                                        # Calculate the position and size of the data to be inserted
                                        xstart = row[0] - xmin
                                        ystart = row[1] - ymin
                                        xend = xstart + row[2]
                                        yend = ystart + row[3]

                                        # Check if the indices are within the bounds of imdata
                                        if 0 <= xstart < imdata.shape[1] and 0 <= ystart < imdata.shape[0]:
                                            # Ensure we do not exceed the bounds of the array
                                            xend = min(xend, imdata.shape[1])
                                            yend = min(yend, imdata.shape[0])

                                            # Seek to the start of the image data
                                            fid.seek(row[5], 0)
                                            data = np.fromfile(fid, dtype=np.uint16, count=row[2]*row[3])

                                            # Reshape data and check if it fits in the imdata slice
                                            if data.size >= (xend - xstart) * (yend - ystart):
                                                data = data.reshape(row[3], row[2])
                                                imdata[ystart:yend,xstart:xend] = data[:yend - ystart,:xend - xstart]
                                            else:
                                                print(f"Data shape does not fit for row {row}")

                                # Crop data to the tile size, ensuring we do not exceed the bounds of imdata
                                xstart = max(xs - xmin, 0)
                                ystart = max(ys - ymin, 0)
                                xend = min(xstart + info['well_info']['xs'], imdata.shape[1])
                                yend = min(ystart + info['well_info']['ys'], imdata.shape[0])

                                imdata = imdata[ystart:yend,xstart:xend]
                                do_save(imdata, save_as, tile, channel, level, well, tiff_compression, verbose, info['well_info'])
                                do_disp(f'Ready Reading Tile: {tile} from Well: {well}',verbose)
                                return imdata
                        else:
                            do_disp(f'Error Reading Well, tile {tile} does not exist',verbose)
                            return None 
                else: # If tile is specified and a pyramid level is requested
                    do_disp(f'Error Reading Tiles {tile} for Well {well}, pyramid {level} for reading tiles is not supported',verbose)

            else: # If no data is found
                do_disp(f'Error Reading Well, pyramid level {level} does not exist',verbose)
                return None
        else:
            do_disp('Error well not Found!',verbose)
            return None
        

# Function to read well information
def read_well_info(image_data, channel_data):
    well_info = {
        'channels': len(channel_data),
        'lutname': [],
        'dye': [],
        'excitation': [],
        'emission': [],
        'tilex': image_data[5],
        'tiley': image_data[6],
        'tiles': image_data[5] * image_data[6],
        'bits': image_data[4],
        'resunit': 'µm',
        'xs': image_data[0],
        'ys': image_data[0],
        'xres': image_data[3],
        'yres': image_data[3],
        'objective': image_data[2]
    }
    for i in range(well_info['channels']):
        well_info['emission'].append(channel_data[i][0])
        well_info['excitation'].append(channel_data[i][1])
        if channel_data[i][1] == 0:
            well_info['dye'].append('TL')
            well_info['lutname'].append('white')
        else:
            well_info['dye'].append(channel_data[i][2].lower())
            well_info['lutname'].append(channel_data[i][4].split()[-1].lower())
    return well_info

def do_save(imdata, save_as, tile, channel, level, well, tiff_compression, verbose, well_info):
    """
    Save image data to TIFF or PNG formats, supporting metadata for TIFF files.

    Parameters:
        imdata (numpy.ndarray or list): Image data to save.
        save_as (str): Path and filename to save the image.
        tile (str or int): Tile information for naming.
        channel (int): Channel number.
        level (int): Image pyramid level.
        well (str): Well identifier.
        tiff_compression (str): Compression type for TIFF ('none', 'lzw', 'deflate').
        verbose (int): Verbosity level for logging.
        well_info (dict): Dictionary containing well metadata, including pixel size and LUT name.
    """
    if not save_as:
        return

    spath, sname = os.path.split(save_as)
    sname, ext = os.path.splitext(sname)

    # Construct the filename based on the provided parameters
    sname += f"_ch{channel}_{add_leading_zero(well)}"
    if not tile:
        sname += f"_level{level}" if level > 0 else ""
        tile = 'full'  # If tile is not specified, set it to full

    if ext not in ['.tif', '.png']:
        do_disp('Error, Only .tif or .png are supported!', verbose)
        return

    if not os.path.isdir(spath) and spath:
        do_disp('Error, Path not found, image not saved!', verbose)
        return

    spath += '/' if spath else ''

    valid_compression_types = {'none', 'lzw', 'deflate'}
    if tiff_compression.lower() not in valid_compression_types:
        do_disp("Error, Only 'none', 'lzw', and 'deflate' are supported TIFF compression values!", verbose)
        return

    # Extract calibration and LUT information from well_info
    pixel_width = well_info.get('xres', 1.0)
    pixel_height = well_info.get('yres', 1.0)
    magnification = well_info.get('objective', 'unknown')
    xres = 1e4 / pixel_width
    yres = 1e4 / pixel_height    
    lut_name = well_info.get('lutname', [])[channel - 1] if 'lutname' in well_info and channel - 1 < len(well_info['lutname']) else 'unknown'

    # Ensure image data is a list for consistency
    imdata = [imdata] if not isinstance(imdata, list) else imdata

    for im_index, img in enumerate(imdata, start=1):
        fname = f"{spath}{sname}_{add_leading_zero(str(im_index if len(imdata) > 1 else tile))}{ext}"
        if ext == '.tif':
            # Save as TIFF with manual metadata modification
            import tifffile
            from tifffile import TiffWriter

            with TiffWriter(fname, bigtiff=False) as tif:
                tif.write(
                    img,
                    compression=tiff_compression.lower(),
                    resolution=(xres, yres),                # sets XResolution, YResolution
                    resolutionunit='CENTIMETER',            # sets ResolutionUnit to 3
                    description=(
                        f"ImageJ=1.53c\n"
                        f"unit=micron\n"
                        f"spacing={pixel_height}\n"
                        f"pixel_width={pixel_width}\n"
                        f"pixel_height={pixel_height}\n"
                        f"LUT={lut_name}\n"
                        f"magnification={magnification}\n"
                    )
                )                
        elif ext == '.png':
            # Save as PNG
            Image.fromarray(img).save(fname)

        do_disp(f"Image saved as {fname}", verbose)

# Function to display text if verbose is enabled
def do_disp(text, verbose):
    if verbose:
        print(text)

# Function to add leading zero to a string containing a number
def add_leading_zero(input_string):
    if len(input_string) == 1:
        return '0' + input_string
    else:
        number_parts = [part for part in input_string.split() if part.isdigit()]
        if number_parts and int(number_parts[0]) < 10:
            return input_string.replace(number_parts[0], '0' + number_parts[0])
        return input_string
    
def remove_leading_zero(well_name: str) -> str:
    """
    Convert something like 'C02' into 'C2'.
    Assumes the first char is a letter, and the rest are digits.
    """
    if not well_name:
        return well_name  # or raise an error

    letter_part = well_name[0]       # e.g. 'C'
    digit_part = well_name[1:]       # e.g. '02'

    # Convert the digits to int, then back to string => removes leading zeros
    try:
        digit_part_noleading = str(int(digit_part))  # '02' -> int(2) -> '2'
    except ValueError:
        # If digit_part isn't purely digits, handle gracefully
        return well_name  # or raise an error

    return letter_part + digit_part_noleading

# Convert .NET ticks to datetime
def convert_dotnet_ticks_to_datetime(net_ticks):
    TICKS_AT_EPOCH = 621355968000000000  # .NET ticks at Unix epoch (1970-01-01T00:00:00Z)
    TICKS_PER_SECOND = 10000000  # .NET ticks per second
    ticks_since_epoch = net_ticks - TICKS_AT_EPOCH
    seconds_since_epoch = ticks_since_epoch // TICKS_PER_SECOND
    microseconds_remainder = (ticks_since_epoch % TICKS_PER_SECOND) // 10  # convert from 100-nanoseconds to microseconds
    utc_datetime = datetime(1970, 1, 1) + timedelta(seconds=seconds_since_epoch, microseconds=microseconds_remainder)
    return utc_datetime

def display_well_matrix(info, experiment_file):
    """
    Displays a matrix of wells used for each timepoint with well names.

    Parameters:
        info (dict): Metadata dictionary containing well and timepoint information.
        experiment_file (str): Path to the experiment SQLite database.
    """
    with sqlite3.connect(experiment_file) as conn:
        cur = conn.cursor()

        # Fetch all TimeSeriesElementId values
        cur.execute("SELECT DISTINCT TimeSeriesElementId FROM SourceImageBase")
        time_series_ids = [row[0] for row in cur.fetchall()]

        # Fetch well data
        cur.execute("SELECT Name, ZoneIndex FROM Well WHERE HasImages = 1")
        wells = cur.fetchall()

        well_names = [add_leading_zero(well[0]) for well in wells]
        well_matrix = []

        for timepoint in time_series_ids:
            cur.execute("""
                SELECT DISTINCT Well.Name FROM SourceImageBase
                JOIN Well ON SourceImageBase.ZoneIndex = Well.ZoneIndex
                WHERE TimeSeriesElementId = ?
            """, (timepoint,))
            wells_at_timepoint = [add_leading_zero(row[0]) for row in cur.fetchall()]

            row = [well if well in wells_at_timepoint else '' for well in well_names]
            well_matrix.append(row)

        print("Timepoint x Well Matrix:")
        for idx, row in enumerate(well_matrix):
            print(f"Timepoint {time_series_ids[idx]}: {row}")

def pretty_print_info(info):
    """
    Pretty print the metadata information.

    Parameters:
        info (dict): Metadata dictionary containing experiment details.
    """
    print("\nExperiment Metadata:")
    print("-------------------")
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")