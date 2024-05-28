import sqlite3
import numpy as np
from PIL import Image
import os
from datetime import datetime, timedelta

# Main function to read and process data
def crxreader(experiment_file, **kwargs):

    # Set default values
    verbose = kwargs.get('verbose', 0)
    channel = kwargs.get('channel', 1)
    well = kwargs.get('well', None)
    tile = kwargs.get('tile', None)
    level = kwargs.get('level', 0)
    save_as = kwargs.get('save_as', None)
    tiff_compression = kwargs.get('tiff_compression', 'deflate')
    info = kwargs.get('info', None)

    # Check if the file exists
    is_ok = False
    if os.path.isfile(experiment_file):
        if not info:
            info = {}
            info['experiment_file'] = experiment_file
            filepath = os.path.dirname(experiment_file)
            info['images_file'] = os.path.join(filepath, 'images-0.db')
            do_disp('Reading Info from CellReporterXpress experiment.db file', verbose)
            try:
                # Connect to the SQLite database
                with sqlite3.connect(experiment_file) as conn:
                    cur = conn.cursor()
                    # Fetch experiment base data
                    cur.execute('SELECT DateCreated, Creator, Name FROM ExperimentBase')
                    rs = cur.fetchone()
                    # Fetch acquisition experiment name
                    cur.execute('SELECT Name FROM AcquisitionExp')
                    rse = cur.fetchone()
                    # Convert date and time
                    dt = convert_dotnet_ticks_to_datetime(int(rs[0]))
                    do_disp(f"Name: {rs[2]}", verbose)
                    do_disp(f"Creator: {rs[1]}", verbose)
                    do_disp(f"Protocol: {rse[0]}", verbose)
                    do_disp(f"Date: {dt.strftime('%Y-%m-%d %H:%M UTC')}", verbose)
                    # Store information in the info dictionary
                    info.update({
                        'name': rs[2],
                        'creator': rs[1],
                        'protocol': rse[0],
                        'dt': dt,
                    })
                    # Fetch well information
                    cur.execute('SELECT Name, ZoneIndex FROM well WHERE HasImages = 1')
                    wells = cur.fetchall()
                    # Fetch additional info required to read well info
                    cur.execute('SELECT SensorSizeYPixels, SensorSizeXPixels, Objective, PixelSizeUm, SensorBitness, SitesX, SitesY FROM AcquisitionExp, AutomaticZonesParametersExp')
                    rsdI = cur.fetchone()
                    cur.execute('SELECT Emission, Excitation, Dye, channelNumber, ColorName FROM ImagechannelExp')
                    rsdC = cur.fetchall()
                    # Process well information
                    info['well_info'] = read_well_info(rsdI, rsdC)
                    info['numwells'] = len(wells)
                    info['wells'] = {well[0]: well[1] for well in wells}
                is_ok = True
                do_disp('Ready Reading Info!', verbose)
            except sqlite3.Error as e:
                do_disp('Error Reading Info from file!', verbose)
                do_disp('Format of data in file is different than expected.', verbose)
        else:
            is_ok = True
            do_disp('Using given Info!', verbose)
    else:
        do_disp('Error File not Found!', verbose)

    # If only information is requested
    if not well and not tile and is_ok:
        return info

    # If Well and/or image data is requested
    if well and is_ok:
        # Check if well is in the info and proceed if everything is okay
        if well in info['wells']:
            # get well ZoneData
            zi = info['wells'][well]
            # Connect to the SQLite database
            conn = sqlite3.connect(info['experiment_file'])
            cur = conn.cursor()
            cur.execute('''
                SELECT CoordX, CoordY, SizeX, SizeY, BitsPerPixel, ImageIndex, channelId 
                FROM SourceImageBase 
                WHERE ZoneIndex = ? AND level = ? 
                ORDER BY CoordX ASC, CoordY ASC
                ''', (zi, level))
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
                    do_save(imdata, save_as, tile, channel, level, well, tiff_compression, verbose)
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
                            do_save(outdata, save_as, tile, channel, level, well, tiff_compression, verbose)
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
                            xse = xs + info['well_info']['xs'] - max(row[2] for row in zd) - 1
                            yse = ys + info['well_info']['ys'] - max(row[3] for row in zd) - 1

                            # Find the bounding coordinates for the tile
                            zdx = sorted(set(row[0] for row in zd))
                            zdy = sorted(set(row[1] for row in zd))
                            xmin = max(x for x in zdx if x <= xs)
                            xmax = min(x for x in zdx if x >= xse)
                            ymin = max(y for y in zdy if y <= ys)
                            ymax = min(y for y in zdy if y >= yse)

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
                                do_save(imdata, save_as, tile, channel, level, well, tiff_compression, verbose)
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
        'objective': image_data[3]
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

# Save image data
def do_save(imdata, save_as, tile, channel, level, well, tiff_compression, verbose):
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
        do_disp('Error, Only .tif or .png are supported!',verbose) # Extension not supported
        return

    if not os.path.isdir(spath) and spath:
        do_disp('Error, Path not found, image not saved!',verbose) # Path not found
        return

    spath += '/' if spath else ''
    valid_compression_types = {'none', 'lzw', 'deflate'}
    if tiff_compression.lower() not in valid_compression_types:
        do_disp("Error, Only 'none', 'lzw', and 'deflate' are supported .tif compression values!") # Compression type not supported
        return

    # Save the image(s)
    imdata = [imdata] if not isinstance(imdata, list) else imdata
    for im_index, img in enumerate(imdata, start=1):
        fname = f"{spath}{sname}_{add_leading_zero(str(im_index if len(imdata) > 1 else tile))}{ext}"
        Image.fromarray(img).save(fname, compression=tiff_compression.lower())
    do_disp('Image Saved!', verbose)



# Convert .NET ticks to datetime
def convert_dotnet_ticks_to_datetime(net_ticks):
    TICKS_AT_EPOCH = 621355968000000000  # .NET ticks at Unix epoch (1970-01-01T00:00:00Z)
    TICKS_PER_SECOND = 10000000  # .NET ticks per second
    ticks_since_epoch = net_ticks - TICKS_AT_EPOCH
    seconds_since_epoch = ticks_since_epoch // TICKS_PER_SECOND
    microseconds_remainder = (ticks_since_epoch % TICKS_PER_SECOND) // 10  # convert from 100-nanoseconds to microseconds
    utc_datetime = datetime(1970, 1, 1) + timedelta(seconds=seconds_since_epoch, microseconds=microseconds_remainder)
    return utc_datetime
