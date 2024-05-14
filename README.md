# crxReader

## Overview
`crxReader.py` is a Python script designed to read and process data from a CellReporterXpress experiment file. It extracts information from an SQLite database, processes image data, and optionally saves the images with specified parameters.

## Requirements
- Python 3.x
- SQLite3
- NumPy
- Pillow (PIL)
- OS and datetime libraries

## Installation
Ensure you have the required libraries installed. You can install the necessary libraries using pip:
```sh
pip install numpy pillow
```

## Usage
The main function in this script is `crxreader`, which reads and processes data from the experiment file.

### Function Signature
```python
def crxreader(experiment_file, **kwargs)
```

### Parameters
- `experiment_file` (str): Path to the experiment file.
- `verbose` (int, optional): Verbosity level. Default is 0.
- `channel` (int, optional): Channel number to process. Default is 1.
- `well` (str, optional): Specific well to process. Default is None.
- `tile` (str, optional): Specific tile to process. Default is None.
- `level` (int, optional): Level of the image to process. Default is 0.
- `save_as` (str, optional): Path to save the processed image. Default is None.
- `tiff_compression` (str, optional): Compression type for TIFF images. Default is 'deflate'.
- `info` (dict, optional): Dictionary to store experiment information. Default is None.

### Examples

#### Example 1: Basic Information Extraction
Extracts information from the experiment file and prints it.
```python
from crxReader import crxreader

info = crxreader('testdata1/experiment.db')
```

#### Example 2: Processing a Specific Well and Tile
Processes a specific well and tile, and saves the image as `test.tif`.
```python
from crxReader import crxreader

info = crxreader('testdata1/experiment.db')
im = crxreader('testdata1/experiment.db', well='B2', tile=1, save_as='test.tif', info=info)
```

#### Example 3: Processing with Different Parameters
Processes a different experiment file with specified well and level, and saves the image as `test.tif`.
```python
from crxReader import crxreader

im = crxreader('testdata2/experiment.db', well='A1', level=3, save_as='test.tif')
```

## Error Handling
The script includes error handling for:
- File existence
- Supported image formats (only `.tif` and `.png` are supported)
- Supported TIFF compression types (`none`, `lzw`, `deflate`)
- Validity of specified paths for saving images

## Additional Information
For more details on the parameters and usage, refer to the docstrings within the script.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes.

## Contact
For questions or issues, please contact the repository owner.

```

Feel free to further customize this `README.md` as needed to fit your specific requirements.
