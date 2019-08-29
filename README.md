# Compare PDF to Image
This project contains solution about comparing pdf with image. The idea of this project is to be used in printing houses 

## How to set up project
To set up this project you need to create virtual environment with following command:
```shell script
    virtualenv venv -p python3
```
Then you need to activate the virtual environment:
```shell script
    source venv/bin/activate
```
The last step of set up project is to install all required libraries:
```shell script
    pip install -r requirements.txt
```
If you are Linux user please execute those commands with sudo

### Required application that our PC need to have

1) **oppler-utils**
    Windows users will have to install poppler for Windows, then add the bin/ folder to PATH.
    Mac users will have to install poppler for Mac.
    Linux users will have both tools pre-installed with Ubuntu 16.04+ and Archlinux. If it's not, run `sudo apt install poppler-utils`

## Technologies
   - Python 3.6
   - OpenCV
   - PyQt5
   