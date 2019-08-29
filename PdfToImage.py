from pdf2image import convert_from_path

'''
This method is used to concert PDF to JPEG and return location of the file
'''


def convert_to_image(input_path, output_path):
    pages = convert_from_path(input_path, dpi= 300,output_folder =output_path, fmt= "jpeg")
    return pages[0].filename
