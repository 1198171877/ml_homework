import cv2

def check_image_properties(image_path):
    """
    Checks the properties of an image including dimensions, channels, and size.

    Args:
        image_path (str): Path to the image file.

    Returns:
        None
    """
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to load the image. Check the file path.")
        return

    # Get the dimensions
    height, width, channels = image.shape

    # Get the file size
    file_size = round((image.nbytes / 1024), 2)  # Size in KB

    # Print image properties
    print(f"Image Path: {image_path}")
    print(f"Image Dimensions: {width}x{height}")
    print(f"Number of Channels: {channels}")
    print(f"File Size: {file_size} KB")

# Example usage
if __name__ == "__main__":
    image_path = "datasets/SOS/palsar/test/masks/10001.png"  # Replace with your image path
    check_image_properties(image_path)
