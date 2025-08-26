import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image

def load_and_normalize_image(filepath):
    # Load the image
    image = Image.open(filepath).convert('L')  # Convert to grayscale
    # Convert image to a NumPy array
    grayscale_data = np.array(image, dtype=np.float32)
    # Normalize the grayscale data to [0, 1]
    normalized_data = (grayscale_data - grayscale_data.min()) / (grayscale_data.max() - grayscale_data.min())
    return normalized_data

def plot_image(field, title):
    """
    :param field: the field to be plotted
    :param title: the title of the plot
    """
    data = field.astype(np.float32)
    mn, mx = data.min(), data.max()
    eps = 1e-12
    data = (data - mn) / (mx - mn + eps)
    plt.figure(figsize=(6,6))
    im = plt.imshow(data, cmap='gray', vmin=0, vmax=1)
    plt.colorbar(im, label="Normalized value")
    plt.title(title); plt.axis('off'); plt.show()

def Transfer_function(W, H, distance, wavelength, pixelSize, numPixels):
    """
    :param W: axis 1
    :param H: axis 2
    :param distance: the propagation distance
    :param wavelength: the illumination's wavelength
    :param pixelSize: the pixel size
    :param numPixels: the number of pixels on W and H
    :return: the transfer function
    """
    FX = W / (pixelSize * numPixels) # Frequency coordination
    FY = H / (pixelSize * numPixels)
    k = 2 * np.pi / wavelength # Wave number
    arg = 1 - (wavelength ** 2) * (FX ** 2 + FY ** 2)
    sq = np.where(arg >= 0, np.sqrt(arg), 0.0)
    H = np.exp(1j * k * distance * sq)
    H[arg < 0] = 0
    return H

def angular_spectrum_method(W, H, distance, wavelength, field, pixelSize, numPixels):
    """
    :param W: axis 1
    :param H: axis 2
    :param distance: the propagation distance
    :param field: the field at the intial position
    :param pixelSize: the pixel size
    :param numPixels: the number of pixels on W and H
    :return: the field after propagation
    """
    GT = fftshift(fft2(ifftshift(field))) # Angular spectrum: Fourier transforming the initial field
    transfer = Transfer_function(W, H, distance, wavelength, pixelSize, numPixels) # Calculate the transfer function
    gt_prime = fftshift(ifft2(ifftshift(GT * transfer))) # Inverse Fourier transform on: transfer function * angular spectrum
    return gt_prime

object = load_and_normalize_image("/Users/wangmusi/Documents/GitHub/LIHM/pic/circle2.png") # Read the sample
plot_image(object,"object") # Plot the object
numPixels = 1024 # number of pixels on each axis of the sensor
pitch_size = 1.2e-6 # sensor's pitch distance; unit: meter
z2 = 2e-3 # the sample-sensor distance; unit: meter
wavelength = 525e-9
region_length = numPixels * pitch_size # Length of sensor
# Define the sensor grid
x = np.arange(numPixels) - numPixels / 2 - 1
y = np.arange(numPixels) - numPixels / 2 - 1
W, H = np.meshgrid(x, y)
# Define the sample property
am = np.exp(-1.6 * object)
ph0 = 3
ph = ph0 * object
field_after_object = am * np.exp(1j * ph)
am_field_after_object = np.abs(field_after_object)
plot_image(am_field_after_object, "Sample field") # Plot the field after sample
# Acquire the hologram
hologram_field = angular_spectrum_method(W, H, z2,wavelength, field_after_object, pitch_size, numPixels)
hologram_amplitude = np.abs(hologram_field)
plot_image(hologram_field,  "Hologram field")

image = angular_spectrum_method(W, H, -z2,wavelength, hologram_field, pitch_size, numPixels)
plot_image(image, "rec field")