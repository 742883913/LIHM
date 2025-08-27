import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image
from skimage.transform import downscale_local_mean

from Sampling_simulation import decimation_factor


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

def IPR(W, H, wavelength, distance, pixelSize,  numPixels, Measured_amplitude, k_max):
    """
    :param W: axis 1
    :param H: axis 2
    :param distance: the propagation distance
    :param pixelSize: the pixel size
    :param numPixels: the number of pixels on W and H
    :param Measured_amplitude: The amplitude of sensor's recorded field
    :param k_max: The maximum iteration number
    :return: The reconstructed sample field
    """
    update_phase = [] # Store every iteration's phase
    last_field = None # Store the reconstructed sample field
    for k in range(k_max):
        # a) On the sensor plane
        if k == 0: # The first iteration
            phase0 = np.zeros(Measured_amplitude.shape) # Give the guess phase equals to 0
            field1 = Measured_amplitude * np.exp(1j * phase0) # The guess field
        else:
            field1 = Measured_amplitude * np.exp(1j * update_phase[k - 1]) # The field on the sensor plane for the coming computation
        # b) Backpropagate the field and apply constraint
        field2 = angular_spectrum_method(W, H, -distance, wavelength, field1, pixelSize, numPixels) # Field after backpropagation
        phase_field2 = np.angle(field2)  # phase of field2
        amp_field2 = np.abs(field2)  # amplitude of field2
        abso = -np.log(amp_field2 + 1e-8) # Calculate the absorption index;1e-8 to prevent 0 value
        # Apply constraints(None-zero constraint)
        mask = abso < 0 # Mask: when the absorption index is less than 0
        phase_field2[mask] = 0 # Set the corresponding phase to 0
        abso[mask] = 0 # Set the absorption as 0
        amp_field2 = np.exp(-abso)
        field22 = amp_field2 * np.exp(1j * phase_field2) # Field after applying the constraint
        # c) Forward propagate the field
        field3 = angular_spectrum_method(W, H, distance, wavelength, field22, pixelSize, numPixels) # Forward propagtion
        phase_field3 = np.angle(field3) # Phase of field3
        update_phase.append(phase_field3) # Input the phase to this array
        # d) Backpropagate to get the image
        field4 = angular_spectrum_method(W, H, -distance, wavelength, field3, pixelSize, numPixels) # Backpropagation
        last_field = field4
    return last_field

object = load_and_normalize_image("/Users/wangmusi/Documents/GitHub/LIHM/pic/stringline_padded.png") # Read the sample
plot_image(object,"object") # Plot the object
pitch_size = [0.4e-6, 1.6e-6] # 0.4e-6 is the hologram's pixel spacing and 1.6e-6 is the sensor's pitch size
numPixels_hologram = 840 # number of pixels on each axis of the sensor
z2 = 1e-3 # the sample-sensor distance; unit: meter
wavelength = 525e-9
# Define the fine grid(hologram grid)
x = np.arange(numPixels_hologram) - numPixels_hologram / 2 - 1
y = np.arange(numPixels_hologram) - numPixels_hologram / 2 - 1
W, H = np.meshgrid(x, y)
# Define the sample property
am = np.exp(-1.6 * object)
ph0 = 3
ph = ph0 * object
field_after_object = am * np.exp(1j * ph)
am_field_after_object = np.abs(field_after_object)
plot_image(am_field_after_object, "Sample field") # Plot the field after sample
# Acquire the hologram
hologram_field = angular_spectrum_method(W, H, z2,wavelength, field_after_object, pitch_size[0], numPixels_hologram)
hologram_amplitude = np.abs(hologram_field)
hologram_intensity = np.abs(hologram_field) ** 2
plot_image(hologram_amplitude,  "Hologram field")

# Sample the hologram
FX = W / (pitch_size[0] * numPixels_hologram)  # Frequency coordination
FY = H / (pitch_size[0] * numPixels_hologram)
Delta = pitch_size[1] # The pixel region size
H_filter = np.sinc(FX * Delta) * np.sinc(FY * Delta)
A = fftshift(fft2(ifftshift(hologram_intensity)))
s = np.real(fftshift(ifft2(ifftshift(A * H_filter)))) # In case of the imaginary value
decimation_factor = int(pitch_size[1] / pitch_size[0])
offset = decimation_factor // 2
sampled_field_intensity = s[offset::decimation_factor, offset::decimation_factor] # The output of the center pixel is the average value of the small rectangular area of the pixel
am_sampled_field = np.sqrt(sampled_field_intensity)
plot_image(am_sampled_field, "Sampled hologram")


# Create the sensor grid
numPixels_sensor = am_sampled_field.shape[0]
x_sen = np.arange(numPixels_sensor) - numPixels_sensor / 2 - 1
y_sen = np.arange(numPixels_sensor) - numPixels_sensor / 2 - 1
W_sen, H_sen = np.meshgrid(x_sen, y_sen)

# IPR reconstruction
rec_field = IPR(W_sen,H_sen,wavelength,z2,pitch_size[1],numPixels_sensor,am_sampled_field,50)
am_rec = np.abs(rec_field)
plot_image(am_rec,"rec")