import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
from tqdm import tqdm

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

def update_support_absorption_simple(field_obj, sigma = 1, alpha = 0.3):
    """
    field_obj : Object-plane field. If complex, the amplitude |field_obj| is used.
    sigma : the Gaussian blur used to suppress noise and fuse fine ripples into coherent blobs.
    alpha :  The threshold is T = alpha * max(blurred_deviation). arger alpha -> tighter (smaller) support; smaller alpha -> looser support.
    S :  support mask; True marks pixels deemed to belong to the object.
    """
    # 1) Amplitude deviation from the flat-field background (≈1)
    amp = np.abs(field_obj)
    dev = np.abs(amp - 1.0)  # larger -> more likely object
    # 2) Gaussian smoothing to suppress pixel-level noise / fuse small ripples
    dev_blur = gaussian_filter(dev, sigma=sigma)
    # 3) Scale-free relative threshold; larger alpha => tighter support
    T = alpha * dev_blur.max()
    S = dev_blur > T # True = object region, False = background
    return S

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
    # This step is to prepare for updating the support area. The blank area obtained will replace the area outside the sample.
    plane_wave = np.ones_like(measured_amplitude, dtype=complex)
    blank_field = angular_spectrum_method(W,H,-distance,wavelength, plane_wave,pixelSize, num_pixel)
    S = np.ones_like(Measured_amplitude, dtype=bool)
    for k in range(k_max):
        # a) On the sensor plane
        if k == 0: # The first iteration
            phase0 = np.zeros(Measured_amplitude.shape) # Give the guess phase equals to 0
            field1 = Measured_amplitude * np.exp(1j * phase0) # The guess field
        else:
            field1 = Measured_amplitude * np.exp(1j * update_phase[k - 1]) # The field on the sensor plane for the coming computation
        # b) Backpropagate the field and apply constraint
        field2 = angular_spectrum_method(W, H, -distance, wavelength, field1, pixelSize, numPixels) # Field after backpropagation
        if k % 2 == 0 and k > 0:
            if k == 2:
                new_S = update_support_absorption_simple(field2, sigma=2, alpha=0.25)
                S = new_S
            else:
                new_S = update_support_absorption_simple(field2, sigma=2, alpha=0.25)
                S = S & new_S
        mask_bg = ~S
        if np.any(mask_bg):
            m = np.abs(field2)[mask_bg].mean() / np.abs(blank_field)[mask_bg].mean()
        else:
            m = 1.0
        field2_updated = field2.copy()
        field2_updated[~S] = blank_field[~S] * m # background field determine
        field2 = field2_updated # update the object field

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

def focus_metric(field_obj):
    # Convert complex field to amplitude (magnitude);
    amp = np.abs(field_obj)
    # Sobel gradient along x-axis
    gx = sobel(amp, axis=0)
    # Sobel gradient along y-axis (columns)
    gy = sobel(amp, axis=1)
    # Gradient magnitude image
    grad = np.hypot(gx, gy)
    # Use the variance of the gradient magnitude as the sharpness score
    return grad.var()

def autofocus(W,H,z_list,wavelength, field_sensor,pixel_size, num_pixel):
    """
    field_sensor :  Complex-valued hologram field at the sensor plane
    z_list : 1D iterable of candidate propagation distances in meters
    pixel_size : pixel pitch at the sensor plane in meters.
    W, H :
    numpixels : number of pixels on W and H
    """
    focus_vals = []
    # Sweep through all candidate propagation distances
    for z in tqdm(z_list):
        # Back-propagate the sensor-plane field by distance z (Angular Spectrum Method)
        field_obj = angular_spectrum_method(W,H,z,wavelength, field_sensor,pixel_size, num_pixel)
        # Compute a scalar sharpness/contrast score for the object-plane field at this z
        focus_vals.append(focus_metric(field_obj))
    # Convert to a NumPy array for convenient argmax and downstream use
    focus_vals = np.array(focus_vals)
    # Pick the z that yields the highest focus score
    idx = np.argmax(focus_vals)
    return z_list[idx], focus_vals

object_intensity = load_and_normalize_image(r"/Users/wangmusi/Desktop/exp_model/d2/7.5.png") # Read the image
measured_amplitude = np.sqrt(object_intensity)

# 系统参数
pitch_size = 5.86e-6
num_pixel = 600
z_list = np.linspace(3e-2, 2e-1, 400) # The range to estimate the sample-sensor distance
wavelength = 525e-9

# 构建坐标系
x = np.arange(num_pixel) - num_pixel / 2 - 1
y = np.arange(num_pixel) - num_pixel / 2 - 1
W, H = np.meshgrid(x, y)
z2, focus_vals = autofocus(W,H,z_list,wavelength, measured_amplitude, pitch_size, num_pixel)
print(f"最佳对焦距离：{z2:.3f} m")

# 执行重建算法
rec_field = IPR(W, H, wavelength, z2, pitch_size,  num_pixel, measured_amplitude, 50)
am_rec = np.abs(rec_field)
plot_image(am_rec, "rec")


