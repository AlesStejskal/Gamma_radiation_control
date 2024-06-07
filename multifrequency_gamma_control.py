### README ###
################################################################################
# Multitone_gamma_conrol.py is the script for numerical calculation of the gamma
# radiation intensity behind the vibrating Moessbauer absorber whose motion profile
# is defined by Fourier series.
#
# The code is optimized for speed. For that reason several preparatory calculations
# appear in the code.

# Cut out the first 340 nanoseconds of the numerical simulation which contain
# numerical artefacts
#
# The script was run by Python version 3.7.4
# with the libraries versions:
# numpy            1.21.6
# matplotlib       3.5.3
################################################################################

### Libraries ###
import numpy as np
import matplotlib.pyplot as plt


### Simulation parameteres ###
minus_t = 0.5            # [us] lower limit of the time integration over t0 - do not change if necessary
time_range = 2.000       # [us] simulation time range - upper integration limit
num_of_points = 1000     # [-] number of points in time domain

vibration_freq = 10      # [MHz] fundamental vibration frequency
E_tuned = 10             # [MHz] Doppler energy modulation

p = 2.4                  # [-] vibration amplitude expressed by dimensionless parameter "p"
harm_num = 1             # [-] number of harmonics in the Fourier series of motion profile
motion_shift = 0.0       # [us] time shift of absorber motion
motion_invertion = "yes" # inversion of motion profile "yes" = inverted, others = original

d_eff = 19.0             # [-] effective thickness
absorption_shift = 0     # [MHz] shift of the absorption function relative to the source
bhyf = 0.0               # [T] hyperfine magnetic field


source_broadening = 1.20    # emission line broadening in multiple of natural linewidth
absorber_broadening = 1.00  # absorption line broadening in multiple of natural linewidth

### Fourier series parameteres - amplitudes and phases ###
### trapezoidal fast step [50 harmonics]
F_amps = [1.567, 0.697, 0.4, 0.267, 0.208, 0.182, 0.167, 0.151, 0.132, 0.115,
0.101, 0.092, 0.087, 0.083, 0.078, 0.073, 0.067, 0.062, 0.059, 0.057, 0.055,
0.053, 0.05, 0.048, 0.046, 0.044, 0.043, 0.042, 0.041, 0.039, 0.037, 0.036,
0.035, 0.034, 0.034, 0.033, 0.031, 0.03, 0.03, 0.029, 0.029,0.028, 0.027,
 0.026, 0.026, 0.025, 0.025, 0.024, 0.024, 0.023]
F_phases = [0.002, 0.006, 0.017, 0.03, 0.038, 0.044, 0.042, 0.046, 0.053,
 0.061, 0.079, 0.087, 0.08, 0.084, 0.09, 0.096, 0.104, 0.112, 0.118,
 0.122, 0.127, 0.131, 0.139, 0.148, 0.154, 0.161, 0.165, 0.169, 0.173,
 0.182, 0.192, 0.197, 0.203, 0.209, 0.209, 0.215, 0.229, 0.237, 0.237,
 0.245, 0.245, 0.254, 0.263, 0.273, 0.273, 0.284, 0.284, 0.295, 0.295, 0.308]

###square  [50 harmonics]
# F_amps = [2.292, 0.014, 0.764, 0.014, 0.458, 0.014, 0.327, 0.014, 0.254,
#  0.014, 0.208, 0.014, 0.176, 0.014, 0.152, 0.014, 0.134, 0.014, 0.12,
#  0.014, 0.109, 0.014, 0.099, 0.014, 0.091, 0.014, 0.084, 0.014, 0.078,
#  0.014, 0.073, 0.014, 0.068, 0.014, 0.064, 0.014, 0.061, 0.014, 0.058,
#  0.014, 0.055, 0.014, 0.052, 0.014, 0.05, 0.014, 0.047, 0.014, 0.045, 0.014]
# F_phases = [0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0,
#  0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0,
#  0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0]

###bipolar pulse [60 harmonics]
# F_amps = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
# 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
# 1,1,1,1,1,1,1,1,1]
# F_phases = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
# 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
# 0,0,0,0,0,0,0,0,0,0]

# dualtone
# F_amps = [1.9,0.9]
# F_phases = [0,0]
################################################################################

################################################################################
### function definitions ###
def generate_motion (t,harm_num,F_amps,F_phases,w,time_delay):
    val=0
    t = np.subtract(t,time_delay) # shift of the time axis for generation of shifted motion profile
    for i in range(1,harm_num+1):
        val += F_amps[i-1]*np.sin(w*i*t-F_phases[i-1]) # sum over all harmonics
    return val

def normalize (array):
    norm_constant = np.max(np.abs(array)) # normalize array by its maximum absolute value
    return np.multiply(array,1/norm_constant)
    # return np.multiply(array,1/norm_constant)
# generation of the to the absorber incident gamma photon field
def generate_photon_field (t,A,j_w_s_plus_gmmma_over_2,p_over_w_s,z_motion):
    return A*np.heaviside(t,1)*np.exp(-(j_w_s_plus_gmmma_over_2)*(t-p_over_w_s*z_motion))
# calculation of the absorption lines positions induced by mangetic splitting

def magnetic_splitting (bhyf,absorption_shift):
    gg = 0.18125 # gyromagnetic ration of ground state
    ge = -0.10348 # gyromagnetic ratio of excited state
    un = 7.6226077 # nuclear magneton in [MHz]

    un_bhyf = un*bhyf/2 # preparatory calculation
    # calculation of individual lines
    E1 = (-gg+3*ge)*un_bhyf+absorption_shift
    E2 = (-gg+ge)*un_bhyf+absorption_shift
    E3 = (-gg-ge)*un_bhyf+absorption_shift
    E4 = (gg+ge)*un_bhyf+absorption_shift
    E5 = (gg-ge)*un_bhyf+absorption_shift
    E6 = (gg-3*ge)*un_bhyf+absorption_shift
    lines_positions = [E1,E2,E3,E4,E5,E6] # assembling in list

    lines_amps =[3,2,1,1,2,3] # sextet lines amplitudes
    normalized_lines_amps = np.divide(lines_amps,sum(lines_amps)) # normalization of amplitudes

    return lines_positions, normalized_lines_amps

def absorption_function (f,bhyf,absorption_shift,gamma_0):
    lines_positions,lines_amplitudes = magnetic_splitting(bhyf,absorption_shift) # get the lines positions and amplitudes by magnetic splittin function

    # conversion into the angular frequencies: w = 2*pi*f
    lines_positions_w = np.multiply (lines_positions,2*np.pi)
    absorption_shift_w = np.multiply(absorption_shift,2*np.pi)
    gamma_0_w = np.multiply(gamma_0,2*np.pi)
    w = np.multiply(f,2*np.pi)


    j_gamma_0w_2 = j*gamma_0_w/2 # preparatory calculation

    val = 0
    for i in range(len(lines_positions)):
        val+= lines_amplitudes[i]*(j_gamma_0w_2)/(w-lines_positions_w[i]+j_gamma_0w_2)#  sum of complex conjugated Lorentzian, complex conjugation is necessary due to numpy FFT algorithm definition   complex conjugate in energy domain causes the reversal in the time domain

    return val
################################################################################

################################################################################
### elementary constants ###
gamma_0 = 1.126830 # [1/us] natural linewidth in [MHz] (0.097 mm/s * 11.61685)
E0 = 14413                      # resonant transition energy level [eV]
q = 1.602176*10**(-19)          # elementary charge [C]
c = 299.792458                  # speed of light in vacuum [m/us]
h = 6.62607015*10**(-34)        # Planck constant [J/s]
j = complex(0,1)                # complex unit

### preparatory calculations ###
dt = time_range/num_of_points       # [us] time axis step
gamma_source = gamma_0*source_broadening # [1/us] source linewidth
gamma_absorber = gamma_0*absorber_broadening # [1/us] absorber linewidth

dt = time_range/num_of_points            # [us] time axis step
# [rad/us] maximum angular frequency of the wave for simulation given by two times
# lower frequency than the maximum frequency range ((2*Pi)/dt)

### conversion of parameters to angular frequencies
w_0 = np.pi/(dt)
w_s = w_0 + 2*np.pi*E_tuned
vibration_w = 2*np.pi*vibration_freq
gamma_source_w = gamma_source*(2*np.pi)
gamma_absorber_w = gamma_absorber*(2*np.pi)
absorption_shift_w = absorption_shift*(2*np.pi)

w_E0 = 10**(-6)*2*np.pi*E0*q/h   # [rad/us] angular frequency of the 14.413 keV transition
wavelength = 10**(12)*(2*np.pi*c/w_E0)   # [pm] gamma photon wavelength
Amp = p*c/(w_E0)                 # [m] amplitude of vibrations

### time intensity t0 integration range
num = int((minus_t/(time_range/num_of_points)))+num_of_points # num of t0 points
t0_range = np.linspace(-minus_t,time_range,num)
################################################################################

################################################################################
### Numerical calculation start ###

### generation of the time and energy domain axis ###
time_axis = np.linspace(0,((num_of_points-1)*dt),num=(num_of_points)) # time axis generation
frequency_axis = np.fft.fftfreq(num_of_points, d=dt) # frequency axis generation

# frequency axis shift for FFT and multiplying to avoid the complex conjugate in energy domain
frequency_axis = np.multiply(np.fft.fftshift(frequency_axis),-1)
w_axis = np.multiply(frequency_axis,2*np.pi)# angular frequency axis generation

### iniciation of the gamma radiation intensity array ###
wave_intensity = np.zeros(num_of_points)

# print the vibration amplitude, wavelength and energy resoluton
print("Vibration amplitude: "+str(round(Amp*10**(12),1))+" pm")
print("Wavelength: "+str(round(wavelength,1))+" pm")
max = max(frequency_axis)
min = min(frequency_axis)
print("Energy resolution [MHz] "+str((max-min)/(len(frequency_axis)-1)))

################################################################################
### generation of movement waveform
absorber_motion = generate_motion(time_axis,harm_num,F_amps,F_phases,vibration_w,motion_shift)
abs_motion_norm = normalize(absorber_motion)
if motion_invertion == "yes":
    # inversion of the absorber motion profile
    abs_motion_norm = np.multiply(abs_motion_norm,-1)

### generation of absorption function of absorber
absorption = absorption_function(w_axis,bhyf,absorption_shift_w,gamma_absorber_w)

### Calculate exponential values for absorption_function array
exponential_absorption = np.exp(-(d_eff/2)*absorption)

### preparatory calculations for generation of the incident photon field
prep1 = j*w_s+gamma_source_w/2
prep2=p/w_s
wave_amp = np.sqrt(gamma_source_w)

counter = 0 ### counter for watching the
################################################################################
### Numerical integration over t0
for t0 in t0_range:

    time_axis_t0 = np.subtract(time_axis,t0) # preparatory calculation (time axis shift)

    ### generation of electric intensity wave
    incident_wave = generate_photon_field(time_axis_t0,wave_amp,prep1,prep2,abs_motion_norm)

    ##FFT transforamtion of the electric intensity from the time domain to energy domain
    incident_field_energy_domain = np.fft.fft(incident_wave)

    ### Absorption process
    energy_domain_after_absorption = incident_field_energy_domain * exponential_absorption

    ### Conversion of the energy domain by inverse FFT back to time domain ###
    wave_after_absorption = np.fft.ifft(energy_domain_after_absorption)

    ### Elctric field intensity calculation ###
    wave_intensity += np.abs(wave_after_absorption)**2 * dt

    counter+=1
    if counter%500 == 0:
        print(str(round((counter/len(t0_range))*100,0))+" % done",end='\r')

print("Calculation done")
### End of integration
################################################################################

################################################################################
### Display the absorber motion profile and gamma radiation intensity ###
fig = plt.figure(figsize=(8,8))
plt.subplots_adjust(left=0.10,
                    bottom=0.10,
                    right=0.70,
                    top=0.90,
                    wspace=0.0,
                    hspace=0.30)
plt1 = fig.add_subplot(2,1,1)
plt1.set_title("Absorber motion",fontsize=12)
plt1.set_xlabel("Time [ns]",fontsize=12)
plt1.set_ylabel("Amplitude [pm]",fontsize=12)
plt1.plot(time_axis*1000,np.multiply(abs_motion_norm,Amp*(10**(12))),'.-',markersize=2,
linewidth=0.5,label='z amplitude')
plt1.set_xlim(((motion_shift))*1000,((time_range)*1000))
plt1.text(1.05,0.65,
" p ='%02.1f'\n $\Delta$E [MHz] = '%02.1f'\n f [MHz] = '%02.1f\n Harm. [-] = '%02.0i'"
%(p,E_tuned,vibration_freq,harm_num),horizontalalignment='left',
verticalalignment='center', transform=plt1.transAxes)

plt2 = fig.add_subplot(2,1,2)
plt2.set_title("Gamma radiation intensity behind absorber",fontsize=12)
plt2.set_xlabel("Time [ns]",fontsize=12)
plt2.set_ylabel("Normalized intensity [-]",fontsize=12)
plt2.plot(time_axis*1000,wave_intensity,'.-',markersize=2,linewidth=0.5,label='z modul')
plt2.legend()
plt2.axhline(y=0.0, color='g', linestyle='--')
plt2.axhline(y=1, color='g', linestyle='--')
plt2.set_xlim(0,(time_range)*1000)
plt.show()
################################################################################
###End of the script
