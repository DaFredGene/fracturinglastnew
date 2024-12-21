import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import time
# import obspy
import math
from obspy import read
import base64

# import pyvista as pv
# from pyvista import Plotter

# read file function
def read_data(file):
    stream = read(file, format="SEGY")
    data = np.array([trace.data for trace in stream])
    return stream, data

def visualize_2d(data):
    st.subheader("2D Visualization")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(data.T, aspect="auto", cmap="seismic", origin="lower")
    ax.set_title("Seismic Data (2D Visualization)")
    ax.set_xlabel("Trace Number")
    ax.set_ylabel("Sample Number")
    st.pyplot(fig)

def apply_fft(data):

    fft_results = np.fft.fft(data, axis=1)
    freq = np.fft.fftfreq(data.shape[1], d=0.004)  # Assuming a 4 ms sampling interval
    return freq, fft_results

def visualize_fft(freq, fft_results):
    st.subheader("FFT Visualization")
    fig, ax = plt.subplots(figsize=(10, 5))
    cax = ax.imshow(
        np.abs(fft_results).T, 
        aspect="auto", 
        cmap="viridis", 
        origin="lower",
        extent=[0, fft_results.shape[0], freq[0], freq[len(freq)//2]]
    )
    cbar = fig.colorbar(cax, ax=ax, orientation="vertical")
    cbar.set_label("Intensity")
    ax.set_title("Frequency Domain (FFT)")
    ax.set_xlabel("Trace Number")
    ax.set_ylabel("Frequency (Hz)")
    st.pyplot(fig)

# def visualize_3d(data):
#     st.subheader("3D Visualization")
#     grid = pv.UniformGrid()
#     grid.dimensions = (*data.shape, 1)  
#     grid.spacing = (1, 1, 1)
#     grid.point_data["Amplitude"] = data.ravel(order="F")
#     plotter = Plotter(notebook=False)
#     plotter.add_volume(grid, opacity="sigmoid", cmap="viridis")
#     plotter.show(jupyter_backend="static")
#     st.write("3D visualization displayed in PyVista.")

def plot_single_trace(data, trace_index, sampling_rate):
    """
    Plot seismic signal from a single trace.
    """
    st.subheader(f"Trace {trace_index + 1}")
    time = np.arange(len(data[trace_index])) / sampling_rate  
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, data[trace_index], label=f"Trace {trace_index + 1}", color="blue")
    ax.set_title("Seismic Signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    st.pyplot(fig)


# Center the entire section and ensure the image is displayed correctly
st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">
        <img src="data:image/png;base64,{}" width="200"/>
        <h1>Lab Instrumentasi Hulu Migas</h1>
        <h2>Microseismic Survey for Unconventional Reservoir</h2>
    </div>
""".format(base64.b64encode(open("LogoUI.png", "rb").read()).decode()), unsafe_allow_html=True)



with st.sidebar:
    conversion_factor = 9.869233e-13
    st.write("Convert Darcy to SI unit")
    darcy = st.number_input("Darcy input (D)", min_value=0.0)
    si_value = darcy * conversion_factor
    st.write(f"{si_value} m²")
    
tab_1, tab_2, tab_3, tab_4, tab_5 = st.tabs(["Fracture Length", "Pressure based on fracture length and permeability", "Fracture Toughness & Crack Length", "Net & Fracture Pressure Calculation", "Read Seismic Data"])

with tab_1:
    col1, col2 = st.columns(2)
    # input for fracturing parameter
    with col1:
        with st.container():
            qi = st.number_input(label='Injection rate (m³/s)', min_value=0.0)
            tp = st.number_input(label='Pumping Time (s)', min_value=0.0)
            Cl = st.number_input(label='Fluid-loss coefficient (m/s½)', min_value=0.0)
            hl = st.number_input(label='Permeable or fluid-loss height (m)', min_value=0.0)
            sp = st.number_input(label='Spurt loss (m³/m²)', min_value=0.0)
            w = st.number_input(label='Fracture Width (m)', min_value=0.0)
            hf = st.number_input(label='Fracture Height (m)', min_value=0.0)

    with col2:
        with st.container():
            st.write("""### <-Input the Fracturing Parameter""")
            
            if st.button("Result", key="1",use_container_width=True):
                
                st.image("frac.jpg")
                numerator = qi * tp
                denominator = float(6*Cl*hl*tp**(1/2)) + (4*hl*sp) + (2*w*hf)
                    
                if denominator == 0:  
                    st.warning("Input the number correctly", icon="⚠️")
                else:
                    length = numerator / denominator
                    st.markdown(f"<div style='text-align: center;'>Fracture Length: {round(length, 3)} m</div>", unsafe_allow_html=True)

                    with st.container():
                        st.latex(r"""L \approx \frac{q_i t_p}{6C_L h_L \sqrt{t_p} + 4 h_L S_p + 2 \bar{w} h_f}""")
                    
                        st.markdown(f"<div style='text-align: center;'><span style='font-size: 1.5rem;'>Calculated Fracture Length: </span><span style='font-size: 2rem;'>{length:.3f} m</span></div>",unsafe_allow_html=True)

with tab_2:
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            e = st.number_input(label="Young's modulus (Pa)", min_value=0.0)      
            q = st.number_input(label="Constant flow rate (m³/s)", min_value=0.0)      
            mu = st.number_input(label="Viscosity (Pa.s)", min_value=0.0)      
            lf = st.number_input(label="Fracturing Length (m)", min_value=0.0)      
            h = st.number_input(label="Fracture height (m)", min_value=0.0)  
            Ptip = st.number_input(label="Fracture tip pressure (Pa)", min_value=0.0)  

    with col2: 
        with st.container():
            st.write("""### <-Input the Net Pressure""")
            
            if st.button("Result", use_container_width=True):
                if h != 0:
                    P_net = (((e**(3/4))/h) * (mu * q * lf)**(1/4)) + Ptip 
                    st.markdown(f"<div style='text-align: center;'>Pressure Net: {round(P_net, 3)} Pa</div>", unsafe_allow_html=True)
                    st.latex(r"""P_{net} = \frac{E^{3/4}}{h} \left( \mu \times Q \times L \right)^{1/4} + P_{tip}""")
                    st.markdown(f"<div style='text-align: center;'><span style='font-size: 1.5rem;'>Calculated Pressure Net: </span><span style='font-size: 2rem;'>{P_net:.3f} Pa</span></div>",unsafe_allow_html=True)
                elif h == 0:
                    st.warning("The fracture height must be non-zero", icon="⚠️")
                

with tab_3:
    st.write("""### Fracture Toughness and Crack Length Calculator""")
    with st.container():
        sigma_t = st.number_input(label="Enter Rock Tensile Stress (σt) [Pa]", value=2.0e6, key="sigma_t_input", min_value=0.0)
        a = st.number_input(label="Enter Initial Crack Length (a) [m]", value=0.05, key="a_input", min_value=0.0)
        Pf = st.number_input("Enter Fluid Pressure (Pf) [Pa]", value=5.0e6, min_value=0.0)
        calculate = st.button("Calculate KIC and Crack Length")
        if Pf == 0 :
            st.warning("The pressure fluid must be non-zero", icon="⚠️")
        else:
            if calculate:
                K_ic = sigma_t * math.sqrt(math.pi * a)
                st.write(f"Fracture Toughness (KIC): {K_ic:.2f} Pa√m")

                x_f = (K_ic**2) / (math.pi * Pf**2)
                st.write(f"Crack Length (Half-Length): {x_f:.2f} m")


with tab_4:
    st.write("""### Net Pressure Calculation for Hydraulic Fracturing""")
    with st.container():
        
        # Input for each parameter
        mu = st.number_input("Enter Fluid Viscosity (μ) [Pa·s]", value=1.0, min_value=0.0)
        q = st.number_input("Enter Fluid Flow Rate (q) [m³/s]", value=1.0, min_value=0.0)
        E = st.number_input("Enter Young's Modulus (E) [Pa]", value=1.0e10, min_value=0.0)
        hf = st.number_input("Enter Fracture Width (hf) [m]", value=0.01, min_value=0.0)
        L = st.number_input("Enter Fracture Length (L) [m]", value=100.0, min_value=0.0)
        sigma_closure = st.number_input("Enter Fracture Closure Stress (σ_closure) [Pa]", value=5.0e6, min_value=0.0)

        if st.button("Calculate"):

            if hf != 0 or L != 0:

                # Formula to calculate P_net
                P_net = ((16 * mu * q * E**3) / (3.1416 * hf**4 * L))**(1/4)

                # Formula to calculate P_frac (P_frac = P_net + sigma_closure)
                P_frac = P_net + sigma_closure
                # Display the result for P_net and P_frac
                st.write(f"**P_net** (Net Pressure) is: {P_net:.2f} Pa")
                st.write(f"**P_frac** (Fracturing Pressure) is: {P_frac:.2f} Pa")
            else:
                if hf == 0:
                    st.warning("The fracture height must be non-zero", icon="⚠️")
                if L == 0:
                    st.warning("The fracture length must be non-zero", icon="⚠️")


with tab_5:
    st.write("""### Read Seismic Data""")
    with st.container():
        st.write("Upload your file to read seismic data 2D or 3D")
        uploaded_file = st.file_uploader("Upload Seismic Data", type=["segy", "sgy"])

        if uploaded_file:
            st.write("The file has been uploaded successfully")
            try:
                # Read SEGY data
                signal_seismic, seismic_data = read_data(uploaded_file)
                st.write(f"This process may take some time")

                st.pyplot(signal_seismic.plot())

                # Display visualization
                visualize_2d(seismic_data)

                freq, fft_results = apply_fft(seismic_data)
                visualize_fft(freq, fft_results)

            except Exception as e:
                st.error(f"An error occurred while reading the file: {e}")