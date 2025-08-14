# optisimplify.py
import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv

# ------------------ LOAD ENV VARIABLES ------------------
# Reads variables from .env file into environment
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY not found. Create a .env file with your API key.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# ------------------ SYSTEM PROMPT ------------------
SYSTEM_PROMPT = (
    "You are an expert in optical system design, including laser optics and imaging systems. "
    "Given user specs, suggest lens configurations, materials, simulate beam or ray paths, and estimate costs. "
    "Avoid weapon-related designs. Always explain your reasoning clearly."
)

# ------------------ ASSISTANT QUERY ------------------
def query_assistant(user_input: str) -> str:
    """Send user prompt to GPT model and return its reply."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # Change to "gpt-4o" or "gpt-4.1" for higher quality
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error contacting assistant: {e}"

# ------------------ GAUSSIAN BEAM UTILITY ------------------
def gaussian_beam_radius(z, w0, wavelength):
    """
    Calculate Gaussian beam radius w(z).
    z, w0, wavelength in meters.
    """
    zr = np.pi * w0**2 / wavelength
    return w0 * np.sqrt(1 + (z / zr) ** 2)

def refractive_index(material, wavelength_nm):
    # Simplified Sellmeier model or hardcoded values
    glass_data = {
        "BK7": {550: 1.5168, 450: 1.5220, 650: 1.5143},
        "F2":  {550: 1.6200, 450: 1.6320, 650: 1.6150},
    }
    return glass_data.get(material, {}).get(wavelength_nm, 1.0)

def snell_refraction(n1, n2, incident_angle):
    # Apply Snell's law: n1 * sin(theta1) = n2 * sin(theta2)
    sin_theta2 = n1 / n2 * np.sin(incident_angle)
    if abs(sin_theta2) > 1:
        return None  # Total internal reflection
    return np.arcsin(sin_theta2)

lens_system = [
    {"type": "surface", "radius": 10.0, "material": "BK7", "thickness": 2.0, "diameter": 6.0},
    {"type": "surface", "radius": -10.0, "material": "F2", "thickness": 1.5, "diameter": 6.0},
    {"type": "image_plane"}
]

def trace_ray(y0, angle0, lens_system, wavelength_nm):
    z = 0
    y = y0
    angle = angle0
    path = [(z, y)]
    n_current = 1.0  # Air

    for element in lens_system:
        if element["type"] == "image_plane":
            z += 5
            y += np.tan(angle) * 5
            path.append((z, y))
            break

        radius = element["radius"]
        thickness = element["thickness"]
        material = element["material"]
        diameter = element["diameter"]
        n_next = refractive_index(material, wavelength_nm)

        # Compute surface center
        z_center = z + radius

        # Intersect ray with spherical surface
        hit = intersect_sphere((z, y), angle, abs(radius), z_center)
        if hit is None or abs(hit[1]) > diameter / 2:
            break  # Missed lens

        path.append(hit)

        # Compute surface normal
        normal = surface_normal(hit, abs(radius), z_center)

        # Ray direction
        ray_dir = (np.cos(angle), np.sin(angle))

        # Refract
        new_dir = snell_with_normal(n_current, n_next, ray_dir, normal)
        if new_dir is None:
            break  # TIR

        angle = np.arctan2(new_dir[1], new_dir[0])
        z = hit[0]
        y = hit[1]
        n_current = n_next

        # Propagate to next surface
        z += thickness
        y += np.tan(angle) * thickness
        path.append((z, y))

    return path

def plot_rays():
    fig, ax = plt.subplots()
    for y0 in np.linspace(-1, 1, 5):  # 5 rays
        ray = trace_ray(y0, 0.0, lens_system, 550)
        z_vals, y_vals = zip(*ray)
        ax.plot(z_vals, y_vals)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Ray Paths Through Lens System")
    ax.grid(True)
    st.pyplot(fig)
    
def intersect_sphere(ray_origin, ray_angle, radius, z_center):
    # Ray: y = y0 + tan(angle) * (z - z0)
    # Sphere: (z - zc)^2 + y^2 = R^2
    y0 = ray_origin[1]
    z0 = ray_origin[0]
    m = np.tan(ray_angle)

    # Solve quadratic: Az^2 + Bz + C = 0
    A = 1 + m**2
    B = 2 * (m * y0 - z0 - m * z_center)
    C = (z0 - z_center)**2 + y0**2 - radius**2

    discriminant = B**2 - 4 * A * C
    if discriminant < 0:
        return None  # No intersection

    z_hit = (-B - np.sqrt(discriminant)) / (2 * A)
    y_hit = y0 + m * (z_hit - z0)
    return (z_hit, y_hit)

def surface_normal(hit_point, radius, z_center):
    z, y = hit_point
    dz = z - z_center
    dy = y
    norm = np.sqrt(dz**2 + dy**2)
    return (dz / norm, dy / norm)

def snell_with_normal(n1, n2, ray_dir, normal):
    # ray_dir and normal are (dz, dy)
    dot = ray_dir[0]*normal[0] + ray_dir[1]*normal[1]
    cos_theta1 = abs(dot)
    sin_theta1 = np.sqrt(1 - cos_theta1**2)
    sin_theta2 = n1 / n2 * sin_theta1
    if sin_theta2 > 1:
        return None  # TIR

    cos_theta2 = np.sqrt(1 - sin_theta2**2)
    # Refracted direction (approximate)
    return (normal[0]*cos_theta2 + ray_dir[0]*sin_theta2,
            normal[1]*cos_theta2 + ray_dir[1]*sin_theta2)



# ------------------ COMPONENTS EXAMPLE ------------------
COMPONENTS = [
    {"name": "Lens A", "focal_length_mm": 50, "diameter_mm": 25, "vendor": "Thorlabs"},
    {"name": "Mirror B", "reflectivity_pct": 99.9, "diameter_mm": 25, "vendor": "Edmund Optics"},
]

IMAGING_COMPONENTS = [
    {"name": "Achromatic Doublet", "focal_length_mm": 5.0, "diameter_mm": 6.0, "vendor": "Edmund Optics"},
    {"name": "Aspheric Lens", "focal_length_mm": 4.5, "diameter_mm": 5.0, "vendor": "Thorlabs"},
    {"name": "Triplet Apochromat", "focal_length_mm": 6.0, "diameter_mm": 6.0, "vendor": "Custom"},
]

# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="OptiSimplify – Laser Optics Assistant", page_icon="🔬")
st.title("🔬 OptiSimplify – Laser Optics Assistant")

with st.expander("Gaussian Beam Calculator", expanded=True):
    col1, col2, col3 = st.columns(3)
    w0_mm = col1.number_input("Beam waist w0 (mm)", value=1.0, min_value=0.01, step=0.1)
    wl_nm = col2.number_input("Wavelength (nm)", value=532.0, min_value=200.0, step=1.0)
    z_max_cm = col3.number_input("Propagation range (cm)", value=100.0, min_value=1.0, step=1.0)

    # Convert to meters
    w0 = w0_mm * 1e-3
    wavelength = wl_nm * 1e-9
    z = np.linspace(0, z_max_cm * 1e-2, 500)

    w = gaussian_beam_radius(z, w0, wavelength)

    fig, ax = plt.subplots()
    ax.plot(z * 100, w * 1000)  # cm on x-axis, mm on y-axis
    ax.set_xlabel("z (cm)")
    ax.set_ylabel("Beam radius w(z) (mm)")
    ax.set_title("Gaussian Beam Radius vs. Propagation Distance")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

st.subheader("Ask the optics assistant")
user_input = st.text_area("Describe your optics design challenge:")

if st.button("Ask Assistant"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        response = query_assistant(user_input.strip())
        st.markdown(response)

with st.expander("Example Component Catalog"):
    st.table(COMPONENTS)
    
st.subheader("📋 Define Your Lens System")

num_elements = st.number_input("Number of lens surfaces", min_value=1, max_value=5, value=2)

lens_elements = []
for i in range(num_elements):
    st.markdown(f"**Lens Surface {i+1}**")
    radius = st.number_input(f"Radius of curvature R{i+1} (mm)", value=10.0, key=f"r{i}")
    thickness = st.number_input(f"Thickness after surface T{i+1} (mm)", value=2.0, key=f"t{i}")
    diameter = st.number_input(f"Diameter D{i+1} (mm)", value=6.0, key=f"d{i}")
    material = st.selectbox(f"Material M{i+1}", ["BK7", "F2"], key=f"m{i}")
    lens_elements.append({
        "type": "surface",
        "radius": radius,
        "thickness": thickness,
        "diameter": diameter,
        "material": material
    })

lens_elements.append({"type": "image_plane"})

wavelength_nm = st.slider("Wavelength (nm)", min_value=400, max_value=700, value=550, step=10)

st.subheader("🌈 Multi-Wavelength Ray Tracing")
wavelengths = st.multiselect(
    "Select wavelengths (nm) to trace",
    options=[450, 500, 550, 600, 650],
    default=[450, 550, 650]
)

colors = {450: "blue", 500: "cyan", 550: "green", 600: "orange", 650: "red"}

if st.button("🌈 Trace Multi-Wavelength Rays"):
    fig, ax = plt.subplots()

    colors = {450: "blue", 500: "cyan", 550: "green", 600: "orange", 650: "red"}

    for wl in wavelengths:
        for y0 in np.linspace(-1.5, 1.5, 5):  # 5 rays
            ray = trace_ray(y0, 0.0, lens_elements, wl)
            if ray:
                z_vals, y_vals = zip(*ray)
                ax.plot(z_vals, y_vals, color=colors.get(wl, "gray"), label=f"{wl} nm")

    ax.set_xlabel("z (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Ray Paths Across Wavelengths")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    st.pyplot(fig)
    

if st.checkbox("Show spot diagram at image plane"):
    fig2, ax2 = plt.subplots()
    for wl in wavelengths:
        spots = []
        for y0 in np.linspace(-1.5, 1.5, 5):
            ray = trace_ray(y0, 0.0, lens_elements, wl)
            if ray:
                z, y = ray[-1]
                spots.append(y)
        ax2.scatter([wl]*len(spots), spots, color=colors.get(wl, "gray"), label=f"{wl} nm")

    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Ray height at image plane (mm)")
    ax2.set_title("Chromatic Shift at Image Plane")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
