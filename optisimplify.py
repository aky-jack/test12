system_prompt = """
You are an expert laser optics assistant. Given user specs, suggest components, simulate beam paths, and estimate costs.
Avoid weapon-related designs. Always explain your reasoning.
"""

import openai
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI

openai.api_key = "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"
client = OpenAI(api_key="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")
def query_assistant(user_input):

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
)

return response.choices[0].message.content


def gaussian_beam(z, w0, λ):
    zr = np.pi * w0**2 / λ
    wz = w0 * np.sqrt(1 + (z/zr)**2)
    return wz

z = np.linspace(0, 100, 500)
wz = gaussian_beam(z, w0=1e-3, λ=532e-9)

fig, ax = plt.subplots()
ax.plot(z, wz)
st.pyplot(fig)


# Example component metadata
components = [
    {"name": "Lens A", "focal_length": 50, "diameter": 25, "vendor": "Thorlabs"},
    {"name": "Mirror B", "reflectivity": 99.9, "diameter": 25, "vendor": "Edmund Optics"},
]

st.title("OptiSimplify – Laser Optics Assistant")

user_input = st.text_input("Describe your optics design challenge:")
if st.button("Ask Assistant"):
    response = query_assistant(user_input)
    st.markdown(response)


