import streamlit as st
print("theoretical loaded.")
def text():
    st.write('The system described by the model is a typical FM/HM interface. In our specific case, a Hall cross with a thin ferromagnetic layer displaying an out of plane magnetization (fig. 1).  ')
    st.image("https://journals.aps.org/prb/article/10.1103/PhysRevB.89.144425/figures/1/medium",
            caption = "*Fig. 1* Hall bar structure. Adapted from Phys. Rev. B 89, 144425 (2014)",
            width   = 400 )
    #($\eta_\text{DL}$ and $\eta_\text{FL}$)
    st.write(r'The LLG equation employed in the model is in explicit form and takes the Slonczewsky spin-orbit-torque coefficients as input. It goes as follows:')
    st.latex(r''' \frac{\partial \vec{m}}{\partial t} = -
    \frac{\gamma}{1+\alpha^2} (\vec{m} \times \vec{H}_{\text{eff}}) - 
    \frac{\gamma \alpha}{1+\alpha^2} \:\vec{m} \times (\vec{m} \times \vec{H}_{\text{eff}})''')
    st.write(r'Where $m$ represents the mgnetization unit vector, $\alpha$ the Gilbert damping constant, $\gamma$ the gyromagnetic ratio, and $\vec{H}_{\text{eff}}$ is the effective magnetic field. The effective magnetic field contains contributions of the applied external field, the effective anisotropy field, and the current induced fields via spin orbit torque effects. It reads as follows:')
    st.latex(r''' \vec{ H }_{\text{eff}} =
    \vec{ H }_{\text{ext}} + \vec{ H }_{\text{k}_\text{eff}} - ( 
    \vec{ H }^{\text{SOT}}_{\text{FL}} + 
    \vec{ H }^{\text{SOT}}_{\text{DL}} ) \\ \:\\ \:\\
    \vec{ H }_{\text{k}} = \frac{2\vec{K}_1}{Js}  \\ \:\\
    \vec{ H }^{\text{SOT}}_{\text{FL}} = \eta_\text{FL} \frac{  j_e \hbar  }{ 2 e t \mu_0 M_s }\:\vec{m} \times (\vec{m} \times \vec{p}) \\ \:\\
    \vec{ H }^{\text{SOT}}_{\text{DL}} = \eta_\text{DL} \frac{  j_e \hbar  }{ 2 e t \mu_0 M_s }\:(\vec{m} \times \vec{p})
    ''')


    st.write(r"The $\vec{p}$ vector represents the spin polarization of electrons. For a current flowing along the x direction, the vector is $(0,-1,0)$. As the here simulated system presents out of plane magnetization along the +z axis, the $\vec{K}_1$ anisotropy constant is represented by $(0,0,K_1)$")
    st.write("Therefore, this simplified model just describes out-of-plane systems with negligible Planar Hall Effect, compared to the Anomalous Hall Effect. It will get improved soon.")

    st.caption("Performing the integration")

    st.write("In order to accurately compute the first and second harmonic components of the Anomalous Hall Voltage, the period is, at least, split in 1000 equidistand time steps. This will ensure an accurate description of the time variation of the voltage induced by the AC current. Additionaly, it will improve the computation of the numerical Fourier integrals for getting the harmonic responses.")
    st.write("Under AC, the voltage is made up by the following harmonics:")
    st.latex(r''' V_{xy}(t) = V^{xy}_0 + V^{xy}_\omega\sin(\omega t) + V^{xy}_{2\omega}\cos(2\omega t) + ...''')
    st.write("Those harmonic components can be computed by applying a sin-cos fit over one full period of relaxed magnetization (in AC current context, equilibrium means the system undergoes just in oscillations due to the current).")

    st.write(r"As the system starts fully pointing in the z direction, it is important to simulate the electric current with a cosine wave $J_x=j_e \sin(\omega t)$. ")
