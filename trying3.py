import streamlit as st

def calculate_bmi(weight, height):
    bmi = weight / (height ** 2)
    return bmi

def main():
    st.title("BMI Calculator")
    st.write("Masukkan berat dan tinggi badan untuk menghitung BMI Anda.")
    
    weight = st.number_input("Berat badan (kg)", min_value=1.0, format="%.2f")
    height = st.number_input("Tinggi badan (m)", min_value=0.5, format="%.2f")
    
    if st.button("Hitung BMI"):
        if height > 0:
            bmi = calculate_bmi(weight, height)
            st.write(f"BMI Anda adalah: {bmi:.2f}")
            
            if bmi < 18.5:
                st.warning("Anda berada dalam kategori underweight (kurus).")
            elif 18.5 <= bmi < 24.9:
                st.success("Anda berada dalam kategori normal.")
            elif 25 <= bmi < 29.9:
                st.warning("Anda berada dalam kategori overweight (kelebihan berat badan).")
            else:
                st.error("Anda berada dalam kategori obesitas.")
        else:
            st.error("Tinggi badan harus lebih dari 0.")

if __name__ == "__main__":
    main()