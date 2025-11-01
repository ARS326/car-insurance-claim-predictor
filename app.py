import streamlit as st
import pandas as pd
import joblib


def data_manipulation(df):
  df.drop(columns=['policy_id'],inplace=True)
  df.dropna(inplace=True)
  # Converting all pieces of data in some columns to numrical. This way, if some data is numerical but obj or str in datatype, it will be converted to int or float as necessary
  df['policy_tenure'] = pd.to_numeric(df['policy_tenure'], errors='coerce')
  df['age_of_car'] = pd.to_numeric(df['age_of_car'], errors='coerce')
  df['age_of_policyholder'] = pd.to_numeric(df['age_of_policyholder'], errors='coerce')
  df['max_torque'] = df['max_torque'].str.extract(r'(\d+\.?\d*)').astype(float)
  df['max_power'] = df['max_power'].str.extract(r'(\d+\.?\d*)').astype(float)
  df['displacement'] = pd.to_numeric(df['displacement'], errors='coerce')
  # Converting all yes/no columns into binary (0/1)
  boolean_columns = ['is_esc', 'is_adjustable_steering', 'is_tpms', 'is_parking_sensors', 'is_parking_camera', 'is_front_fog_lights',
                   'is_rear_window_wiper', 'is_rear_window_washer', 'is_rear_window_defogger', 'is_brake_assist', 'is_power_door_locks',
                   'is_central_locking', 'is_power_steering', 'is_driver_seat_height_adjustable', 'is_day_night_rear_view_mirror', 'is_ecw', 'is_speed_alert']
  for col in boolean_columns:
    df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)
  return df

# Load the pipeline (make sure it's in the same directory)
pipeline = joblib.load("pipeline.pkl")

st.set_page_config(page_title="Car Insurance Claim Predictor", layout="wide")

st.title("🚗 Car Insurance Claim Predictor")
st.markdown("Fill in the details below to check if a claim is likely to occur.")

# --- User Input Section ---
with st.form("user_input_form"):
    st.subheader("Enter Policy & Vehicle Details")

    policy_id = st.text_input("Policy ID",'ID00000')
    policy_tenure = (st.number_input("Policy Tenure (in months)", min_value=0.0, max_value=120.0, step=1.0))/100
    age_of_car = (st.number_input("Age of Car (in years)", min_value=0.0, max_value=20.0, step=0.1))/100
    age_of_policyholder = (st.number_input("Age of Policyholder (in years)", min_value=18, max_value=100, step=1))/100

    area_cluster = st.selectbox("Area Cluster", ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22'])
    population_density = st.number_input("Population Density", min_value=0, step=1)
    make = st.selectbox("Make", ['1','2','3','4','5'])
    segment = st.selectbox("Segment",['A', 'C1', 'C2', 'B2', 'B1', 'Utility'])

    model = st.selectbox("Model Name", ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11'])
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])

    max_torque = st.text_input("Max Torque", "60Nm@3500rpm")
    max_power = st.text_input("Max Power", "40.36bhp@6000rpm")
    engine_type = st.selectbox("Engine Type", ['F8D Petrol Engine', '1.2 L K12N Dualjet', '1.0 SCe',
       '1.5 L U2 CRDi', '1.5 Turbocharged Revotorq', 'K Series Dual jet',
       '1.2 L K Series Engine', 'K10C', 'i-DTEC', 'G12B',
       '1.5 Turbocharged Revotron'])

    airbags = st.number_input("Airbags", min_value=0, max_value=10)
    is_esc = st.selectbox("ESC (Electronic Stability Control)", ["Yes", "No"])
    is_adjustable_steering = st.selectbox("Adjustable Steering", ["Yes", "No"])
    is_tpms = st.selectbox("Tyre Pressure Monitoring System", ["Yes", "No"])
    is_parking_sensors = st.selectbox("Parking Sensors", ["Yes", "No"])
    is_parking_camera = st.selectbox("Parking Camera", ["Yes", "No"])

    rear_brakes_type = st.selectbox("Rear Brakes Type", ["Disc", "Drum"])
    displacement = st.number_input("Displacement (cc)", min_value=600, max_value=5000, step=1)
    cylinder = st.number_input("Cylinder Count", min_value=1, max_value=12, step=1)
    transmission_type = st.selectbox("Transmission Type", ["Manual", "Automatic"])
    gear_box = st.number_input("Gear Box", min_value=1, max_value=10)
    steering_type = st.selectbox("Steering Type", ["Power", "Manual", "Electric"])
    turning_radius = st.number_input("Turning Radius (m)", min_value=1.0, max_value=10.0, step=0.1)
    length = st.number_input("Car Length (mm)", min_value=2000, max_value=6000)
    width = st.number_input("Car Width (mm)", min_value=1000, max_value=3000)
    height = st.number_input("Car Height (mm)", min_value=1000, max_value=3000)
    gross_weight = st.number_input("Gross Weight (kg)", min_value=500, max_value=5000)

    is_front_fog_lights = st.selectbox("Front Fog Lights", ["Yes", "No"])
    is_rear_window_wiper = st.selectbox("Rear Window Wiper", ["Yes", "No"])
    is_rear_window_washer = st.selectbox("Rear Window Washer", ["Yes", "No"])
    is_rear_window_defogger = st.selectbox("Rear Window Defogger", ["Yes", "No"])
    is_brake_assist = st.selectbox("Brake Assist", ["Yes", "No"])
    is_power_door_locks = st.selectbox("Power Door Locks", ["Yes", "No"])
    is_central_locking = st.selectbox("Central Locking", ["Yes", "No"])
    is_power_steering = st.selectbox("Power Steering", ["Yes", "No"])
    is_driver_seat_height_adjustable = st.selectbox("Driver Seat Height", ["Yes", "No"])
    is_day_night_rear_view_mirror = st.selectbox("Day Night Rear View Mirror", ["Yes", "No"])
    is_ecw = st.selectbox("Ecw", ["Yes", "No"])
    is_speed_alert = st.selectbox("Speed Alert", ["Yes", "No"])

    ncap_rating = st.selectbox("NCAP Rating", [0, 1, 2, 3, 4, 5])

    submitted = st.form_submit_button("🔍 Predict")

# --- Process the input and display output ---
if submitted:
    try:
        # Create a DataFrame
        input_data = pd.DataFrame([{
            'policy_id': policy_id,
            'policy_tenure': policy_tenure,
            'age_of_car': age_of_car,
            'age_of_policyholder': age_of_policyholder,
            'area_cluster': area_cluster,
            'population_density': population_density,
            'make': make,
            'segment': segment,
            'model': model,
            'fuel_type': fuel_type,
            'max_torque': max_torque,
            'max_power': max_power,
            'engine_type': engine_type,
            'airbags': airbags,
            'is_esc': is_esc,
            'is_adjustable_steering': is_adjustable_steering,
            'is_tpms': is_tpms,
            'is_parking_sensors': is_parking_sensors,
            'is_parking_camera': is_parking_camera,
            'rear_brakes_type': rear_brakes_type,
            'displacement': displacement,
            'cylinder': cylinder,
            'transmission_type': transmission_type,
            'gear_box': gear_box,
            'steering_type': steering_type,
            'turning_radius': turning_radius,
            'length': length,
            'width': width,
            'height': height,
            'gross_weight': gross_weight,

            'is_front_fog_lights':is_front_fog_lights,
            'is_rear_window_wiper':is_rear_window_wiper,
            'is_rear_window_washer':is_rear_window_washer,
            'is_rear_window_defogger':is_rear_window_defogger,
            'is_brake_assist':is_brake_assist,
            'is_power_door_locks':is_power_door_locks,
            'is_central_locking':is_central_locking,
            'is_power_steering':is_power_steering,
            'is_driver_seat_height_adjustable':is_driver_seat_height_adjustable,
            'is_day_night_rear_view_mirror':is_day_night_rear_view_mirror,
            'is_ecw':is_ecw,
            'is_speed_alert':is_speed_alert,

            'ncap_rating': ncap_rating
        }])


        # Predict
        prediction = pipeline.predict(input_data)[0]
        label = "✅ Claim Not Likely" if prediction == 0 else "⚠️ Claim Likely"

        st.success(f"**Prediction Result:** {label}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
