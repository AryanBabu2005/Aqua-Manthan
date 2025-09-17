import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_plotly
import folium
from folium import IFrame

# --- 1. LOAD THE DATA FROM YOUR EXCEL FILE ---
file_path = r"C:\Users\babua\Downloads\Ground Water Level_Delhi new delhi chanakyapuri india gate pz (2).xlsx"
try:
    station_df = pd.read_excel(file_path, sheet_name='Ground Water Level - Ground Wat', skiprows=6)
except FileNotFoundError:
    print(f"❌ ERROR: The file was not found at the path: {file_path}")
    exit()

# --- 2. CLEAN AND PREPARE THE DATA ---
print("--- Data Cleaning and Preparation ---")
station_df.columns = station_df.columns.str.strip()
station_df = station_df[['Data Time', 'Data Value']]
station_df = station_df.rename(columns={'Data Time': 'ds', 'Data Value': 'y'})
station_df['ds'] = pd.to_datetime(station_df['ds'])
station_df['y'] = station_df['y'].astype(float) * -1
station_df = station_df.drop_duplicates(subset='ds').set_index('ds')
station_df = station_df.resample('D').mean()
station_df['y'] = station_df['y'].interpolate(method='time')
print("✅ Data cleaning complete.")

# --- 3. TRAIN AI MODEL AND CREATE FORECAST ---
print("\n--- AI Forecasting ---")
prophet_df = station_df.reset_index()
model = Prophet(yearly_seasonality=True, daily_seasonality=False)
model.add_country_holidays(country_name='IN')
model.fit(prophet_df)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
print("✅ Forecast for the next year complete.")

# --- 4. MODEL PERFORMANCE EVALUATION ---
print("\n--- Calculating Model Accuracy (Error Metrics) ---")
df_cv = cross_validation(model, initial='500 days', period='90 days', horizon='180 days', parallel="processes")
df_p = performance_metrics(df_cv)
print("✅ Cross-validation complete.")

# --- 5. CREATE INTERACTIVE VISUALIZATIONS AS HTML ---
print("\n--- Generating HTML for Visualizations ---")

# --- Create the Interactive Graph ---
fig = plot_plotly(model, forecast)
fig.update_layout(
    title='AI-Powered Groundwater Level Forecast',
    xaxis_title='Date',
    yaxis_title='Water Level (meters below ground)'
)
# Convert the plot to an HTML string
graph_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

# --- Create the Performance Table ---
performance_table_df = df_p[['horizon', 'mae', 'rmse', 'mape']].head().copy()
# (Formatting code for the table)
performance_table_df['mae'] = performance_table_df['mae'].round(4)
performance_table_df['rmse'] = performance_table_df['rmse'].round(4)
performance_table_df['mape'] = (performance_table_df['mape'] * 100).round(2).astype(str) + '%'
performance_table_df = performance_table_df.rename(columns={
    'horizon': 'Forecast Horizon', 'mae': 'Avg. Error (MAE)',
    'rmse': 'RMSE', 'mape': 'Avg. % Error (MAPE)'
})
table_html = performance_table_df.to_html(index=False, justify='center', classes='table table-striped')

# --- Combine visuals into a single HTML block for the popup ---
popup_html = f"""
<html>
<head>
    <title>Station Report</title>
    <style> 
        body {{ font-family: sans-serif; }} 
        table {{ border-collapse: collapse; margin: 15px 0; font-size: 0.9em; }} 
        thead tr {{ background-color: #009879; color: #ffffff; text-align: left; }} 
        th, td {{ padding: 10px 12px; }} 
    </style>
</head>
<body>
<h3>Model Performance</h3>
    {table_html}
    {graph_html}
</body>
</html>
"""

# --- 6. GENERATE THE FINAL INTERACTIVE MAP ---
print("\n--- Generating Final Interactive Map ---")

# Station metadata
station_lat, station_lon, station_name = 28.6125, 77.225, "India Gate, New Delhi"
map_center = [station_lat, station_lon]

# Create the base map
final_map = folium.Map(location=map_center, zoom_start=12)

# Create an IFrame with the combined HTML content
iframe = IFrame(popup_html, width=700, height=550)
popup = folium.Popup(iframe, max_width=700)

# Add the final marker to the map
folium.Marker(
    location=[station_lat, station_lon],
    popup=popup,
    tooltip=f"<strong>{station_name}</strong><br>Click to see full report"
).add_to(final_map)

# Save the final map
final_report_filename = "Jal_Drishti_Interactive_Report.html"
final_map.save(final_report_filename)

print(f"\n✅✅✅ Success! Your final interactive report has been saved as '{final_report_filename}'.")
print("Open this file in your browser. Click the marker to see all the data.")