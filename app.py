import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# Seitenkonfiguration
st.set_page_config(
    page_title="CO2-Fußabdruck-Tracker für Reisen",
    page_icon="🌍",
    layout="wide"
)

# Titel und Beschreibung
st.title("CO2-Fußabdruck-Tracker für Reisen")
st.markdown("""
Diese App hilft dir, den CO2-Fußabdruck deiner Reisen zu berechnen und zu visualisieren.
Vergleiche verschiedene Transportmittel und finde die umweltfreundlichste Option für deine Reiseroute!
""")

# Seitenleiste für Eingaben
with st.sidebar:
    st.header("Reisedetails")
    
    # Startpunkt und Ziel eingeben
    start_point = st.text_input("Startpunkt (Stadt oder Adresse)", "Zürich, Schweiz")
    end_point = st.text_input("Ziel (Stadt oder Adresse)", "Berlin, Deutschland")
    
    # Transportmittel auswählen
    transport_options = ["Auto", "Flugzeug", "Zug", "Bus", "Motorrad", "Fahrrad"]
    selected_transports = st.multiselect(
        "Wähle Transportmittel zum Vergleich",
        transport_options,
        default=["Auto", "Flugzeug", "Zug"]
    )
    
    # Personen
    travelers = st.number_input("Anzahl der Reisenden", min_value=1, value=1)
    
    # Berechnung starten
    calculate_button = st.button("CO2-Fußabdruck berechnen")
    
    # Datenquellen angeben
    st.markdown("---")
    st.caption("""
    **Datenquellen:**
    - Distanzberechnung: OpenStreetMap via GeoPy
    - CO2-Faktoren: Durchschnittswerte basierend auf öffentlichen Umweltdatenbanken
    """)

# CO2-Emissionsfaktoren in kg CO2 pro Personenkilometer
emission_factors = {
    "Auto": 0.17,
    "Flugzeug": 0.24,
    "Zug": 0.04,
    "Bus": 0.07,
    "Motorrad": 0.11,
    "Fahrrad": 0.0
}

# Hauptfunktion zur Berechnung des CO2-Fußabdrucks
def calculate_co2_footprint(start, end, transports, num_travelers):
    # Geocoding von Start- und Endpunkt
    geolocator = Nominatim(user_agent="co2_footprint_tracker")
    
    try:
        start_location = geolocator.geocode(start)
        end_location = geolocator.geocode(end)
        
        if not start_location or not end_location:
            st.error("Konnte einen oder beide Standorte nicht finden. Bitte überprüfe deine Eingaben.")
            return None
        
        # Berechnung der Distanz
        start_coords = (start_location.latitude, start_location.longitude)
        end_coords = (end_location.latitude, end_location.longitude)
        
        # Direkte Distanz in km
        direct_distance = round(geodesic(start_coords, end_coords).kilometers, 2)
        
        # Realistischere Distanzen für verschiedene Transportmittel (Anpassungsfaktoren)
        distance_factors = {
            "Auto": 1.2,
            "Flugzeug": 1.0,
            "Zug": 1.3,
            "Bus": 1.3,
            "Motorrad": 1.2,
            "Fahrrad": 1.1
        }
        
        # Berechnung der CO2-Emissionen für die ausgewählten Transportmittel
        results = []
        for transport in transports:
            distance = direct_distance * distance_factors[transport]
            co2_per_km = emission_factors[transport]
            total_co2 = distance * co2_per_km
            co2_per_person = total_co2 / num_travelers
            
            results.append({
                "Transportmittel": transport,
                "Distanz (km)": round(distance, 2),
                "CO2 pro Person (kg)": round(co2_per_person, 2),
                "Gesamt CO2 (kg)": round(total_co2, 2)
            })
        
        return pd.DataFrame(results), direct_distance, start_coords, end_coords
    
    except Exception as e:
        st.error(f"Fehler bei der Berechnung: {e}")
        return None

# Hauptbereich - Ergebnisdarstellung
if calculate_button:
    with st.spinner("Berechne CO2-Fußabdruck..."):
        result = calculate_co2_footprint(start_point, end_point, selected_transports, travelers)
        
        if result:
            df, direct_distance, start_coords, end_coords = result
            
            # Ergebnisübersicht
            st.header(f"Reise von {start_point} nach {end_point}")
            st.write(f"Direkte Entfernung: {direct_distance} km")
            
            # Daten anzeigen
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # CO2-Vergleichsdiagramm
                fig = px.bar(
                    df,
                    x="Transportmittel",
                    y="CO2 pro Person (kg)",
                    color="Transportmittel",
                    title=f"CO2-Vergleich verschiedener Transportmittel für {travelers} Reisende",
                    labels={"CO2 pro Person (kg)": "CO2-Emissionen pro Person (kg)"}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Umweltauswirkungen visualisieren
                st.subheader("Umweltauswirkungen")
                for transport in df["Transportmittel"]:
                    transport_data = df[df["Transportmittel"] == transport].iloc[0]
                    co2 = transport_data["CO2 pro Person (kg)"]
                    
                    # Umweltvergleich erstellen
                    trees_needed = round(co2 / 21, 2)  # Ein Baum absorbiert ca. 21 kg CO2 pro Jahr
                    equivalent_days = round(co2 / 10, 1)  # Durchschnittliche CO2-Emissionen pro Person pro Tag (geschätzt)
                    
                    st.write(f"**{transport}:** {co2} kg CO2 pro Person")
                    st.write(f"Entspricht dem CO2, das {trees_needed} Bäume in einem Jahr aufnehmen")
                    st.write(f"Oder den durchschnittlichen CO2-Emissionen von {equivalent_days} Tagen")
                    st.write("---")
            
            with col2:
                # Detaillierte Ergebnistabelle
                st.subheader("Details")
                st.dataframe(df, use_container_width=True)
                
                # Tipps zur Reduzierung des CO2-Fußabdrucks
                st.subheader("Tipps zum CO2-Sparen")
                tips = [
                    "Wähle möglichst oft öffentliche Verkehrsmittel",
                    "Erwäge Fahrgemeinschaften für Autofahrten",
                    "Kompensiere deine CO2-Emissionen durch Klimaschutzprojekte",
                    "Vermeide Kurzstreckenflüge, wenn möglich",
                    "Kombiniere Reisen, um Wege zu sparen"
                ]
                for tip in tips:
                    st.write(f"- {tip}")
    
    # Vergleichende Darstellung
    st.header("CO2-Einsparpotential")
    if len(selected_transports) > 1:
        min_co2_transport = df.loc[df["CO2 pro Person (kg)"].idxmin()]["Transportmittel"]
        max_co2_transport = df.loc[df["CO2 pro Person (kg)"].idxmax()]["Transportmittel"]
        
        min_co2 = df[df["Transportmittel"] == min_co2_transport]["CO2 pro Person (kg)"].values[0]
        max_co2 = df[df["Transportmittel"] == max_co2_transport]["CO2 pro Person (kg)"].values[0]
        
        savings = max_co2 - min_co2
        saving_percentage = (savings / max_co2) * 100
        
        st.write(f"Durch die Wahl von **{min_co2_transport}** statt **{max_co2_transport}** kannst du **{savings:.2f} kg CO2** pro Person einsparen.")
        st.write(f"Das entspricht einer Reduktion von **{saving_percentage:.1f}%**!")
        
        # Visualisierung des Einsparpotentials
        saving_data = pd.DataFrame([
            {"Transport": max_co2_transport, "CO2 (kg)": max_co2, "Typ": "Höchste Emissionen"},
            {"Transport": min_co2_transport, "CO2 (kg)": min_co2, "Typ": "Niedrigste Emissionen"}
        ])
        
        fig = px.bar(
            saving_data,
            x="Transport",
            y="CO2 (kg)",
            color="Typ",
            title="CO2-Einsparpotential durch Wahl des umweltfreundlichsten Transportmittels",
            color_discrete_map={"Höchste Emissionen": "red", "Niedrigste Emissionen": "green"}
        )
        st.plotly_chart(fig, use_container_width=True)

# Abschnitt für zukünftige Erweiterungen
st.markdown("---")
st.subheader("Unterstützte Transportmittel")
col1, col2, col3 = st.columns(3)

with col1:
    st.write("🚗 Auto")
    st.write("✈️ Flugzeug")
with col2:
    st.write("🚂 Zug")
    st.write("🚌 Bus")
with col3:
    st.write("🏍️ Motorrad")
    st.write("🚲 Fahrrad")

# Über die App
st.markdown("---")
st.subheader("Über die App")
st.write("""
Diese App wurde im Rahmen eines Gruppenprojekts für den Kurs 'Grundlagen und Methoden der Informatik' 
an der Universität St. Gallen entwickelt. Sie hilft Nutzern, den ökologischen Fußabdruck ihrer Reisen 
zu verstehen und umweltbewusstere Entscheidungen zu treffen.
""")