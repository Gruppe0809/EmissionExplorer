# Grundlagen und Methoden der Informatik für Wirtschaftswissenschaften
Gruppe 08.09

## EmissionExplorer - CO2-Fussabdruck-Tracker für Reisen

EmissionExplorer ist eine Streamlit-Webanwendung, die entwickelt wurde, um den CO2-Fussabdruck von Reisen zu berechnen und zu visualisieren. Die App ermöglicht es Nutzern, verschiedene Transportmittel für ihre Reiseroute zu vergleichen und die umweltfreundlichste Option zu finden.

Diese Anwendung wurde im Rahmen des Gruppenprojekts für den Kurs "Grundlagen und Methoden der Informatik für Wirtschaftswissenschaften" an der Universität St. Gallen entwickelt.

## Funktionen

- **Routenberechnung**: Berechnung der Distanz zwischen Start- und Zielort
- **CO2-Vergleich**: Vergleich des CO2-Fussabdrucks verschiedener Transportmittel
- **Interaktive Visualisierung**: Graphische Darstellung der CO2-Emissionen
- **Personalisierte Berechnung**: Anpassung der Berechnung an die Anzahl der Reisenden
- **Machine Learning Analyse**: Erweiterte CO2-Vorhersagen für Autofahrten unter Berücksichtigung von Faktoren wie Fahrzeugtyp, alter, Jahreszeit und Verkehrsaufkommen
- **Umweltempfehlungen**: Tipps zur Reduzierung des CO2-Fussabdrucks

## Technische Anforderungen

1. **Klares Problem**: Die App löst das Problem der CO2-Fussabdruckberechnung für Reisen
2. **Datennutzung**: Die App lädt Geodaten über die Nominatim API
3. **Datenvisualisierung**: Interaktive Diagramme zur Darstellung der CO2-Emissionen
4. **Benutzerinteraktion**: Eingabe von Start- und Zielort sowie Auswahl verschiedener Transportmittel und Parameter
5. **Machine Learning**: Implementation eines Random Forest Regression Modells zur präzisen CO2-Vorhersage
6. **Dokumentation**: Ausführliche Kommentare im Quellcode

## Installation

### Voraussetzungen

- Python 3.8 oder höher
- pip (Python Package Manager)

### Installationsschritte

1. Repository klonen oder Dateien herunterladen
   ```
   git clone https://github.com/Gruppe0809/EmissionExplorer
   cd EmissionExplorer
   ```

2. Virtuelle Umgebung erstellen (empfohlen)
   ```
   python -m venv venv
   ```

3. Virtuelle Umgebung aktivieren
     ```
     source venv/bin/activate
     ```

4. Abhängigkeiten installieren
   ```
   pip install -r requirements.txt
   ```

## Anwendung starten

Nach der Installation kann die App mit folgendem Befehl gestartet werden:

```
streamlit run app.py
```

Nach der Ausführung dieses Befehls wird ein lokaler Webserver gestartet und die Anwendung sollte automatisch in deinem Standardbrowser geöffnet werden (typischerweise unter http://localhost:8501).

## Benutzung

1. Gib den Startpunkt und das Ziel deiner Reise ein
2. Wähle die Transportmittel aus, die du vergleichen möchtest
3. Gib die Anzahl der Reisenden an
4. Optional: Aktiviere die erweiterte CO2-Berechnung mit Machine Learning für detailliertere Analysen von Autofahrten
5. Klicke auf "CO2-Fussabdruck berechnen"
6. Analysiere die Ergebnisse in den verschiedenen Visualisierungen und Tabellen

## Externe Abhängigkeiten

- **streamlit**: Erstellt die interaktive webbasierte Benutzeroberfläche für den CO2-Fußabdruck-Tracker
- **pandas**: Verarbeitet Daten und unterstützt die Analyse von Transportemissionen
- **plotly.express**: Erstellt interaktive Visualisierungen für CO2-Vergleiche
- **numpy**: Handhabt numerische Berechnungen und die Erzeugung synthetischer Daten
- **geopy**: Bietet Geocoding-Funktionalität, um Adressen in Koordinaten umzuwandeln und Entfernungen zu berechnen
- **scikit-learn**: Implementiert Machine-Learning-Modelle (Random Forest) für CO2-Vorhersagen
- **requests**: Ermöglicht HTTP-Anfragen für Standortdaten

## Quellen und Referenzen

- **Streamlit Dokumentation**: https://docs.streamlit.io
- **Pandas Dokumentation**: https://pandas.pydata.org/docs/
- **Plotly Dokumentation**: https://plotly.com/python/
- **GeoPy Dokumentation**: https://geopy.readthedocs.io/
- **Scikit-learn Dokumentation**: https://scikit-learn.org/stable/
- **OpenStreetMap**: https://www.openstreetmap.org/
- **CO2-Emissionsdaten**: https://www.umweltbundesamt.de/themen/verkehr-laerm/emissionsdaten
- **ChatGPT und GitHub Copilot**: Code-Optimierung, Debugging und Strukturierung des Readme Files.

## Hinweise zur Weiterentwicklung

Für die Weiterentwicklung der App könnten folgende Funktionen implementiert werden:
- Integration weiterer Transportmittel wie E-Scooter oder Fahrrad
- Verbindung zu echten Reisedaten-APIs für präzisere Routenberechnung
- Erweiterung des ML-Modells mit realen Trainingsdaten
- Implementierung einer Funktion zur CO2-Kompensationsberechnung
- Möglichkeit, mehrere Reisen zu speichern und zu vergleichen