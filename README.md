# EmissionExplorer - CO2-Fussabdruck-Tracker für Reisen

## Projektbeschreibung

EmissionExplorer ist eine Streamlit-Webanwendung, die entwickelt wurde, um den CO2-Fussabdruck von Reisen zu berechnen und zu visualisieren. Die App ermöglicht es Nutzern, verschiedene Transportmittel für ihre Reiseroute zu vergleichen und die umweltfreundlichste Option zu finden.

Diese Anwendung wurde im Rahmen des Gruppenprojekts für den Kurs "Grundlagen und Methoden der Informatik" an der Universität St. Gallen entwickelt.

## Funktionen

- **Routenberechnung**: Berechnung der Distanz zwischen Start- und Zielort
- **CO2-Vergleich**: Vergleich des CO2-Fussabdrucks verschiedener Transportmittel
- **Interaktive Visualisierung**: Graphische Darstellung der CO2-Emissionen
- **Personalisierte Berechnung**: Anpassung der Berechnung an die Anzahl der Reisenden
- **Machine Learning Analyse**: Erweiterte CO2-Vorhersagen für Autofahrten unter Berücksichtigung von Faktoren wie Fahrzeugtyp, alter, Jahreszeit und Verkehrsaufkommen
- **Umweltempfehlungen**: Tipps zur Reduzierung des CO2-Fussabdrucks

## Technische Anforderungen

Die App erfüllt alle im Projektauftrag geforderten Anforderungen:

1. **Klares Problem**: Die App löst das Problem der CO2-Fussabdruckberechnung für Reisen
2. **Datennutzung**: Die App lädt Geodaten über die Nominatim API
3. **Datenvisualisierung**: Interaktive Diagramme zur Darstellung der CO2-Emissionen
4. **Benutzerinteraktion**: Eingabe von Start- und Zielort sowie Auswahl verschiedener Transportmittel und Parameter
5. **Machine Learning**: Implementation eines Random Forest Regression Modells zur präzisen CO2-Vorhersage
6. **Dokumentation**: Ausführliche Kommentare im Quellcode

## Installation

### Voraussetzungen

- Python 3.7 oder höher
- pip (Python Package Manager)

### Installationsschritte

1. Repository klonen oder Dateien herunterladen

2. Ins Projektverzeichnis wechseln
   ```
   cd pfad/zu/emission-explorer
   ```

3. Virtuelle Umgebung erstellen (empfohlen)
   ```
   python -m venv venv
   ```

4. Virtuelle Umgebung aktivieren
     ```
     source venv/bin/activate
     ```

5. Abhängigkeiten installieren
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

## Datenquellen

- Distanzberechnung: OpenStreetMap via GeoPy
- CO2-Faktoren: Durchschnittswerte basierend auf öffentlichen Umweltdatenbanken
- ML-Modell: Trainiert auf synthetischen Daten (für Demonstrationszwecke)

## Projektbeitragende

Dieses Projekt wurde im Rahmen des Kurses "Grundlagen und Methoden der Informatik" an der Universität St. Gallen entwickelt.

## Hinweise zur Weiterentwicklung

Für die Weiterentwicklung der App könnten folgende Funktionen implementiert werden:
- Integration weiterer Transportmittel wie E-Scooter oder Fahrrad
- Verbindung zu echten Reisedaten-APIs für präzisere Routenberechnung
- Erweiterung des ML-Modells mit realen Trainingsdaten
- Implementierung einer Funktion zur CO2-Kompensationsberechnung
- Möglichkeit, mehrere Reisen zu speichern und zu vergleichen

## Hinweis zu KI-Unterstützung

In Übereinstimmung mit den Referenzierungsregeln für Generative KI an der HSG wurden Teile des Codes unter Verwendung von KI-Tools entwickelt. Die grundlegende Struktur und Logik der Anwendung wurden jedoch eigenständig konzipiert und implementiert.