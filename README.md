# 🗺️ convert2GeoJSON
A lightweight Python toolkit to convert geospatial, atmospheric science data into GeoJSON format. GeoJSON is a standardised geospatial data interchange format based on JavaScript Object Notation (JSON). Convert2GeoJSON takes data from a number of different sources (models and observations) and produces GeoJSON files useful for online mapping. The code is highly configurable for contouring data; allowing for data smoothing, custom metadata, contour specification, color selection, and coordinate precision.

---

## 🔧 Features

- Converts various structured geodata to GeoJSON `FeatureCollection`
- Adjustable coordinate rounding to manage file size and precision
- Customizable metadata and feature attributes
- Compact JSON output
- Easy to incorporate into Python scripts and pipelines

---

## 🚀 Installation

### Clone the repo
```bash
git clone https://github.com/earajr/convert2GeoJSON.git
cd convert2GeoJSON
```
### Download test data
```bash
wget https://homepages.see.leeds.ac.uk/~earajr/convert2GeoJSON/testdata.zip

