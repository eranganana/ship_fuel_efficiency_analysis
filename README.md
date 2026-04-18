# Ship Fuel Efficiency and CO2 Emissions Analysis

## 📊 Project Overview
This project performs Monte Carlo simulation to analyze ship fuel efficiency and CO2 emissions using exponential distributions. The analysis includes:

- Statistical distribution fitting using MLE (Maximum Likelihood Estimation)
- Monte Carlo simulations for fuel consumption and CO2 emissions
- Comparative analysis by ship type, weather conditions, and fuel type
- Confidence interval estimation
- Visualization of simulation results vs actual data

## 🚢 Key Features
- **Distribution Analysis**: Identifies best-fit distributions for numerical variables
- **Monte Carlo Simulation**: 10,000 iterations for robust statistical inference
- **Category Analysis**: 
  - Ship type comparison
  - Weather conditions impact
  - Fuel type efficiency
- **Validation**: Compares simulated results with actual data

## 📈 Visualizations Included
- Pairplot for correlation analysis
- Histograms for simulated distributions
- KDE plots for comparative analysis
- Bar charts for categorical comparisons
- Line graphs for distribution comparisons

## 🛠️ Technologies Used
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy

## 📋 Results
The simulation provides:
- 95% confidence intervals for fuel consumption and CO2 emissions
- Comparative metrics across different categories
- Validation of exponential distribution assumptions

## 🚀 How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
  
3. Update the dataset path in the code
4. Run the analysis:
   ```bash
    python ship_fuel_efficiency_analysis.py
  
## 📊 Sample Output
Distance Simulation:
- mean: 150.98 km
- standard deviation: 149.61 km

  - Fuel Consumption 95% Confidence Interval: 122.54 - 17465.87 liters
- CO2 Emissions 95% Confidence Interval: 328.41 - 46808.52 kg

## 📝 Report
Detailed findings and methodology are available in the project report.

## 👩‍💻 Author
Achini Eranga Nanayakkara

## 📅 Date
2026-04-18
