"""
Ship Fuel Efficiency and CO2 Emissions Analysis
Monte Carlo Simulation for Maritime Environmental Impact Assessment
Author: Achini Eranga Nanayakkara
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. DATA LOADING AND EXPLORATORY DATA ANALYSIS
# ============================================================================

# Load the dataset
df = pd.read_csv(r"C:\D\achini\computer intensive statistics\ship fuel efficiency analysis\ship_fuel_efficiency.csv")

# Print basic information about the dataset
print("=" * 80)
print("SHIP FUEL EFFICIENCY ANALYSIS - MONTE CARLO SIMULATION")
print("=" * 80)
print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst 5 rows of the dataset:")
print(df.head())

# Print descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Draw pairplot to identify the correlation among the variables
plt.figure(figsize=(10, 8))
sns.pairplot(df)
plt.suptitle("Pairplot of Numerical Variables", y=1.02)
plt.savefig('images/pairplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Get correlation for numerical variables
correlation_matrix = df[['distance', 'fuel_consumption', 'CO2_emissions', 'engine_efficiency']].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# ============================================================================
# 2. DISTRIBUTION FITTING USING MAXIMUM LIKELIHOOD ESTIMATION (MLE)
# ============================================================================

print("\n" + "=" * 80)
print("DISTRIBUTION FITTING USING MLE")
print("=" * 80)

for var in ['distance', 'CO2_emissions', 'fuel_consumption']:
    distributions = [st.laplace, st.norm, st.expon]
    mles = []
    
    for distribution in distributions:
        pars = distribution.fit(df[var])
        mle = distribution.nnlf(pars, df[var])
        mles.append(mle)
    
    best_fit = sorted(zip(distributions, mles), key=lambda d: d[1])[0]
    print(f"Best fit: {best_fit[0].name}, MLE value: {best_fit[1]:.2f}, for variable {var}")

# ============================================================================
# 3. MAIN MONTE CARLO SIMULATION
# ============================================================================

print("\n" + "=" * 80)
print("MAIN MONTE CARLO SIMULATION (10,000 iterations)")
print("=" * 80)

# Calculate lambda rate for each feature
fuel_lambda = 1 / df['fuel_consumption'].mean()
emission_factor = 2.68  # kg CO2 per liter of fuel
distance_lambda = 1 / df['distance'].mean()

# Number of simulations
num_simulations = 10000

# Simulate data using exponential distributions
simulated_distance = np.random.exponential(scale=1/distance_lambda, size=num_simulations)
simulated_fuel = np.random.exponential(scale=1/fuel_lambda, size=num_simulations)
simulated_emissions = simulated_fuel * emission_factor

# Analyse simulation results
print("\nDistance Simulation:")
print(f"Mean: {np.mean(simulated_distance):.2f} km")
print(f"Standard Deviation: {np.std(simulated_distance):.2f} km")
print(f"Min: {np.min(simulated_distance):.2f} km")
print(f"Max: {np.max(simulated_distance):.2f} km")

print("\nFuel Consumption Simulation:")
print(f"Mean: {np.mean(simulated_fuel):.2f} liters")
print(f"Standard Deviation: {np.std(simulated_fuel):.2f} liters")
print(f"Min: {np.min(simulated_fuel):.2f} liters")
print(f"Max: {np.max(simulated_fuel):.2f} liters")

print("\nCO2 Emissions Simulation:")
print(f"Mean: {np.mean(simulated_emissions):.2f} kg")
print(f"Standard Deviation: {np.std(simulated_emissions):.2f} kg")
print(f"Min: {np.min(simulated_emissions):.2f} kg")
print(f"Max: {np.max(simulated_emissions):.2f} kg")

# ============================================================================
# 4. VISUALIZATION OF SIMULATION RESULTS
# ============================================================================

# Plot simulated fuel consumption histogram
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(simulated_fuel, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
plt.title("Simulated Fuel Consumption Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Fuel Consumption (Liters)")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)

# Plot simulated CO2 emissions histogram
plt.subplot(1, 2, 2)
plt.hist(simulated_emissions, bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
plt.title("Simulated CO2 Emissions Distribution", fontsize=14, fontweight='bold')
plt.xlabel("CO2 Emissions (Kg)")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('images/simulation_histograms.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 5. CONFIDENCE INTERVALS
# ============================================================================

print("\n" + "=" * 80)
print("CONFIDENCE INTERVALS (95%)")
print("=" * 80)

fuel_lower, fuel_upper = np.percentile(simulated_fuel, [2.5, 97.5])
emissions_lower, emissions_upper = np.percentile(simulated_emissions, [2.5, 97.5])
distance_lower, distance_upper = np.percentile(simulated_distance, [2.5, 97.5])

print(f"Distance 95% Confidence Interval: {distance_lower:.2f} - {distance_upper:.2f} km")
print(f"Fuel Consumption 95% Confidence Interval: {fuel_lower:.2f} - {fuel_upper:.2f} liters")
print(f"CO2 Emissions 95% Confidence Interval: {emissions_lower:.2f} - {emissions_upper:.2f} kg")

# ============================================================================
# 6. ANALYSIS BY SHIP TYPE
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS BY SHIP TYPE")
print("=" * 80)

categories = df['ship_type'].unique()
results = {}

for category in categories:
    # Subset data for the current category
    subset = df[df['ship_type'] == category]
    
    # Calculate exponential lambda for fuel consumption and distance
    fuel_lambda = 1 / subset['fuel_consumption'].mean()
    distance_lambda = 1 / subset['distance'].mean()
    
    # Monte Carlo Simulation for this category
    simulated_distance = np.random.exponential(scale=1/distance_lambda, size=num_simulations)
    simulated_fuel = np.random.exponential(scale=1/fuel_lambda, size=num_simulations)
    simulated_emissions = simulated_fuel * emission_factor
    
    # Store results
    results[category] = {
        "mean_fuel": np.mean(simulated_fuel),
        "mean_emissions": np.mean(simulated_emissions),
        "distance_ci": np.percentile(simulated_distance, [2.5, 97.5]),
        "fuel_ci": np.percentile(simulated_fuel, [2.5, 97.5]),
        "emissions_ci": np.percentile(simulated_emissions, [2.5, 97.5]),
    }
    
    print(f"\n{category}:")
    print(f"  Mean Fuel: {results[category]['mean_fuel']:.2f} liters")
    print(f"  Mean CO2: {results[category]['mean_emissions']:.2f} kg")
    print(f"  Fuel CI: {results[category]['fuel_ci'][0]:.2f} - {results[category]['fuel_ci'][1]:.2f} liters")

# Plot CO2 emission by ship type
plt.figure(figsize=(10, 6))
ship_types = list(results.keys())
co2_means = [results[st]['mean_emissions'] for st in ship_types]
plt.bar(ship_types, co2_means, color='coral', edgecolor='black')
plt.title('CO2 Emissions by Ship Type', fontsize=14, fontweight='bold')
plt.xlabel('Ship Type')
plt.ylabel('CO2 Emissions (kg)')
plt.xticks(rotation=45)
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('images/co2_by_ship_type.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 7. ANALYSIS BY WEATHER CONDITIONS
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS BY WEATHER CONDITIONS")
print("=" * 80)

weather_conditions = df['weather_conditions'].unique()
weather_results = {}

for weather in weather_conditions:
    # Subset the dataset for the current weather condition
    subset = df[df['weather_conditions'] == weather]
    
    # Calculate lambda (rate) for fuel consumption and distance
    fuel_lambda = 1 / subset['fuel_consumption'].mean()
    distance_lambda = 1 / subset['distance'].mean()
    
    # Perform Monte Carlo simulations
    simulated_distance = np.random.exponential(scale=1/distance_lambda, size=num_simulations)
    simulated_fuel = np.random.exponential(scale=1/fuel_lambda, size=num_simulations)
    simulated_emissions = simulated_fuel * emission_factor
    
    # Store simulation results
    weather_results[weather] = {
        "mean_fuel": np.mean(simulated_fuel),
        "mean_emissions": np.mean(simulated_emissions),
        "distance_ci": np.percentile(simulated_distance, [2.5, 97.5]),
        "fuel_ci": np.percentile(simulated_fuel, [2.5, 97.5]),
        "emissions_ci": np.percentile(simulated_emissions, [2.5, 97.5]),
    }
    
    print(f"\n{weather}:")
    print(f"  Mean Fuel: {weather_results[weather]['mean_fuel']:.2f} liters")
    print(f"  Mean CO2: {weather_results[weather]['mean_emissions']:.2f} kg")

# Plot CO2 emission by weather condition
plt.figure(figsize=(10, 6))
weather_types = list(weather_results.keys())
co2_means_weather = [weather_results[w]['mean_emissions'] for w in weather_types]
plt.bar(weather_types, co2_means_weather, color='lightblue', edgecolor='black')
plt.title('CO2 Emissions by Weather Condition', fontsize=14, fontweight='bold')
plt.xlabel('Weather Condition')
plt.ylabel('CO2 Emissions (kg)')
plt.xticks(rotation=45)
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('images/co2_by_weather.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 8. ANALYSIS BY FUEL TYPE
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS BY FUEL TYPE")
print("=" * 80)

fuel_types = df['fuel_type'].unique()
fuel_type_results = {}

for fuel in fuel_types:
    # Subset the dataset for the current fuel type
    subset = df[df['fuel_type'] == fuel]
    
    # Calculate lambda (rate) for fuel consumption and distance
    fuel_lambda = 1 / subset['fuel_consumption'].mean()
    distance_lambda = 1 / subset['distance'].mean()
    
    # Use specific emission factors for different fuel types
    if fuel == "Diesel":
        emission_factor_fuel = 2.68  # kg CO2/L
    elif fuel == "LNG":
        emission_factor_fuel = 2.75  # kg CO2/L
    elif fuel == "Heavy Fuel Oil":
        emission_factor_fuel = 3.15  # kg CO2/L
    else:
        emission_factor_fuel = 2.5  # Default value
    
    # Perform Monte Carlo simulations
    simulated_distance = np.random.exponential(scale=1/distance_lambda, size=num_simulations)
    simulated_fuel = np.random.exponential(scale=1/fuel_lambda, size=num_simulations)
    simulated_emissions = simulated_fuel * emission_factor_fuel
    
    # Store simulation results
    fuel_type_results[fuel] = {
        "mean_fuel": np.mean(simulated_fuel),
        "mean_emissions": np.mean(simulated_emissions),
        "emission_factor": emission_factor_fuel,
        "distance_ci": np.percentile(simulated_distance, [2.5, 97.5]),
        "fuel_ci": np.percentile(simulated_fuel, [2.5, 97.5]),
        "emissions_ci": np.percentile(simulated_emissions, [2.5, 97.5]),
    }
    
    print(f"\n{fuel}:")
    print(f"  Emission Factor: {emission_factor_fuel:.2f} kg CO2/L")
    print(f"  Mean Fuel: {fuel_type_results[fuel]['mean_fuel']:.2f} liters")
    print(f"  Mean CO2: {fuel_type_results[fuel]['mean_emissions']:.2f} kg")

# Plot CO2 emission by fuel type
plt.figure(figsize=(10, 6))
fuel_list = list(fuel_type_results.keys())
co2_means_fuel = [fuel_type_results[f]['mean_emissions'] for f in fuel_list]
plt.bar(fuel_list, co2_means_fuel, color='lightgreen', edgecolor='black')
plt.title('CO2 Emissions by Fuel Type', fontsize=14, fontweight='bold')
plt.xlabel('Fuel Type')
plt.ylabel('CO2 Emissions (kg)')
plt.xticks(rotation=45)
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('images/co2_by_fuel_type.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 9. KDE PLOTS FOR DISTRIBUTION COMPARISONS
# ============================================================================

# Line graph for CO2 emissions by ship types
plt.figure(figsize=(12, 6))
for ship_type, metrics in results.items():
    simulated_emissions = np.random.exponential(
        scale=metrics['mean_emissions'], 
        size=num_simulations
    )
    sns.kdeplot(simulated_emissions, label=f"Ship Type: {ship_type}", linewidth=2)

plt.title("CO2 Emissions Distribution by Ship Types", fontsize=16, fontweight='bold')
plt.xlabel("CO2 Emissions (kg)", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(title="Ship Type", fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('images/co2_kde_by_ship_type.png', dpi=300, bbox_inches='tight')
plt.show()

# Line graph for CO2 emissions by weather conditions
plt.figure(figsize=(12, 6))
for weather, metrics in weather_results.items():
    simulated_emissions = np.random.exponential(
        scale=metrics['mean_emissions'], 
        size=num_simulations
    )
    sns.kdeplot(simulated_emissions, label=f"Weather: {weather}", linewidth=2)

plt.title("CO2 Emissions Distribution by Weather Conditions", fontsize=16, fontweight='bold')
plt.xlabel("CO2 Emissions (kg)", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(title="Weather Condition", fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('images/co2_kde_by_weather.png', dpi=300, bbox_inches='tight')
plt.show()

# Line graph for CO2 emissions by fuel types
plt.figure(figsize=(12, 6))
for fuel, metrics in fuel_type_results.items():
    simulated_emissions = np.random.exponential(
        scale=metrics['mean_emissions'], 
        size=num_simulations
    )
    sns.kdeplot(simulated_emissions, label=f"Fuel Type: {fuel}", linewidth=2)

plt.title("CO2 Emissions Distribution by Fuel Types", fontsize=16, fontweight='bold')
plt.xlabel("CO2 Emissions (kg)", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(title="Fuel Type", fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('images/co2_kde_by_fuel_type.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 10. MONTE CARLO SIMULATION FOR SHIP TYPE COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("MONTE CARLO SIMULATION RESULTS VS ACTUAL DATA")
print("=" * 80)

ship_types = df['ship_type'].unique()
simulation_results_ship_type = []

for ship_type in ship_types:
    subset = df[df['ship_type'] == ship_type]
    
    # Simulate fuel consumption and CO2 emissions using exponential distribution
    fuel_consumption_sim = np.random.exponential(scale=subset['fuel_consumption'].mean(), size=10000)
    co2_emissions_sim = np.random.exponential(scale=subset['CO2_emissions'].mean(), size=10000)
    
    simulation_results_ship_type.append((
        ship_type, 
        fuel_consumption_sim.mean(), 
        co2_emissions_sim.mean()
    ))

# Compare with actual data
ship_type_comparison = pd.DataFrame(
    simulation_results_ship_type, 
    columns=['ship_type', 'Simulated Fuel Consumption', 'Simulated CO2 Emissions']
)
ship_type_actual = df.groupby('ship_type')[['fuel_consumption', 'CO2_emissions']].mean().reset_index()
comparison_ship_type = pd.merge(
    ship_type_actual, 
    ship_type_comparison, 
    on='ship_type', 
    suffixes=('_Actual', '_Simulated')
)

print("\nComparison of Actual vs Simulated - Ship Types:")
print(comparison_ship_type.to_string(index=False))

# Plotting Comparison for Ship Type
plt.figure(figsize=(12, 6))
x = np.arange(len(comparison_ship_type['ship_type']))
width = 0.35

plt.bar(x - width/2, comparison_ship_type['fuel_consumption'], 
        width=width, label='Actual Fuel Consumption', color='blue', alpha=0.7)
plt.bar(x + width/2, comparison_ship_type['Simulated Fuel Consumption'], 
        width=width, label='Simulated Fuel Consumption', color='skyblue', alpha=0.7)
plt.xticks(x, comparison_ship_type['ship_type'], rotation=45)
plt.xlabel('Ship Type', fontsize=12)
plt.ylabel('Fuel Consumption (Liters)', fontsize=12)
plt.title('Comparison of Actual and Simulated Fuel Consumption by Ship Type', 
          fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('images/fuel_comparison_ship_type.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 11. WEATHER CONDITIONS COMPARISON
# ============================================================================

weather_conditions = df['weather_conditions'].unique()
simulation_results_weather = []

for weather in weather_conditions:
    subset = df[df['weather_conditions'] == weather]
    
    # Simulate fuel consumption and CO2 emissions
    fuel_consumption_sim = np.random.exponential(subset['fuel_consumption'].mean(), size=10000)
    co2_emissions_sim = np.random.exponential(subset['CO2_emissions'].mean(), size=10000)
    
    simulation_results_weather.append((
        weather, 
        fuel_consumption_sim.mean(), 
        co2_emissions_sim.mean()
    ))

# Create a comparison DataFrame
weather_comparison = pd.DataFrame(
    simulation_results_weather, 
    columns=['weather_conditions', 'Simulated Fuel Consumption', 'Simulated CO2 Emissions']
)
weather_actual = df.groupby('weather_conditions')[['fuel_consumption', 'CO2_emissions']].mean().reset_index()
comparison_weather = pd.merge(
    weather_actual, 
    weather_comparison, 
    on='weather_conditions', 
    suffixes=('_Actual', '_Simulated')
)

print("\nComparison of Actual vs Simulated - Weather Conditions:")
print(comparison_weather.to_string(index=False))

# Plotting Comparison for Weather Type
plt.figure(figsize=(12, 6))
x = np.arange(len(comparison_weather['weather_conditions']))
width = 0.35

plt.bar(x - width/2, comparison_weather['CO2_emissions'], 
        width=width, label='Actual CO2 Emission', color='blue', alpha=0.7)
plt.bar(x + width/2, comparison_weather['Simulated CO2 Emissions'], 
        width=width, label='Simulated CO2 Emission', color='skyblue', alpha=0.7)
plt.xticks(x, comparison_weather['weather_conditions'], rotation=45)
plt.xlabel('Weather Condition', fontsize=12)
plt.ylabel('CO2 Emission (kg)', fontsize=12)
plt.title('Comparison of Actual and Simulated CO2 Emissions for Weather Conditions', 
          fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('images/co2_comparison_weather.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 12. FUEL TYPE COMPARISON
# ============================================================================

fuel_types = df['fuel_type'].unique()
simulation_results_fuel = []

for fuel in fuel_types:
    subset = df[df['fuel_type'] == fuel]
    
    # Simulate CO2 emissions
    co2_emissions_sim = np.random.exponential(scale=subset['CO2_emissions'].mean(), size=10000)
    
    simulation_results_fuel.append((fuel, co2_emissions_sim.mean()))

# Create a comparison DataFrame
fuel_comparison = pd.DataFrame(simulation_results_fuel, columns=['fuel_type', 'Simulated CO2 Emissions'])
fuel_actual = df.groupby('fuel_type')[['CO2_emissions']].mean().reset_index()
comparison_fuel = pd.merge(
    fuel_actual, 
    fuel_comparison, 
    on='fuel_type', 
    suffixes=('_Actual', '_Simulated')
)

print("\nComparison of Actual vs Simulated - Fuel Types:")
print(comparison_fuel.to_string(index=False))

# Plotting Comparison for Fuel Type
plt.figure(figsize=(10, 6))
x = np.arange(len(comparison_fuel['fuel_type']))
width = 0.35

plt.bar(x - width/2, comparison_fuel['CO2_emissions'], 
        width=width, label='Actual CO2 Emission', color='blue', alpha=0.7)
plt.bar(x + width/2, comparison_fuel['Simulated CO2 Emissions'], 
        width=width, label='Simulated CO2 Emission', color='skyblue', alpha=0.7)
plt.xticks(x, comparison_fuel['fuel_type'], rotation=45)
plt.xlabel('Fuel Type', fontsize=12)
plt.ylabel('CO2 Emission (kg)', fontsize=12)
plt.title('Comparison of Actual and Simulated CO2 Emissions by Fuel Type', 
          fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('images/co2_comparison_fuel_type.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 13. MONTHLY ANALYSIS
# ============================================================================

plt.figure(figsize=(12, 6))
monthly_co2 = df.groupby('month')['CO2_emissions'].mean()
plt.bar(range(len(monthly_co2)), monthly_co2.values, color='purple', alpha=0.7, edgecolor='black')
labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.xticks(range(0, 12), labels, rotation=45)
plt.title('Comparison of CO2 Emissions by Month', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('CO2 Emissions (kg)')
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('images/co2_by_month.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 14. SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("PROJECT SUMMARY")
print("=" * 80)

print(f"\nTotal Monte Carlo Simulations Performed: {num_simulations:,}")
print(f"Number of Ship Types Analyzed: {len(ship_types)}")
print(f"Number of Weather Conditions Analyzed: {len(weather_conditions)}")
print(f"Number of Fuel Types Analyzed: {len(fuel_types)}")

print("\nKey Findings:")
print(f"1. Average CO2 emissions: {df['CO2_emissions'].mean():.2f} kg per trip")
print(f"2. Average fuel consumption: {df['fuel_consumption'].mean():.2f} liters per trip")
print(f"3. Most efficient ship type: {comparison_ship_type.loc[comparison_ship_type['fuel_consumption'].idxmin(), 'ship_type']}")
print(f"4. Most efficient fuel type: {comparison_fuel.loc[comparison_fuel['CO2_emissions'].idxmin(), 'fuel_type']}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 80)