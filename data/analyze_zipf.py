"""
Analyze Zipfian distribution parameter (s) from real population data
using the CORRECT Log-Log Regression method (Gabaix, Krugman standard).

Zipf's Law: Population(rank) = C / rank^s
In log-log space: log(Pop) = log(C) - s * log(rank)
The slope of the log-log regression gives us -s
"""
import numpy as np
from scipy import stats
import os
import glob

def read_population_csv(filepath):
    """Read population data from CSV file (semicolon or comma separated)"""
    data = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
        
    # Detect separator
    header = lines[0]
    sep = ';' if ';' in header else ','
    
    for line in lines[1:]:  # Skip header
        line = line.strip()
        if not line or line.startswith(';') or line.startswith(','):
            continue
            
        parts = line.split(sep)
        if len(parts) >= 2:
            name = parts[0].strip()
            pop_str = parts[1].strip().replace(',', '').replace('.', '')
            
            # Skip summary rows
            if any(skip in name.lower() for skip in ['total', 'indonesia', 'catatan', 'note']):
                continue
                
            try:
                population = int(pop_str)
                if population > 0:
                    data.append((name, population))
            except ValueError:
                continue
                
    return data

def fit_zipf_log_log(data, name):
    """
    Fit Zipfian distribution using LOG-LOG LINEAR REGRESSION.
    
    This is the standard method used by Gabaix (1999), Krugman, and 
    papers in Nature/Science on Zipf's Law.
    
    Zipf's Law: P(r) = C / r^s
    Taking log: log(P) = log(C) - s * log(r)
    
    So the slope of log(P) vs log(r) gives us -s
    """
    # Extract populations and sort descending
    populations = [pop for _, pop in data]
    sorted_data = np.array(sorted(populations, reverse=True), dtype=float)
    ranks = np.arange(1, len(sorted_data) + 1, dtype=float)
    
    # Take logarithms
    log_ranks = np.log(ranks)
    log_pops = np.log(sorted_data)
    
    # Linear regression in log-log space
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_pops)
    
    # The Zipf exponent s is the NEGATIVE of the slope
    s = -slope
    r_squared = r_value ** 2
    
    # Calculate concentration metrics
    top_10_idx = max(1, len(sorted_data) // 10)
    top_20_idx = max(1, len(sorted_data) // 5)
    top_10_pct = sorted_data[:top_10_idx].sum() / sorted_data.sum() * 100
    top_20_pct = sorted_data[:top_20_idx].sum() / sorted_data.sum() * 100
    
    print(f"\n{'='*60}")
    print(f"Country: {name}")
    print(f"{'='*60}")
    print(f"Number of regions: {len(data)}")
    print(f"")
    print(f"LOG-LOG REGRESSION RESULTS:")
    print(f"  Slope (log-log):        {slope:.4f}")
    print(f"  Zipf exponent s:        {s:.4f}")
    print(f"  R-squared:              {r_squared:.4f}")
    print(f"  Standard error:         {std_err:.4f}")
    print(f"  p-value:                {p_value:.2e}")
    print(f"")
    print(f"CONCENTRATION METRICS:")
    print(f"  Top 10% regions hold:   {top_10_pct:.1f}% of population")
    print(f"  Top 20% regions hold:   {top_20_pct:.1f}% of population")
    print(f"  Largest/Smallest ratio: {sorted_data[0]/sorted_data[-1]:.1f}x")
    
    return s, r_squared, std_err

def main():
    # Find all CSV files in the data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = glob.glob(os.path.join(script_dir, "*.csv"))
    
    if not csv_files:
        print("No CSV files found in data directory")
        return
    
    results = {}
    
    for csv_file in sorted(csv_files):
        filename = os.path.basename(csv_file)
        # Extract country name from filename
        country = filename.replace('-Population-per-Province.csv', '') \
                          .replace('-Population-per-Land.csv', '') \
                          .replace('-Population-per-State.csv', '') \
                          .replace('.csv', '') \
                          .replace('-', ' ')
        
        print(f"\nReading: {filename}")
        data = read_population_csv(csv_file)
        
        if len(data) >= 5:  # Need at least 5 data points
            s, r2, std_err = fit_zipf_log_log(data, country)
            results[country] = (s, r2, std_err)
        else:
            print(f"  Skipping: only {len(data)} valid rows found")
    
    # Summary
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY: Zipfian s Parameters (Log-Log Regression)")
        print(f"{'='*60}")
        print(f"{'Country':<15} {'s':<10} {'R²':<10} {'Std Err':<10}")
        print("-" * 45)
        for country, (s, r2, std_err) in sorted(results.items()):
            print(f"{country:<15} {s:<10.4f} {r2:<10.4f} {std_err:<10.4f}")
        
        avg_s = np.mean([s for s, _, _ in results.values()])
        print("-" * 45)
        print(f"{'Average':<15} {avg_s:<10.4f}")
        print(f"\nRecommendation: Use s ≈ {avg_s:.2f} for experiments")

if __name__ == "__main__":
    main()
