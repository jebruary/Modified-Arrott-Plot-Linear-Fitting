# -*- coding: utf-8 -*-
"""
This script performs a scaling analysis on magnetization vs. magnetic field data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

def analyze_and_plot_data():
    """
    This script performs a scaling analysis on magnetization vs. magnetic field data.

    It prompts the user for:
    1. Beta (β) exponent
    2. Gamma (γ) exponent
    3. Critical magnetic field (Bc)
    4. Paths to multiple data files.

    For each dataset, it calculates transformed variables:
    - y = M^(1/β)
    - x = (B/M)^(1/γ)

    It then finds the data point closest to the critical magnetic field,
    and performs a linear fit on the transformed data points greater than
    or equal to this critical point.

    Finally, it plots all transformed data and their corresponding linear fits
    on a single graph and prints the y-intercept of each fit.
    """
    try:
      
        beta = float(input("Enter the value for beta (β): "))
        gamma = float(input("Enter the value for gamma (γ): "))
        critical_b = float(input("Enter the critical magnetic field (Bc): "))

        data_files = []
        while True:
            file_path = input("Enter the path to a data file (or press Enter to finish): ")
            if not file_path:
                break
            if os.path.exists(file_path):
                data_files.append(file_path)
            else:
                print(f"Warning: File not found at '{file_path}'. Please check the path.")

        if not data_files:
            print("No data files were provided. Exiting.")
            return

        plt.figure(figsize=(10, 8))
        plt.xlabel(fr'$(B/M)^{{1/\gamma}}$ (where $\gamma$ = {gamma})')
        plt.ylabel(fr'$M^{{1/\beta}}$ (where $\beta$ = {beta})')
        plt.title('Scaling Plot of Magnetization Data')
        plt.grid(True)

        intercepts = {}

        for i, file_path in enumerate(data_files):
            try:
             
                data = np.loadtxt(file_path)
                B = data[:, 0]  
                M = data[:, 1] 

                valid_indices = M > 0
                if not np.any(valid_indices):
                    print(f"Skipping {os.path.basename(file_path)}: No valid magnetization data (M > 0).")
                    continue

                B = B[valid_indices]
                M = M[valid_indices]

              
                y = M**(1/beta)
                x_base = B / M
                
               
                valid_x_indices = x_base >= 0
                x = x_base[valid_x_indices]**(1/gamma)
                y = y[valid_x_indices]
                B_filtered = B[valid_x_indices]

      
                dataset_label = f'Dataset: {os.path.basename(file_path)}'
                plt.scatter(x, y, label=dataset_label, alpha=0.6)

              
                critical_idx = np.argmin(np.abs(B_filtered - critical_b))

                
                x_fit = x[critical_idx:].reshape(-1, 1)
                y_fit = y[critical_idx:]

                if len(x_fit) < 2:
                    print(f"Warning for {os.path.basename(file_path)}: Not enough data points (need at least 2) for linear fitting after the critical field. Skipping fit.")
                    continue

             
                model = LinearRegression()
                model.fit(x_fit, y_fit)
                intercept = model.intercept_
                slope = model.coef_[0]

                intercepts[os.path.basename(file_path)] = intercept

                fit_line_x = np.array([np.min(x_fit), np.max(x_fit)]).reshape(-1, 1)
                fit_line_y = model.predict(fit_line_x)

                plt.plot(fit_line_x, fit_line_y,
                         linestyle='--',
                         label=f'Fit for {os.path.basename(file_path)} (Intercept: {intercept:.4f})')

            except Exception as e:
                print(f"An error occurred while processing {os.path.basename(file_path)}: {e}")

      
        plt.legend()
        plt.show()

        print("\n--- Intercepts of Linear Fittings ---")
        if intercepts:
            for filename, intercept_val in intercepts.items():
                print(f"  - {filename}: {intercept_val:.6f}")
        else:
            print("No intercepts were calculated.")

    except ValueError:
        print("Invalid input. Please make sure beta, gamma, and the critical field are numbers.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':

    analyze_and_plot_data()
