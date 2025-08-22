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

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlabel(fr'$(B/M)^{{1/\gamma}}$ (where $\gamma$ = {gamma})')
        ax.set_ylabel(fr'$M^{{1/\beta}}$ (where $\beta$ = {beta})')
        ax.set_title('Scaling Plot of Magnetization Data')
        ax.grid(True)
        
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        intercepts = {}
        all_x_data_for_plot = []
        all_y_data_for_plot = []

        for i, file_path in enumerate(data_files):
            try:
                
              
                filename_without_ext = os.path.splitext(os.path.basename(file_path))[0]
                
                data = np.loadtxt(file_path)
                B = data[:, 0]
                M = data[:, 1]
                
                valid_indices = M > 0
                if not np.any(valid_indices):
                    print(f"Skipping {filename_without_ext}: No valid magnetization data (M > 0).")
                    continue

                B = B[valid_indices]
                M = M[valid_indices]

                y = M**(1/beta)
                x_base = B / M
                
                valid_x_indices = x_base >= 0
                x = x_base[valid_x_indices]**(1/gamma)
                y = y[valid_x_indices]
                B_filtered = B[valid_x_indices]
                
                ax.plot(x, y, 
                        marker='o', 
                        linestyle='none', 
                        markerfacecolor='none', 
                        markeredgecolor=f'C{i}', 
                        label=f'{filename_without_ext}') 

                critical_idx = np.argmin(np.abs(B_filtered - critical_b))

                x_fit = x[critical_idx:].reshape(-1, 1)
                y_fit = y[critical_idx:]

                if len(x_fit) < 2:
                    print(f"Warning for {filename_without_ext}: Not enough data points (need at least 2) for linear fitting after the critical field. Skipping fit.")
                    continue

                model = LinearRegression()
                model.fit(x_fit, y_fit)
                intercept = model.intercept_
                slope = model.coef_[0]

                intercepts[filename_without_ext] = intercept 
        
                x_fit_start = x_fit.min()
                x_fit_end = x_fit.max()
                y_fit_start = model.predict([[x_fit_start]])[0]
                y_fit_end = model.predict([[x_fit_end]])[0]
                
                ax.plot([x_fit_start, x_fit_end], [y_fit_start, y_fit_end],
                        linestyle='-', 
                        linewidth=3,    
                        color=f'C{i}',
                        label=f'Fit for {filename_without_ext}') 
                
                ax.plot([0, x_fit_start], [intercept, y_fit_start],
                        linestyle='--',
                        linewidth=1.5,
                        color=f'C{i}')
                
                all_x_data_for_plot.append(x)
                all_y_data_for_plot.append(y)
                all_y_data_for_plot.append(np.array([intercept])) 

            except Exception as e:
                print(f"An error occurred while processing {filename_without_ext}: {e}") 
        

        if all_x_data_for_plot and all_y_data_for_plot:
            global_x_max = np.max(np.concatenate(all_x_data_for_plot))
            global_y_min = np.min(np.concatenate(all_y_data_for_plot))
            global_y_max = np.max(np.concatenate(all_y_data_for_plot))
            
            ax.set_xlim(0, global_x_max * 1.1)
            ax.set_ylim(global_y_min * 1.5, global_y_max * 1.1)

        ax.legend()
        
        plt.savefig('scaling_plot.png')

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
