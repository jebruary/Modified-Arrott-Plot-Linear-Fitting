# -*- coding: utf-8 -*-
"""
This code makes several linear fittings using the beta and gamma provided by
the user.


"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

input_file_name = "branch21K.dat"

def read_magnetization_data(filename: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Reads a two-column data file, skipping header lines.

    This function opens a text file and parses it, assuming the first
    column is the magnetic field and the second is magnetization. It
    automatically skips any initial lines that are not valid data.

    Args:
        filename: The path to the data file.

    Returns:
        A tuple containing two NumPy arrays (magnetic_field, magnetization).
        If the file cannot be read or is empty, it returns (None, None).
    """
    magnetic_field = []
    magnetisation = []

    try:
        with open(filename, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    parts = line.split()
                    field = float(parts[0])
                    mag = float(parts[1])
                    magnetic_field.append(field)
                    magnetisation.append(mag)
                except (ValueError, IndexError):
                    continue
            
            if not magnetic_field:
                print("Error: No numerical data found in the file.")
                return None, None

            return np.array(magnetic_field), np.array(magnetisation)

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None, None

def get_user_parameters() -> tuple[np.ndarray, np.ndarray, float, float, float, float, int]:
    """
    Asks the user for central values of gamma, beta, the critical
    magnetic field, step size, and number of steps. It then creates
    a range of points around gamma and beta.

    Returns:
        A tuple containing the gamma range, beta range, central gamma,
        central beta, the critical magnetic field, step size, and number of steps.
    """
    gamma = None
    beta = None
    critical_field = None
    step = None
    num_steps = None

    while gamma is None:
        try:
            gamma_input = input("Enter the central value for gamma: ")
            temp_gamma = float(gamma_input)
            if temp_gamma < 0:
                print("Invalid input. Gamma cannot be negative. Please enter a positive value.")
            else:
                gamma = temp_gamma
        except ValueError:
            print("Invalid input. Please enter a number for gamma.")

    while beta is None:
        try:
            beta_input = input("Enter the central value for beta: ")
            temp_beta = float(beta_input)
            if temp_beta < 0:
                print("Invalid input. Beta cannot be negative. Please enter a positive value.")
            else:
                beta = temp_beta
        except ValueError:
            print("Invalid input. Please enter a number for beta.")
            
    while critical_field is None:
        try:
            critical_field_input = input("Enter the critical magnetic field (Hc): ")
            critical_field = float(critical_field_input)
        except ValueError:
            print("Invalid input. Please enter a number for the critical magnetic field.")

    while step is None:
        try:
            step_input = input("Enter the step size for gamma and beta (e.g., 0.05). This can be a float: ")
            step = float(step_input)
            if step <= 0:
                print("Invalid input. Step size must be a positive number.")
                step = None 
            else:
                print(f"Warning: Step size ({step}) is a float. Calculations will proceed with this value.")
        except ValueError:
            print("Invalid input. Please enter a number for the step size.")

    while num_steps is None:
        try:
            num_steps_input = input("Enter the number of steps to explore on each side (e.g., 5). This must be an integer: ")
            num_steps = int(num_steps_input)
            if num_steps < 0:
                print("Invalid input. Number of steps cannot be negative. Please enter a non-negative integer.")
                num_steps = None 
        except ValueError:
            print("Invalid input. Please enter an integer for the number of steps.")

    offsets = np.arange(-num_steps, num_steps + 1) * step
    
    gamma_range = gamma + offsets
    beta_range = beta + offsets

    return gamma_range, beta_range, gamma, beta, critical_field, step, num_steps

def create_and_save_scaled_datasets(m_data, b_field, beta_values, gamma_values, folder_path, step, num_steps):
    """
    Generates scaled datasets and saves each one to a text file.

    Args:
        m_data: NumPy array of magnetization data.
        b_field: NumPy array of magnetic field data.
        beta_values: NumPy array of beta values to test.
        gamma_values: NumPy array of gamma values to test.
        folder_path: The path to the directory where files will be saved.
        step (float): The step size used for gamma and beta.
        num_steps (int): The number of steps used for gamma and beta.

    Returns:
        A list of dictionaries for further processing.
    """
    print("\n--- Creating and Saving Scaled Datasets ---")
    scaled_datasets = []
    
    
   
    valid_data_mask = (m_data > 0) & (b_field > 0)
    m_data_filtered = m_data[valid_data_mask]
    b_field_filtered = b_field[valid_data_mask]

    if m_data_filtered.size == 0:
        print("Warning: No data with positive M and B found. Cannot create scaled datasets.")
        return []

    for gamma in gamma_values:
        for beta in beta_values:
            y_scaled = m_data_filtered ** (1 / beta)
            x_scaled = (b_field_filtered / m_data_filtered) ** (1 / gamma)
            
            dataset = {
                'gamma': gamma,
                'beta': beta,
                'x_data': x_scaled,
                'y_data': y_scaled,
                'original_b_field': b_field_filtered,
            }

            scaled_datasets.append(dataset)

            file_name = f"scaled_data_gamma_{gamma:.2f}_beta_{beta:.2f}_step_{step:.2f}_nsteps_{num_steps}.txt"
            file_path = os.path.join(folder_path, file_name)
            
            output_data = np.column_stack((b_field_filtered, m_data_filtered, x_scaled, y_scaled))
            header = f"Original_B_Field\tOriginal_Magnetization\tScaled_X_Data\tScaled_Y_Data\nGamma: {gamma:.2f}, Beta: {beta:.2f}, Step: {step:.2f}, Num_Steps: {num_steps}"
            np.savetxt(file_path, output_data, delimiter='\t', header=header, fmt='%.8f')
            
    print(f"Successfully created and saved {len(scaled_datasets)} scaled datasets in '{folder_path}'")
    return scaled_datasets

def perform_fitting_and_save_plots(datasets, hc_value, folder_path, input_filename, step, num_steps):
    """
    Performs linear fitting, finds the greatest normalised intercept and R-squared,
    saves all plots, and displays the two best plots.
    """
    print("\n--- Performing Linear Fitting and Saving Plots ---")
    
    all_results = []
    for dataset in datasets:
        gamma, beta = dataset['gamma'], dataset['beta']
        x_data, y_data = dataset['x_data'], dataset['y_data']
        original_b = dataset['original_b_field']

        start_index = np.argmin(np.abs(original_b - hc_value))
        x_fit, y_fit = x_data[start_index:], y_data[start_index:]
        
        if len(y_fit) < 2 or np.var(y_fit) == 0:
            continue

        slope, intercept = np.polyfit(x_fit, y_fit, 1)
        y_predicted = slope * x_fit + intercept
        
        correlation_matrix = np.corrcoef(y_fit, y_predicted)
        r_squared = correlation_matrix[0, 1]**2
        
        max_y_value = np.max(y_data)
        normalised_intercept = intercept / max_y_value if max_y_value != 0 else 0
        
      
        if np.isnan(r_squared) or np.isnan(intercept):
            continue

        all_results.append({
            'gamma': gamma, 'beta': beta, 'intercept': intercept, 'r_squared': r_squared,
            'normalised_intercept': normalised_intercept,
            'x_data': x_data, 'y_data': y_data, 'x_fit': x_fit, 'y_predicted': y_predicted
        })

    if not all_results:
        print("Could not find any valid data to fit. Try adjusting Hc or the gamma/beta ranges.")
        return [], {}, {}

    
    best_r_squared_params = max(all_results, key=lambda x: x['r_squared'])
    greatest_norm_intercept_params = max(all_results, key=lambda x: x['normalised_intercept'])

    
    for result in all_results:
        gamma, beta = result['gamma'], result['beta']
        fig = plt.figure()
        plt.scatter(result['x_data'], result['y_data'], label='Scaled Data', alpha=0.6)
        plt.plot(result['x_fit'], result['y_predicted'], color='red', linewidth=2, label='Linear Fit')
        
        plt.title(f'{input_filename} | Fit: γ={gamma:.2f}, β={beta:.2f}\nStep: {step:.2f}, Num Steps: {num_steps}')
        
        plt.xlabel(f'(B/M)^(1/{gamma:.2f})')
        plt.ylabel(f'M^(1/{beta:.2f})')
        
        fit_text = (f"Intercept = {result['intercept']:.6f}\n"
                    f"Norm. Intercept = {result['normalised_intercept']:.6f}\n"
                    f"R² = {result['r_squared']:.6f}")
                    
        plt.text(0.05, 0.95, fit_text, transform=plt.gca().transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.legend(loc='lower right')
        plt.grid(True)
        
        save_path = folder_path
        filename = f"plot_gamma_{gamma:.2f}_beta_{beta:.2f}_step_{step:.2f}_nsteps_{num_steps}.png"

        if result == greatest_norm_intercept_params:
            save_path = "."
            filename = f"GREATEST_NORM_INTERCEPT_gamma_{gamma:.2f}_beta_{beta:.2f}_step_{step:.2f}_nsteps_{num_steps}.png"
        if result == best_r_squared_params:
            save_path = "."
            filename = f"BEST_R_SQUARED_gamma_{gamma:.2f}_beta_{beta:.2f}_step_{step:.2f}_nsteps_{num_steps}.png"
        
        plt.savefig(os.path.join(save_path, filename))
        plt.close(fig)

    print(f"Saved all 2D plots. Best plots are in the main directory, others are in '{folder_path}'.")

    
    def show_plot(result, title_prefix):
        fig = plt.figure()
        gamma, beta = result['gamma'], result['beta']
        plt.scatter(result['x_data'], result['y_data'], label='Scaled Data', alpha=0.6)
        plt.plot(result['x_fit'], result['y_predicted'], color='red', linewidth=2, label='Linear Fit')
        
        plt.title(f"{input_filename} | {title_prefix}\nγ={gamma:.2f}, β={beta:.2f}\nStep: {step:.2f}, Num Steps: {num_steps}")
        
        plt.xlabel(f'(B/M)^(1/{gamma:.2f})')
        plt.ylabel(f'M^(1/{beta:.2f})')
        
        fit_text = (f"Intercept = {result['intercept']:.6f}\n"
                    f"Norm. Intercept = {result['normalised_intercept']:.6f}\n"
                    f"R² = {result['r_squared']:.6f}")
                    
        plt.text(0.05, 0.95, fit_text, transform=plt.gca().transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.legend(loc='lower right')
        plt.grid(True)

    print("\n--- Displaying Best Plots ---")
    if greatest_norm_intercept_params:
        show_plot(greatest_norm_intercept_params, "Greatest Normalised Intercept")
    if best_r_squared_params:
        show_plot(best_r_squared_params, "Highest R-squared (Best Fit)")
    
    plt.show()

    return all_results, greatest_norm_intercept_params, best_r_squared_params


def create_summary_plots(all_results, greatest_norm_intercept, best_r_squared, folder_path, input_filename, step, num_steps):
    """
    Creates, saves, and displays 2D and 3D scatter plots for R-squared and
    Intercept values.

    Args:
        all_results (list): A list of dictionaries containing all fitting results.
        greatest_norm_intercept (dict): The result dictionary for the greatest norm intercept.
        best_r_squared (dict): The result dictionary for the best R-squared.
        folder_path (str): The path to save the plots.
        input_filename (str): The name of the original data file.
        step (float): The step size used for gamma and beta.
        num_steps (int): The number of steps used for gamma and beta.
    """
    print("\n--- Creating Summary Plots ---")
    if not all_results:
        print("No results to create summary plots.")
        return

    gammas = np.array([res['gamma'] for res in all_results])
    betas = np.array([res['beta'] for res in all_results])
    r_squareds = np.array([res['r_squared'] for res in all_results])
    intercepts = np.array([res['intercept'] for res in all_results])
    normalised_intercepts = np.array([res['normalised_intercept'] for res in all_results])

   
    fig_int_2d = plt.figure(figsize=(10, 7))
    ax_int_2d = fig_int_2d.add_subplot(111)
    sc_int_2d = ax_int_2d.scatter(gammas, betas, c=intercepts, cmap='plasma', s=100, alpha=0.8)
    
    positive_mask = intercepts > 0
    if np.any(positive_mask):
        ax_int_2d.scatter(
            gammas[positive_mask],
            betas[positive_mask],
            marker='*',
            color='black',
            s=60,
            label='Positive Intercept'
        )
        ax_int_2d.legend()

    ax_int_2d.set_xlabel('Gamma (γ)')
    ax_int_2d.set_ylabel('Beta (β)')
    ax_int_2d.set_title(f'{input_filename} | 2D Scatter of Intercept vs. Exponents\nStep: {step:.2f}, Num Steps: {num_steps}')
    ax_int_2d.grid(True)
    fig_int_2d.colorbar(sc_int_2d, label='Intercept Value')
    int_2d_plot_filename = os.path.join(folder_path, f"2D_Intercept_Scatter_step_{step:.2f}_nsteps_{num_steps}.png")
    plt.savefig(int_2d_plot_filename)
    print(f"Saved Intercept 2D scatter plot to '{int_2d_plot_filename}'")
    
    
   
    fig_r2_2d = plt.figure(figsize=(10, 7))
    ax_r2_2d = fig_r2_2d.add_subplot(111)
    
   
    sc_r2_2d = ax_r2_2d.scatter(gammas, betas, c=r_squareds, cmap='viridis', s=100, alpha=0.8, vmin=0.95, vmax=1.0)
    
    ax_r2_2d.set_xlabel('Gamma (γ)')
    ax_r2_2d.set_ylabel('Beta (β)')
    ax_r2_2d.set_title(f'{input_filename} | 2D Scatter of R² vs. Exponents\nStep: {step:.2f}, Num Steps: {num_steps}')
    ax_r2_2d.grid(True)
    fig_r2_2d.colorbar(sc_r2_2d, label='R-squared Value')
    r2_2d_plot_filename = os.path.join(folder_path, f"2D_R_Squared_Scatter_step_{step:.2f}_nsteps_{num_steps}.png")
    plt.savefig(r2_2d_plot_filename)
    print(f"Saved R-squared 2D scatter plot to '{r2_2d_plot_filename}'")
    
    
   
    fig_norm_int_2d = plt.figure(figsize=(10, 7))
    ax_norm_int_2d = fig_norm_int_2d.add_subplot(111)
    sc_norm_int_2d = ax_norm_int_2d.scatter(gammas, betas, c=normalised_intercepts, cmap='magma', s=100, alpha=0.8)
    ax_norm_int_2d.set_xlabel('Gamma (γ)')
    ax_norm_int_2d.set_ylabel('Beta (β)')
    ax_norm_int_2d.set_title(f'{input_filename} | 2D Scatter of Norm. Intercept vs. Exponents\nStep: {step:.2f}, Num Steps: {num_steps}')
    ax_norm_int_2d.grid(True)
    fig_norm_int_2d.colorbar(sc_norm_int_2d, label='Normalised Intercept Value')
    norm_int_2d_plot_filename = os.path.join(folder_path, f"2D_Normalised_Intercept_Scatter_step_{step:.2f}_nsteps_{num_steps}.png")
    plt.savefig(norm_int_2d_plot_filename)
    print(f"Saved Normalised Intercept 2D scatter plot to '{norm_int_2d_plot_filename}'")
    

   
    fig_r2_3d = plt.figure(figsize=(10, 7))
    ax_r2_3d = fig_r2_3d.add_subplot(111, projection='3d')
    sc_r2_3d = ax_r2_3d.scatter(gammas, betas, r_squareds, c=r_squareds, cmap='viridis', label='All R² values', vmin=0.8, vmax=1.0)
    g_r2, b_r2, r2_val = best_r_squared['gamma'], best_r_squared['beta'], best_r_squared['r_squared']
    ax_r2_3d.scatter(g_r2, b_r2, r2_val, color='red', s=150, edgecolor='black', depthshade=True, label=f"Highest R² ({r2_val:.4f})")
    ax_r2_3d.text(g_r2, b_r2, r2_val, f'  ({g_r2:.2f}, {b_r2:.2f}, {r2_val:.4f})', color='black', fontsize=12)
    ax_r2_3d.set_xlabel('Gamma (γ)')
    ax_r2_3d.set_ylabel('Beta (β)')
    ax_r2_3d.set_zlabel('R-squared (R²)')
    ax_r2_3d.set_title(f'{input_filename} | R-squared 3D Scatter Plot\nStep: {step:.2f}, Num Steps: {num_steps}')
    ax_r2_3d.legend()
    fig_r2_3d.colorbar(sc_r2_3d, ax=ax_r2_3d, shrink=0.5, aspect=5, label='R-squared Value')
    r2_plot_filename = os.path.join(folder_path, f"3D_R_Squared_Scatter_step_{step:.2f}_nsteps_{num_steps}.png")
    plt.savefig(r2_plot_filename)
    print(f"Saved R-squared 3D plot to '{r2_plot_filename}'")
    
   
    fig_int_3d = plt.figure(figsize=(10, 7))
    ax_int_3d = fig_int_3d.add_subplot(111, projection='3d')
    sc_int_3d = ax_int_3d.scatter(gammas, betas, intercepts, c=intercepts, cmap='plasma', label='All Intercept values')
    
    g_int, b_int, int_val = greatest_norm_intercept['gamma'], greatest_norm_intercept['beta'], greatest_norm_intercept['intercept']
    norm_int_val = greatest_norm_intercept['normalised_intercept']
    ax_int_3d.scatter(g_int, b_int, int_val,
                          color='cyan', s=150, edgecolor='black', depthshade=True, label=f"Greatest Norm. Intercept ({norm_int_val:.4f})")
    ax_int_3d.text(g_int, b_int, int_val, f'  ({g_int:.2f}, {b_int:.2f}, {int_val:.4f})', color='black', fontsize=12)
    
    ax_int_3d.set_xlabel('Gamma (γ)')
    ax_int_3d.set_ylabel('Beta (β)')
    ax_int_3d.set_zlabel('Intercept')
    ax_int_3d.set_title(f'{input_filename} | Intercept 3D Scatter Plot\nStep: {step:.2f}, Num Steps: {num_steps}')
    ax_int_3d.legend()
    fig_int_3d.colorbar(sc_int_3d, ax=ax_int_3d, shrink=0.5, aspect=5, label='Intercept Value')
    int_plot_filename = os.path.join(folder_path, f"3D_Intercept_Scatter_step_{step:.2f}_nsteps_{num_steps}.png")
    plt.savefig(int_plot_filename)
    print(f"Saved Intercept 3D plot to '{int_plot_filename}'")

    print("\n--- Displaying Summary Plots ---")
    plt.show()


def main():
    """Main function to run the entire analysis."""
    
    print("--- Data Loading ---")
    b_field, m_data = read_magnetization_data(input_file_name)
    if b_field is None or m_data is None:
        print("Exiting script due to data loading error.")
        return
    print("Data loaded successfully.")
    print("-" * 20)

    
    print("\n--- Parameter Input ---")
    gamma_values, beta_values, gamma_center, beta_center, hc_value, step, num_steps = get_user_parameters()
    base_name = os.path.splitext(input_file_name)[0]
    folder_name = f"{base_name}_gamma_{gamma_center:.2f}_beta_{beta_center:.2f}_hc_{hc_value:.2f}_step_{step:.2f}_nsteps_{num_steps}"
    os.makedirs(folder_name, exist_ok=True)
    print(f"Created output directory: '{folder_name}'")

    
    all_datasets = create_and_save_scaled_datasets(m_data, b_field, beta_values, gamma_values, folder_name, step, num_steps)
    
    all_results, greatest_norm_intercept, best_r_squared = perform_fitting_and_save_plots(
        all_datasets, hc_value, folder_name, input_file_name, step, num_steps
    )

    if all_results:
        create_summary_plots(
            all_results, greatest_norm_intercept, best_r_squared, folder_name, input_file_name, step, num_steps
        )

    print("\n--- Best Fit Summary ---")
    if greatest_norm_intercept:
        print("Combination with the GREATEST NORMALISED INTERCEPT:")
        print(f"  Gamma: {greatest_norm_intercept['gamma']:.4f}")
        print(f"  Beta: {greatest_norm_intercept['beta']:.4f}")
        print(f"  Normalised Intercept: {greatest_norm_intercept['normalised_intercept']:.9f}")
        print(f"  (Raw Intercept: {greatest_norm_intercept['intercept']:.9f})")
    
    if best_r_squared:
        print("\nCombination with the HIGHEST R-squared (Best Linear Fit):")
        print(f"  Gamma: {best_r_squared['gamma']:.4f}")
        print(f"  Beta: {best_r_squared['beta']:.4f}")
        print(f"  R-squared: {best_r_squared['r_squared']:.9f}")
    
    print("\nScript finished.")


if __name__ == "__main__":
    main()
    
    
    
    