import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

# ==========================================
# --- 1. CONFIGURATION DASHBOARD ---
# ==========================================
class Config:
    # --- SIMULATION TIMELINE ---
    GENERATIONS = 2000      
    INIT_POP_SIZE = 300     
    
    # --- BIOLOGICAL PARAMETERS ---
    F = 4.0                 
    ALPHA = 0.8             
    MUTATION_RATE = 0.02    
    
    # --- PLASTICITY & EPIGENETICS ---
    B = 1.75                 
    C = 0.75                 
    A = 0.4                 
    COST_FACTOR = 0.25       
    
    # --- ENVIRONMENT PARAMETERS ---
    SIGMA = 3.50            
    K_PER_HABITAT = 600     
    HABITAT_OPTIMA = np.array([1.0, 2.0, 3.0]) 

# ==========================================
# --- 2. CORE SIMULATION ENGINE ---
# ==========================================
def run_timeseries_simulation():
    print("Starting simulation... (This may take a few seconds)\n")
    start_time = time.time()
    
    m = len(Config.HABITAT_OPTIMA)
    optima = Config.HABITAT_OPTIMA
    opt_row = optima[np.newaxis, :] 
    
    safe_sigma = max(Config.SIGMA, 1e-9)
    two_sig_sq = 2 * (safe_sigma**2)

    pop_x = np.random.normal(loc=1.0, scale=0.1, size=Config.INIT_POP_SIZE)
    pop_epi = np.zeros(Config.INIT_POP_SIZE)
    pop_hab = np.zeros(Config.INIT_POP_SIZE, dtype=int)

    # DATA TRACKERS
    all_generations = []
    all_x_values = []
    all_ybirth_values = []
    all_yreal_values = []

    for gen in range(Config.GENERATIONS):
        n = len(pop_x)
        
        # --- TERMINAL POPULATION COUNTER ---
        n_hab = np.bincount(pop_hab, minlength=m) if n > 0 else np.zeros(m, dtype=int)
        if gen % 100 == 0:
            print(f"Gen {gen:4d} | Hab 1: {n_hab[0]:4d} | Hab 2: {n_hab[1]:4d} | Hab 3: {n_hab[2]:4d} | Total: {n}")

        if n == 0: 
            print(f"\nPopulation went extinct at generation {gen}!")
            break 

        c_opts = optima[pop_hab]
        
        # Calculate phenotypes
        y_birth = pop_x + pop_epi
        y_real = np.clip(c_opts, pop_x - Config.A, pop_x + Config.A)

        # --- TRACKING ---
        all_generations.append(np.full(n, gen))
        all_x_values.append(pop_x)
        all_ybirth_values.append(y_birth)
        all_yreal_values.append(y_real)

        # 1. Local Density & Survival
        fit_exponent = -((y_real - c_opts)**2) / two_sig_sq
        P_surv = np.clip(Config.ALPHA * np.exp(fit_exponent) - np.abs(y_real - y_birth) * Config.COST_FACTOR, 0, 1)

        dens_fact = np.clip(1.0 - (n_hab[pop_hab] / Config.K_PER_HABITAT), 0, 1)
        
        off_counts = np.random.binomial(int(Config.F), np.clip(P_surv * dens_fact, 0, 1))
        total_off = np.sum(off_counts)
        if total_off == 0: 
            pop_x = np.array([])
            continue

        # 2. Reproduction & Inheritance
        idx = np.repeat(np.arange(n), off_counts)
        p_x_vals = pop_x[idx]
        off_x = p_x_vals + np.random.normal(0, Config.MUTATION_RATE, total_off)
        off_epi = Config.C * (y_real[idx] - p_x_vals)

        # 3. Habitat Selection (Strict Survival Probability)
        x_c = off_x[:, np.newaxis]
        epi_c = off_epi[:, np.newaxis]
        
        pred_y = np.clip(opt_row, x_c - Config.A, x_c + Config.A)
        fit_exponent_pred = -((pred_y - opt_row)**2) / two_sig_sq
        pred_P_raw = Config.ALPHA * np.exp(fit_exponent_pred) - np.abs(pred_y - (x_c + epi_c)) * Config.COST_FACTOR
        pred_P = np.clip(pred_P_raw, 0, 1)
        
        w = np.power(pred_P, Config.B)
        row_s = w.sum(axis=1, keepdims=True)
        w /= np.where(row_s == 0, 1, row_s)
        
        pop_hab = (np.random.rand(total_off, 1) < w.cumsum(axis=1)).argmax(axis=1)
        pop_x, pop_epi = off_x, off_epi

    print(f"\nSimulation finished in {time.time() - start_time:.2f} seconds.")
    
    if len(all_generations) > 0:
        return (np.concatenate(all_generations), 
                np.concatenate(all_x_values), 
                np.concatenate(all_ybirth_values), 
                np.concatenate(all_yreal_values))
    else:
        return np.array([]), np.array([]), np.array([]), np.array([])

# ==========================================
# --- 3. PLOTTING FUNCTION ---
# ==========================================
def plot_evolution(gens, xs, y_births, y_reals):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    if len(gens) == 0:
        print("Nothing to plot. Population died immediately.")
    else:
        # Plot 1: Genotype (x)
        axes[0].plot(gens, xs, 'o', markersize=1.5, alpha=0.03, color='#1f77b4', rasterized=True)
        axes[0].set_ylabel("Genotype (x)", fontsize=12, fontweight='bold', color='#1f77b4')
        
        # Plot 2: Phenotype at Birth (y at birth)
        axes[1].plot(gens, y_births, 'o', markersize=1.5, alpha=0.03, color='#9467bd', rasterized=True) 
        axes[1].set_ylabel("Birth Phenotype\n(x + epi)", fontsize=12, fontweight='bold', color='#9467bd')

        # Plot 3: Expressed Phenotype (y)
        axes[2].plot(gens, y_reals, 'o', markersize=1.5, alpha=0.03, color='#2ca02c', rasterized=True)
        axes[2].set_ylabel("Expressed Phenotype\n(y)", fontsize=12, fontweight='bold', color='#2ca02c')

        for ax in axes:
            for opt in Config.HABITAT_OPTIMA:
                ax.axhline(y=opt, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
            ax.set_ylim(0.5, 3.5)

        axes[-1].set_xlabel("Generation", fontsize=14, fontweight='bold')

    # Title
    title_text = (f"Evolutionary Timeseries: Traits Over Time\n"
                  f"B={Config.B} | C={Config.C} | Sigma={Config.SIGMA} | "
                  f"A={Config.A} | Cost={Config.COST_FACTOR} | K={Config.K_PER_HABITAT}")
    fig.suptitle(title_text, fontsize=16, fontweight='bold')
    
    plt.xlim(0, Config.GENERATIONS)
    plt.tight_layout()

    # --- SAVING LOGIC ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.join(script_dir, "evolution_plots")
    
    if not os.path.exists(save_folder): 
        os.makedirs(save_folder)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Timeseries_Stacked_B{Config.B}_C{Config.C}_S{Config.SIGMA}_A{Config.A}_{timestamp}.png"
    save_path = os.path.join(save_folder, filename)
    
    plt.savefig(save_path, dpi=300)
    print(f"Plot successfully saved to: {save_path}")
    
    plt.show()

# ==========================================
# --- 4. EXECUTION ---
# ==========================================
if __name__ == "__main__":
    gens_data, xs_data, ybirths_data, yreals_data = run_timeseries_simulation()
    plot_evolution(gens_data, xs_data, ybirths_data, yreals_data)