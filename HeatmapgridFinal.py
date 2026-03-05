import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import csv
import concurrent.futures
import gc
from tqdm import tqdm

# ==========================================
# --- 1. DYNAMIC PARAMETER DASHBOARD ---
# ==========================================
class Config:
    # --- A. AXIS MAPPING (Swap these freely!) ---
    HEATMAP_X = 'B'       
    HEATMAP_Y = 'Sigma'       
    
    MACRO_1 = 'C'         # Fastest changing in Excel (Changes every row)
    MACRO_2 = 'Cost'      # Medium changing
    MACRO_3 = 'A'         # Slowest changing

    # --- B. PARAMETER RANGES ---
    RANGES = {
        'Sigma': (0.01, 3.0, 15),
        'Cost':  (0.0, 1.5, 7),
        'A':     (0.1, 1.0, 7),
        'B':     (0.0, 3.5, 12),
        'C':     (0.0, 1.0, 5)
    }

    # --- C. SIMULATION TIMELINE & REPLICATES ---
    GENERATIONS = 1500
    BURN_IN = 500            
    INIT_POP_SIZE = 300
    REPLICATES = 10          
    
    # --- D. CORE BIOLOGICAL PARAMETERS ---
    F = 4.0                  
    ALPHA = 0.8              
    MUTATION_RATE = 0.02     
    K_PER_HABITAT = 600      
    HABITAT_OPTIMA = np.array([1.0, 2.0, 3.0]) 
    
    # --- E. HYSTERESIS & SPECIATION THRESHOLDS ---
    THRESHOLD_UP = 0.20      
    THRESHOLD_DOWN = 0.04    
    SPECIES_BARRIER = 0.2      

# ==========================================
# --- 2. CORE SIMULATION ENGINE ---
# ==========================================
def run_simulation(params):
    B = params['B']
    C = params['C']
    Sigma = params['Sigma']
    A = params['A']
    Cost = params['Cost']

    m = len(Config.HABITAT_OPTIMA)
    optima = Config.HABITAT_OPTIMA
    opt_row = optima[np.newaxis, :] 
    two_sig_sq = 2 * (max(Sigma, 1e-9)**2)

    pop_x = np.random.normal(loc=1.0, scale=0.1, size=Config.INIT_POP_SIZE)
    pop_epi = np.zeros(Config.INIT_POP_SIZE)
    pop_hab = np.zeros(Config.INIT_POP_SIZE, dtype=int)

    species_active = [False, False, False]
    species_history = []
    pop_history_last_10 = []

    for gen in range(Config.GENERATIONS):
        n = len(pop_x)
        if n == 0: break 

        c_opts = optima[pop_hab]
        y_real = np.clip(c_opts, pop_x - A, pop_x + A)
        
        fit_exponent = -((y_real - c_opts)**2) / two_sig_sq
        P_surv = np.clip(Config.ALPHA * np.exp(fit_exponent) - np.abs(y_real - (pop_x + pop_epi)) * Cost, 0, 1)

        n_hab = np.bincount(pop_hab, minlength=m)[pop_hab]
        dens_fact = np.clip(1.0 - (n_hab / Config.K_PER_HABITAT), 0, 1)
        
        off_counts = np.random.binomial(int(Config.F), np.clip(P_surv * dens_fact, 0, 1))
        total_off = np.sum(off_counts)
        if total_off == 0: 
            pop_x = np.array([])
            continue

        idx = np.repeat(np.arange(n), off_counts)
        p_x_vals = pop_x[idx]
        off_x = p_x_vals + np.random.normal(0, Config.MUTATION_RATE, total_off)
        off_epi = C * (y_real[idx] - p_x_vals)

        # "OLD" HABITAT SELECTION SYSTEM (Bounded Rationality)
        x_c = off_x[:, np.newaxis]
        epi_c = off_epi[:, np.newaxis]
        pred_y = np.clip(opt_row, x_c - A, x_c + A)
        fit_exponent_pred = -((pred_y - opt_row)**2) / two_sig_sq
        pred_P_raw = Config.ALPHA * np.exp(fit_exponent_pred) - np.abs(pred_y - (x_c + epi_c)) * Cost
        pred_P = np.clip(pred_P_raw, 0, 1)

        w = np.power(pred_P, B)
        row_s = w.sum(axis=1, keepdims=True)
        w /= np.where(row_s == 0, 1, row_s)
        
        pop_hab = (np.random.rand(total_off, 1) < w.cumsum(axis=1)).argmax(axis=1)
        pop_x, pop_epi = off_x, off_epi

        # STRICT SPECIES BARRIER CHECK (3 Habitats)
        g1 = np.sum(np.abs(pop_x - 1.0) <= Config.SPECIES_BARRIER) / total_off
        g2 = np.sum(np.abs(pop_x - 2.0) <= Config.SPECIES_BARRIER) / total_off
        g3 = np.sum(np.abs(pop_x - 3.0) <= Config.SPECIES_BARRIER) / total_off
        groups = [g1, g2, g3]

        current_species_count = 0
        for i in range(3):
            if not species_active[i] and groups[i] >= Config.THRESHOLD_UP:
                species_active[i] = True
            elif species_active[i] and groups[i] < Config.THRESHOLD_DOWN:
                species_active[i] = False
            if species_active[i]: current_species_count += 1
            
        eff_count = max(1, current_species_count) if total_off > 0 else 0
        species_history.append(eff_count)
        
        if gen >= Config.GENERATIONS - 10: pop_history_last_10.append(total_off)

    final_pop = np.mean(pop_history_last_10) if pop_history_last_10 else 0
    
    post_burn_in = species_history[Config.BURN_IN:] if len(species_history) > Config.BURN_IN else species_history
    clean_history = [s for i, s in enumerate(post_burn_in) if i == 0 or s != post_burn_in[i-1]]
    is_volatile = not (clean_history == sorted(clean_history))
    
    # DETERMINE BOTH SCORES
    if final_pop == 0: 
        typo, spec_score, class_score = "Ext", 0, 0
    elif species_active.count(True) == 3: 
        typo, spec_score, class_score = "Rad", 2, 3
    elif species_active == [True, False, True]: 
        typo, spec_score, class_score = "Sq", 2, 2
    elif species_active.count(True) == 2: 
        typo, spec_score, class_score = "Inc", 2, 2
    elif species_active.count(True) == 1:
        typo = "Gen" if species_active[1] else "Edg"
        spec_score, class_score = 1, 1
    else: 
        typo, spec_score, class_score = "Gen", 1, 1

    return spec_score, class_score, is_volatile, final_pop, typo

# ==========================================
# --- 3. DYNAMIC HEATMAP GENERATOR ---
# ==========================================
def run_heatmap_experiment(macro_dict, save_folder, filename):
    x_var = Config.HEATMAP_X
    y_var = Config.HEATMAP_Y
    
    x_start, x_end, x_bins = Config.RANGES[x_var]
    y_start, y_end, y_bins = Config.RANGES[y_var]
    
    X_vals = np.linspace(x_start, x_end, int(x_bins)) 
    Y_vals = np.linspace(y_start, y_end, int(y_bins)) 
    
    grid_colors = np.zeros((int(y_bins), int(x_bins)))
    cell_data = {} 
    
    total_spec_score = 0 
    total_class_score = 0
    
    for i, y_val in enumerate(Y_vals): 
        for j, x_val in enumerate(X_vals): 
            params = macro_dict.copy()
            params[x_var] = x_val
            params[y_var] = y_val
            
            rep_spec_scores, rep_class_scores, rep_pops = [], [], []
            outcomes = {}
            volatility_flag = False
            
            for _ in range(Config.REPLICATES):
                sp_sc, cl_sc, is_vol, pop, typo = run_simulation(params)
                rep_spec_scores.append(sp_sc)
                rep_class_scores.append(cl_sc)
                rep_pops.append(pop)
                if is_vol: volatility_flag = True
                outcomes[typo] = outcomes.get(typo, 0) + 1
            
            avg_spec = np.mean(rep_spec_scores)
            avg_class = np.mean(rep_class_scores)
            avg_pop = np.mean(rep_pops)
            
            total_spec_score += avg_spec
            total_class_score += avg_class
            
            grid_colors[i, j] = avg_class
            
            out_keys = sorted(list(outcomes.keys()))
            out_lines = []
            for k in range(0, len(out_keys), 2): 
                pair = out_keys[k:k+2]
                out_lines.append(" ".join([f"{key}:{outcomes[key]}" for key in pair]))
            
            small_text = f"P:{int(avg_pop)}\n" + "\n".join(out_lines)
            text_color = "white" if avg_class < 1.5 else "black"
            
            # Heatmap Visual text: Only shows the Classic Diversity Score
            main_title = f"{avg_class:.1f}{'V' if volatility_flag else 'S'}"
            
            cell_data[(i, j)] = {
                'main': main_title,
                'small': small_text,
                'color': text_color
            }

    fig, ax = plt.subplots(figsize=(14, 12))
    dynamic_fs = np.clip(100 / max(x_bins, y_bins), 4, 12) 
    
    sns.heatmap(grid_colors, annot=False, cmap='RdYlGn', vmin=1.0, vmax=3.0, 
                xticklabels=[f"{x:.2f}" for x in X_vals], 
                yticklabels=[f"{y:.2f}" for y in Y_vals], ax=ax,
                cbar_kws={'label': 'Classic Diversity Score (1 to 3)'})
    
    plt.gca().invert_yaxis() 

    for i in range(int(y_bins)):
        for j in range(int(x_bins)):
            d = cell_data[(i, j)]
            ax.text(j + 0.5, i + 0.75, d['main'], ha='center', va='center', 
                    color=d['color'], fontsize=dynamic_fs * 0.9, fontweight='bold')
            ax.text(j + 0.5, i + 0.35, d['small'], ha='center', va='center', 
                    color=d['color'], fontsize=dynamic_fs * 0.8, linespacing=1.2)

    ax.set_xlabel(f"{x_var} Axis")
    ax.set_ylabel(f"{y_var} Axis")
    title_str = f"Diversity Map | {Config.MACRO_1}={macro_dict[Config.MACRO_1]:.2f} | {Config.MACRO_2}={macro_dict[Config.MACRO_2]:.2f} | {Config.MACRO_3}={macro_dict[Config.MACRO_3]:.2f}"
    ax.set_title(title_str, fontweight="bold", fontsize=14)
    
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close('all')
    gc.collect() 
    
    return total_spec_score / (x_bins * y_bins), total_class_score / (x_bins * y_bins)

# ==========================================
# --- 4. BATCH RUNNER ---
# ==========================================
def run_single_sweep(args):
    run_id, total_runs, macro_dict, script_dir = args
    v1 = macro_dict[Config.MACRO_1]
    v2 = macro_dict[Config.MACRO_2]
    v3 = macro_dict[Config.MACRO_3]
    
    folder = os.path.join(script_dir, "Sweep_Results_final", f"{Config.MACRO_3}_{v3:.2f}", f"{Config.MACRO_2}_{v2:.2f}")
    os.makedirs(folder, exist_ok=True)
    
    img_name = f"HM_{Config.MACRO_1}_{v1:.2f}_final.png"
    target_path = os.path.join(folder, img_name)
    hyperlink = f'=HYPERLINK("{target_path}", "View Heatmap")'
    
    if os.path.exists(target_path): return None 
    
    try:
        spec_score, class_score = run_heatmap_experiment(macro_dict, save_folder=folder, filename=img_name)
        return [run_id, round(v1, 2), round(v2, 2), round(v3, 2), spec_score, class_score, hyperlink]
    except Exception as e:
        print(f"\nCRASH on {run_id}: {e}"); return None

def finalize_and_analyze(log_path):
    print("\n[Finalizing] Organizing Master Log and Generating Dashboards...")
    try:
        df = pd.read_csv(log_path)
        for col in [Config.MACRO_1, Config.MACRO_2, Config.MACRO_3]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        sort_order = [Config.MACRO_2, Config.MACRO_3, Config.MACRO_1]
        df = df.sort_values(by=sort_order, ascending=[True, True, True])
        
        spaced_data = []
        last_m2, last_m3 = None, None
        
        for _, row in df.iterrows():
            curr_m2 = round(float(row[Config.MACRO_2]), 4)
            curr_m3 = round(float(row[Config.MACRO_3]), 4)
            
            if last_m2 is not None and (curr_m2 != last_m2 or curr_m3 != last_m3):
                spacer = {col: "" for col in df.columns}
                spaced_data.append(spacer)
            
            spaced_data.append(row.to_dict())
            last_m2, last_m3 = curr_m2, curr_m3
            
        df_final = pd.DataFrame(spaced_data)
        df.to_csv(log_path, index=False) 
        
        excel_path = log_path.replace('.csv', '.xlsx')
        df_final.to_excel(excel_path, index=False, engine='openpyxl')
        
        # --- GENERATE DASHBOARD 1: Speciation Score ---
        df_plot = df.dropna(subset=['Speciation_Score'])
        fig1 = plt.figure(figsize=(12, 10))
        ax1 = fig1.add_subplot(111, projection='3d')
        sc1 = ax1.scatter(df_plot[Config.MACRO_1], df_plot[Config.MACRO_2], df_plot[Config.MACRO_3], 
                          c=df_plot['Speciation_Score'], cmap='viridis', s=50, vmin=1.0, vmax=2.0)
        ax1.set_xlabel(f"Macro 1: {Config.MACRO_1}")
        ax1.set_ylabel(f"Macro 2: {Config.MACRO_2}")
        ax1.set_zlabel(f"Macro 3: {Config.MACRO_3}")
        plt.colorbar(sc1, label='Speciation Score (Max 2)')
        plt.title("Master Sweep: Speciation Rate")
        plt.savefig(os.path.join(os.path.dirname(log_path), "Macro_Dashboard_Speciation_final.png"), dpi=300)
        plt.close(fig1)

        # --- GENERATE DASHBOARD 2: Classic Score ---
        fig2 = plt.figure(figsize=(12, 10))
        ax2 = fig2.add_subplot(111, projection='3d')
        sc2 = ax2.scatter(df_plot[Config.MACRO_1], df_plot[Config.MACRO_2], df_plot[Config.MACRO_3], 
                          c=df_plot['Classic_Score'], cmap='viridis', s=50, vmin=1.0, vmax=3.0)
        ax2.set_xlabel(f"Macro 1: {Config.MACRO_1}")
        ax2.set_ylabel(f"Macro 2: {Config.MACRO_2}")
        ax2.set_zlabel(f"Macro 3: {Config.MACRO_3}")
        plt.colorbar(sc2, label='Classic Score (Max 3)')
        plt.title("Master Sweep: Classic Diversity Rate")
        plt.savefig(os.path.join(os.path.dirname(log_path), "Macro_Dashboard_Classic_final.png"), dpi=300)
        plt.close(fig2)

        print(f"SUCCESS: Data organized. 2 Dashboards Generated. {len(df_final)} rows processed.")
        
    except Exception as e: 
        print(f"CRITICAL ERROR in Finalization: {e}")

# ==========================================
# --- 6. MAIN EXECUTION LOOP ---
# ==========================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    log_path = os.path.join(script_dir, "Master_Sweep_Log_final.csv")
    
    if not os.path.exists(log_path):
        with open(log_path, mode='w', newline='') as f:
            csv.writer(f).writerow(["Run_ID", Config.MACRO_1, Config.MACRO_2, Config.MACRO_3, "Speciation_Score", "Classic_Score", "Image_Link"])

    m1_start, m1_end, m1_bins = Config.RANGES[Config.MACRO_1]
    m2_start, m2_end, m2_bins = Config.RANGES[Config.MACRO_2]
    m3_start, m3_end, m3_bins = Config.RANGES[Config.MACRO_3]

    m1_vals = np.linspace(m1_start, m1_end, int(m1_bins))
    m2_vals = np.linspace(m2_start, m2_end, int(m2_bins))
    m3_vals = np.linspace(m3_start, m3_end, int(m3_bins))

    tasks = []
    run_count = 0
    total_runs = int(m1_bins * m2_bins * m3_bins)
    
    for v3 in m3_vals:
        for v2 in m2_vals:
            for v1 in m1_vals:
                run_count += 1
                macro_dict = {Config.MACRO_1: v1, Config.MACRO_2: v2, Config.MACRO_3: v3}
                tasks.append((run_count, total_runs, macro_dict, script_dir))

    with open(log_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(run_single_sweep, t): t for t in tasks}
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Sweeping"):
                res = fut.result()
                if res: 
                    writer.writerow(res)
                    file.flush()
    
    finalize_and_analyze(log_path)