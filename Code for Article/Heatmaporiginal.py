import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import concurrent.futures
import os
from datetime import datetime

# ==========================================
# --- 1. MASTER CONFIGURATION DASHBOARD ---
# ==========================================
class Config:
    # --- DYNAMIC AXIS SETTINGS ---
    # Options: 'B', 'C', 'SIGMA', 'A', 'COST_FACTOR', 'MUTATION_RATE', 'F', 'ALPHA'
    X_AXIS_PARAM = 'B'       
    X_START, X_END = 0.0, 3.5
    X_BINS = 15              

    Y_AXIS_PARAM = 'SIGMA'       
    Y_START, Y_END = 0.1, 3.5   
    Y_BINS = 30              

    REPLICATES = 15          
    
    # --- STATIC BIOLOGICAL PARAMETERS ---
    F = 4.0                 # Max fecundity
    ALPHA = 0.8             # Max base survival probability
    MUTATION_RATE = 0.02    # Genetic mutation (SD)
    
    # --- STATIC PLASTICITY & EPIGENETICS ---
    B = 1.5                 # Habitat Selection strength
    C = 0.5                 # Epigenetic Inheritance fraction
    A = 0.4                 # Plastic reach cap
    COST_FACTOR = 0.25       # Metabolic tax for plasticity
    
    # --- STATIC ENVIRONMENT PARAMETERS ---
    SIGMA = 1.0             # Niche width (Selection strength)
    K_PER_HABITAT = 600     # LOCAL carrying capacity per niche
    HABITAT_OPTIMA = np.array([1.0, 2.0, 3.0]) 

    # --- SIMULATION STEERING ---
    GENERATIONS = 1500      
    INIT_POP_SIZE = 300     
    BURN_IN_PERIOD = 500    
    THRESHOLD_UP = 0.20     
    THRESHOLD_DOWN = 0.04   
    SPECIES_BARRIER = 0.20 

# ==========================================
# --- 2. CORE SIMULATION ENGINE ---
# ==========================================
def run_simulation(current_coords):
    m = len(Config.HABITAT_OPTIMA)
    optima = Config.HABITAT_OPTIMA
    opt_row = optima[np.newaxis, :] 
    
    p = {
        'B': current_coords.get('B', Config.B),
        'C': current_coords.get('C', Config.C),
        'SIGMA': current_coords.get('SIGMA', Config.SIGMA),
        'A': current_coords.get('A', Config.A),
        'COST_FACTOR': current_coords.get('COST_FACTOR', Config.COST_FACTOR),
        'MUTATION_RATE': current_coords.get('MUTATION_RATE', Config.MUTATION_RATE),
        'F': current_coords.get('F', Config.F),
        'ALPHA': current_coords.get('ALPHA', Config.ALPHA)
    }
    
    safe_sigma = max(p['SIGMA'], 1e-9)
    two_sig_sq = 2 * (safe_sigma**2)

    pop_x = np.random.normal(loc=1.0, scale=0.1, size=Config.INIT_POP_SIZE)
    pop_epi = np.zeros(Config.INIT_POP_SIZE)
    pop_hab = np.zeros(Config.INIT_POP_SIZE, dtype=int)

    species_active = [False, False, False]
    species_history = []
    pop_history_last_10 = []

    for gen in range(Config.GENERATIONS):
        n = len(pop_x)
        if n == 0: break 

        # Real survival reality check
        c_opts = optima[pop_hab]
        y_real = np.clip(c_opts, pop_x - p['A'], pop_x + p['A'])
        fit_exponent = -((y_real - c_opts)**2) / two_sig_sq
        P_surv = np.clip(p['ALPHA'] * np.exp(fit_exponent) - np.abs(y_real - (pop_x + pop_epi)) * p['COST_FACTOR'], 0, 1)

        n_hab = np.bincount(pop_hab, minlength=m)[pop_hab]
        dens_fact = np.clip(1.0 - (n_hab / Config.K_PER_HABITAT), 0, 1)
        off_counts = np.random.binomial(int(p['F']), np.clip(P_surv * dens_fact, 0, 1))
        total_off = np.sum(off_counts)
        
        if total_off == 0: 
            pop_x = np.array([])
            break

        # Track population like Doc 1: offspring count in final 10 generations
        if gen >= Config.GENERATIONS - 10:
            pop_history_last_10.append(total_off)

        idx = np.repeat(np.arange(n), off_counts)
        p_x_vals = pop_x[idx]
        off_x = p_x_vals + np.random.normal(0, p['MUTATION_RATE'], total_off)
        off_epi = p['C'] * (y_real[idx] - p_x_vals)

        # --- HABITAT SELECTION (EPIGENETICS + COST INCLUDED) ---
        x_c = off_x[:, np.newaxis]
        epi_c = off_epi[:, np.newaxis]
        
        pred_y = np.clip(opt_row, x_c - p['A'], x_c + p['A'])
        fit_exponent_pred = -((pred_y - opt_row)**2) / two_sig_sq
        pred_P_raw = p['ALPHA'] * np.exp(fit_exponent_pred) - np.abs(pred_y - (x_c + epi_c)) * p['COST_FACTOR']
        pred_P = np.clip(pred_P_raw, 0, 1)
        
        w = np.power(pred_P, p['B'])
        row_s = w.sum(axis=1, keepdims=True)
        w /= np.where(row_s == 0, 1, row_s)
        pop_hab = (np.random.rand(total_off, 1) < w.cumsum(axis=1)).argmax(axis=1)
        pop_x, pop_epi = off_x, off_epi

        # STRICT EVERY-GENERATION SPECIES TRACKING
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

    final_n = np.mean(pop_history_last_10) if pop_history_last_10 else 0
    
    # Check volatility
    post_burn_in = species_history[Config.BURN_IN_PERIOD:] if len(species_history) > Config.BURN_IN_PERIOD else species_history
    clean_history = [s for i, s in enumerate(post_burn_in) if i == 0 or s != post_burn_in[i-1]]
    is_v = not (clean_history == sorted(clean_history))
    
    # DUAL SCORE LOGIC
    if final_n == 0: 
        f_code, spec_score, class_score = "Ext", 0, 0
    elif species_active.count(True) == 3: 
        f_code, spec_score, class_score = "Rad", 2, 3
    elif species_active == [True, False, True]: 
        f_code, spec_score, class_score = "Sq", 2, 2
    elif species_active.count(True) == 2: 
        f_code, spec_score, class_score = "Inc", 2, 2
    elif species_active.count(True) == 1:
        f_code = "Gen" if species_active[1] else "Edg"
        spec_score, class_score = 1, 1
    else: 
        f_code, spec_score, class_score = "Gen", 1, 1

    return spec_score, class_score, is_v, final_n, f_code

# ==========================================
# --- 3. MULTI-CORE PROCESSING ---
# ==========================================
def run_grid_point(args):
    i, j, x_v, y_v, x_p, y_p, reps = args
    t_spec, t_class, t_pop, any_v, outs = 0, 0, 0, False, {}
    coords = {x_p: x_v, y_p: y_v}
    
    for _ in range(reps):
        s_sc, c_sc, v, n, code = run_simulation(coords)
        t_spec += s_sc; t_class += c_sc; t_pop += n 
        if v: any_v = True
        outs[code] = outs.get(code, 0) + 1
        
    avg_s = t_spec / reps
    avg_c = t_class / reps
    avg_p = int(round((t_pop / reps) / 10.0) * 10)
    
    status = 'V' if any_v else ''
    out_keys = sorted(list(outs.keys()))
    out_lines = []
    for k in range(0, len(out_keys), 2):
        pair = out_keys[k:k+2]
        line = " ".join([f"{key}:{outs[key]}" for key in pair])
        out_lines.append(line)
    
    outcome_str = "\n".join(out_lines)
    # Showing Dual Scores in text
    custom_text = f"Sp:{avg_s:.1f} | Cl:{avg_c:.1f}{status}\nP:{avg_p}\n{outcome_str}"
    
    terminal_summary = " ".join([f"{k}:{v}" for k, v in outs.items()])
    print(f"DONE | {x_p}={x_v:.2f}, {y_p}={y_v:.2f} | Pop: {avg_p} | Sp:{avg_s:.1f} | {terminal_summary}")
    
    # We return avg_c to color the heatmap by the Classic Score (1 to 3)
    return (i, j, avg_c, custom_text)

# ==========================================
# --- 4. VISUALIZATION & SAVING ---
# ==========================================
def run_heatmap_experiment():
    x_vals = np.linspace(Config.X_START, Config.X_END, Config.X_BINS)
    y_vals = np.linspace(Config.Y_START, Config.Y_END, Config.Y_BINS)
    
    all_params = {'B': Config.B, 'C': Config.C, 'SIGMA': Config.SIGMA, 'A': Config.A, 'COST': Config.COST_FACTOR, 'MUT': Config.MUTATION_RATE}
    static_labels = [f"{k}={v}" for k, v in all_params.items() if k != Config.X_AXIS_PARAM and k != Config.Y_AXIS_PARAM]
    params_subtitle = " | ".join(static_labels)

    color_m, text_m = np.zeros((len(y_vals), len(x_vals))), np.empty((len(y_vals), len(x_vals)), dtype=object)
    tasks = [(i, j, x, y, Config.X_AXIS_PARAM, Config.Y_AXIS_PARAM, Config.REPLICATES) for i, y in enumerate(y_vals) for j, x in enumerate(x_vals)]

    start = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run_grid_point, tasks))

    for r in results:
        i, j, avg, txt = r
        color_m[i, j], text_m[i, j] = avg, txt

    plt.figure(figsize=(20, 12)) 
    
    sns.heatmap(color_m, annot=text_m, fmt="", cmap="RdYlGn", vmin=1.0, vmax=3.0,
                xticklabels=[f"{v:.2f}" for v in x_vals], yticklabels=[f"{v:.2f}" for v in y_vals], 
                annot_kws={"size": 6.5, "va": "center", "linespacing": 1.2},
                cbar_kws={'label': 'Classic Diversity Score (1 to 3)'})
    
    plt.gca().invert_yaxis() 
    plt.title(f"Radiation Phase Diagram: {Config.X_AXIS_PARAM} vs {Config.Y_AXIS_PARAM}\n{params_subtitle} | K={Config.K_PER_HABITAT}", fontsize=14)
    plt.xlabel(Config.X_AXIS_PARAM); plt.ylabel(Config.Y_AXIS_PARAM)
    plt.tight_layout()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_folder = os.path.join(script_dir, "heatmaps")
    
    try:
        if not os.path.exists(save_folder): os.makedirs(save_folder)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Study_{Config.X_AXIS_PARAM}_vs_{Config.Y_AXIS_PARAM}_{timestamp}.png"
        save_path = os.path.join(save_folder, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSUCCESS: Heatmap saved to: {save_path}")
    except Exception as e:
        print(f"\nCRITICAL ERROR SAVING IMAGE: {e}")
        fallback_name = f"FALLBACK_{Config.X_AXIS_PARAM}_vs_{Config.Y_AXIS_PARAM}.png"
        plt.savefig(fallback_name, dpi=300, bbox_inches='tight')
        print(f"Image saved to current directory as fallback: {fallback_name}")

    plt.show()

if __name__ == "__main__":
    run_heatmap_experiment()