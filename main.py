    """ Edit code below """ 
        # =========================================================================
    # 0. ROBUST HARDWARE PARSING & HELPER FUNCTIONS
    # =========================================================================
    from collections import defaultdict
    import csv
    import math
    import logging
    import sys
    import orjson
    import re
    from pathlib import Path
    from datetime import datetime

    # Hardware path inference
    hardware_path = None
    try:
        if len(sys.argv) > 1:
            workspace_name = next((arg for arg in sys.argv[1:] if not arg.startswith("-")), None)
            if workspace_name:
                c1 = Path(f"workspaces/manual/{workspace_name}/hardware.json")
                c2 = Path(f"{workspace_name}/hardware.json")
                if c1.exists(): hardware_path = c1
                elif c2.exists(): hardware_path = c2
    except Exception as e:
        logging.warning(f"   [HW] Path inference error: {e}")

    if not hardware_path or not hardware_path.exists():
        hardware_path = Path("workspaces/manual/TP_AS25/hardware.json")

    # Parse hardware
    nodes_data = {}
    grid_config = {}

    if hardware_path and hardware_path.exists():
        try:
            with open(hardware_path, "rb") as f:
                hw_cfg = orjson.loads(f.read())
                grid_config = hw_cfg.get("grid_config", {})
                nodes_raw = hw_cfg.get("component_mapping") or hw_cfg.get("nodes") or hw_cfg.get("routers", {})
                if isinstance(hw_cfg, list): nodes_raw = hw_cfg
                
                if isinstance(nodes_raw, list):
                    for item in nodes_raw:
                        nid = item.get("id", item.get("router_id"))
                        if nid is not None: nodes_data[int(nid)] = item
                elif isinstance(nodes_raw, dict):
                    for k, v in nodes_raw.items(): nodes_data[int(k)] = v
                
                logging.info(f"   [HW] Parsed {len(nodes_data)} nodes from {hardware_path.name}")
        except Exception as e:
            logging.error(f"   [ERROR] JSON Read Failed: {e}")

    # Build pools
    CORE_POOL = []
    DRAM_POOL = []
    ROUTER_TYPES = {}

    if not nodes_data:
        logging.warning("   [HW] WARNING: Using Fallback Topology")
        GRID_W_EST = 8
        all_router_ids = list(range(48))
        for rid in all_router_ids:
            col = rid % GRID_W_EST
            if col == 0 or col == (GRID_W_EST - 1):
                nodes_data[rid] = {"type": "dram"}
            else:
                nodes_data[rid] = {"type": "core"}
    else:
        all_router_ids = sorted(nodes_data.keys())

    for rid in all_router_ids:
        info = nodes_data[rid]
        rtype = info.get("type", "core").lower()
        is_dram = any(x in rtype for x in ["dram", "memory", "mc", "hbm"])
        
        if is_dram:
            DRAM_POOL.append(rid)
            ROUTER_TYPES[rid] = "dram"
        else:
            CORE_POOL.append(rid)
            ROUTER_TYPES[rid] = "core"

    DRAM_POOL.sort()
    CORE_POOL.sort()

    # Grid dimensions
    if "grid_x" in grid_config and "grid_y" in grid_config:
        GRID_WIDTH = int(grid_config["grid_x"])
        GRID_HEIGHT = int(grid_config["grid_y"])
    else:
        total_routers = len(all_router_ids)
        side = int(math.sqrt(total_routers)) if total_routers > 0 else 1
        GRID_WIDTH = side
        GRID_HEIGHT = math.ceil(total_routers / side)

    # Router positions
    router_positions = {}
    for rid in all_router_ids:
        node_info = nodes_data.get(rid, {})
        if "x" in node_info and "y" in node_info:
            router_positions[rid] = (int(node_info["x"]), int(node_info["y"]))
        else:
            router_positions[rid] = (rid % GRID_WIDTH, rid // GRID_WIDTH)

    MIN_CORE_RATIO = 0.03
    MIN_CORES_PER_TASK = max(1, int(len(CORE_POOL) * MIN_CORE_RATIO))

    logging.info(f"   [HW] Grid: {GRID_WIDTH}x{GRID_HEIGHT}, Cores: {len(CORE_POOL)}, DRAMs: {len(DRAM_POOL)}")

    def validate_router_type(router_id, expected_type):
        return ROUTER_TYPES.get(router_id, "unknown") == expected_type

    def validate_router_list(router_ids, expected_type):
        return [r for r in router_ids if validate_router_type(r, expected_type)]

    def get_pos(rid):
        return router_positions.get(rid, (0, 0))

    def get_dist(r1, r2):
        p1, p2 = get_pos(r1), get_pos(r2)
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def safe_float(value, default=0.0):
        if value is None or value == '' or str(value).upper() in ['NA', 'NAN', 'NONE']:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def update_partition_dims(node, node_name_hint=None):
        rids = node.get("router_ids", [])
        count = len(rids)
        if count == 0: 
            node["partition_dims"] = [1, 1, 1, 1]
            return
        
        old_dims = node.get("partition_dims", [1, 1, 1, 1])
        new_dims = [1] * len(old_dims)
        
        if len(old_dims) == 4:
            if len(new_dims) > 2: new_dims[2] = count 
        elif len(old_dims) == 3:
            if len(new_dims) > 1: new_dims[1] = count
        else:
            target_indices = [i for i, d in enumerate(old_dims) if d > 1]
            if not target_indices: target_indices = [len(old_dims)-1]
            if len(target_indices) == 1:
                new_dims[target_indices[0]] = count
            else:
                idx1, idx2 = target_indices[-2], target_indices[-1]
                xs = [get_pos(r)[0] for r in rids]
                ys = [get_pos(r)[1] for r in rids]
                phys_ratio = 1.0
                if xs and ys:
                    w = max(xs) - min(xs) + 1
                    h = max(ys) - min(ys) + 1
                    phys_ratio = h / w if w > 0 else 1.0
                best_d1, best_d2 = 1, count
                min_diff = float('inf')
                for i in range(1, int(math.sqrt(count)) + 1):
                    if count % i == 0:
                        d1, d2 = i, count // i
                        if abs((d1/d2) - phys_ratio) < min_diff:
                            min_diff = abs((d1/d2) - phys_ratio)
                            best_d1, best_d2 = d1, d2
                        if abs((d2/d1) - phys_ratio) < min_diff:
                            min_diff = abs((d2/d1) - phys_ratio)
                            best_d1, best_d2 = d2, d1
                new_dims[idx1] = best_d1
                new_dims[idx2] = best_d2
        node["partition_dims"] = new_dims

    def set_node_config(node, router_ids, node_name=None, expected_type="core"):
        unique_ids = []
        if router_ids:
            unique_ids = sorted(list(set(int(r) for r in router_ids)))
            invalid_ids = [r for r in unique_ids if not validate_router_type(r, expected_type)]
            if invalid_ids:
                logging.warning(f"   [WARN] {node_name} invalid routers: {invalid_ids}")
                unique_ids = [r for r in unique_ids if r not in invalid_ids]
        node["router_ids"] = unique_ids
        update_partition_dims(node, node_name)

    def get_counts_from_cfg(cfg_obj, group_key, nodes):
        if not cfg_obj: return {}
        grp = cfg_obj.get("partitions", {}).get(group_key, {}).get("compute_nodes", {})
        return {n: len(grp.get(n, {}).get("router_ids", [])) or 1 for n in nodes}

    # =========================================================================
    # Data Loading Functions
    # =========================================================================
    def load_iter_data(target_iter, ref_dir):
        if target_iter < 1 or not ref_dir: return None, {}, {}
        results_root = ref_dir.parent
        target_dir = results_root / f"iter_{target_iter:03d}"
        if not target_dir.exists(): return None, {}, {}
            
        t_cfg = None
        try:
            with open(target_dir / "config.json", "rb") as f: t_cfg = orjson.loads(f.read())
        except: pass
        
        t_metrics = {}
        t_link_effs = {}
        
        logs_dir = target_dir / "logs"
        if logs_dir.exists():
            for group_dir in logs_dir.glob("layer_group_*"):
                g_key = group_dir.name
                alloc_csv = group_dir / "allocation.csv"
                exec_csv = group_dir / "execution_time.csv"
                link_csv = group_dir / "link_load.csv"
                
                group_data = defaultdict(lambda: {
                    "macs": 0, "time": 0.0, "comp_t": 0.0, 
                    "mem_t": 0.0, "store_t": 0.0, "load_t": 0.0, "routers": set()
                })
                
                core_to_node = {}
                if alloc_csv.exists():
                    try:
                        with open(alloc_csv, "r") as f:
                            for row in csv.DictReader(f):
                                c = row.get("component_name")
                                n = row.get("compute_node")
                                r = row.get("router_id")
                                if c and n: 
                                    core_to_node[c] = n
                                    group_data[n]["macs"] += safe_float(row.get("compute_mac_count"))
                                    if r: group_data[n]["routers"].add(int(r))
                    except: pass
                
                if exec_csv.exists():
                    try:
                        with open(exec_csv, "r") as f:
                            for row in csv.DictReader(f):
                                c = row.get("component_name")
                                n = core_to_node.get(c)
                                if n:
                                    t_load = safe_float(row.get("load_complete_ns"))
                                    t_comp = safe_float(row.get("compute_complete_ns"))
                                    t_store = safe_float(row.get("store_complete_ns"))
                                    t_total = t_load + t_comp + t_store
                                    group_data[n]["time"] = max(group_data[n]["time"], t_total)
                                    group_data[n]["comp_t"] += t_comp
                                    group_data[n]["mem_t"] += (t_load + t_store)
                                    group_data[n]["store_t"] += t_store
                                    group_data[n]["load_t"] += t_load
                    except: pass
                
                # Link efficiency parsing
                router_effs = defaultdict(list)
                if link_csv.exists():
                    try:
                        with open(link_csv, "r") as f:
                            for row in csv.DictReader(f):
                                link = row.get("link", "")
                                bw_eff = safe_float(row.get("bandwidth_eff(GB/s)"))
                                r_ids = re.findall(r"Router(\d+)", link)
                                for rid in r_ids:
                                    router_effs[int(rid)].append(bw_eff)
                    except: pass
                
                avg_router_eff = {}
                for r, effs in router_effs.items():
                    avg_router_eff[r] = sum(effs)/len(effs) if effs else 0.0
                
                t_metrics[g_key] = group_data
                t_link_effs[g_key] = avg_router_eff
                
        return t_cfg, t_metrics, t_link_effs

    def get_mixed_best_config(start_iter, end_iter, ref_dir):
        """Get best partition/placement for each group across iteration range"""
        best_map = {} 
        results_root = ref_dir.parent
        
        for i in range(start_iter, end_iter + 1):
            i_dir = results_root / f"iter_{i:03d}"
            res_file = i_dir / "simulation_result.json"
            cfg_file = i_dir / "config.json"
            if not (res_file.exists() and cfg_file.exists()): continue
            
            try:
                with open(res_file, "rb") as f: 
                    groups_res = orjson.loads(f.read()).get("data", {}).get("groups", [])
                with open(cfg_file, "rb") as f: 
                    partitions_cfg = orjson.loads(f.read()).get("partitions", {})
                
                for g_res in groups_res:
                    g_suffix = g_res.get("group_id")
                    g_key = f"layer_group_{g_suffix}"
                    if g_key not in partitions_cfg: continue
                    time_val = g_res.get("simulation_time_ns", float('inf'))
                    
                    if g_key not in best_map or time_val < best_map[g_key][0]:
                        best_map[g_key] = (time_val, partitions_cfg[g_key], i)
            except: pass
        
        for k, v in best_map.items():
            logging.info(f"   [BEST] {k}: iter_{v[2]} ({v[0]:.0f}ns)")
        
        return {k: v[1] for k, v in best_map.items()}

    def get_best_performance(start_iter, end_iter, ref_dir):
        """Get overall best performance across iteration range"""
        best_time = float('inf')
        best_iter = None
        results_root = ref_dir.parent
        
        for i in range(start_iter, end_iter + 1):
            i_dir = results_root / f"iter_{i:03d}"
            res_file = i_dir / "simulation_result.json"
            if not res_file.exists(): continue
            
            try:
                with open(res_file, "rb") as f: 
                    data = orjson.loads(f.read())
                    time_val = data.get("data", {}).get("summary", {}).get("total_simulation_time_ns", float('inf'))
                    if time_val < best_time:
                        best_time = time_val
                        best_iter = i
            except: pass
        
        return best_time, best_iter

    # =========================================================================
    # Placement Strategies
    # =========================================================================
    def get_placement(strategy, count, available_pool, node_name="", use_name_heuristic=False):
        if count <= 0: return []
        valid_pool = validate_router_list(available_pool, "core")
        if count > len(valid_pool): 
            logging.warning(f"   [WARN] {node_name}: Requested {count} but only {len(valid_pool)} available")
            return valid_pool
        
        pool_coords = [(rid, get_pos(rid)[0], get_pos(rid)[1]) for rid in valid_pool]
        
        # NAME-BASED HEURISTIC
        if use_name_heuristic:
            name_lower = node_name.lower() if node_name else ""
            
            is_dram_group = any(k in name_lower for k in ["matmul", "v_proj"])
            is_ff_group = "ff" in name_lower
            is_q_proj = "q_proj" in name_lower
            is_k_proj = "k_proj" in name_lower
            
            q_col = GRID_WIDTH // 2 - 1 if GRID_WIDTH > 3 else 0
            k_col = GRID_WIDTH // 2 if GRID_WIDTH > 4 else 1
            
            if is_dram_group:
                pool_coords.sort(key=lambda k: (min(k[1], GRID_WIDTH-1-k[1]), k[2]))
            elif is_ff_group:
                mid_row = (GRID_HEIGHT - 1) / 2.0
                pool_coords.sort(key=lambda k: (abs(k[2] - mid_row), k[1]))
            elif is_q_proj:
                pool_coords.sort(key=lambda k: (abs(k[1] - q_col), k[2]))
            elif is_k_proj:
                pool_coords.sort(key=lambda k: (abs(k[1] - k_col), k[2]))
            else:
                if strategy == 'H': 
                    pool_coords.sort(key=lambda k: (k[2], k[1]))
                elif strategy == 'V': 
                    pool_coords.sort(key=lambda k: (k[1], k[2]))
                elif strategy == 'S': 
                    cx = sum(p[1] for p in pool_coords)/len(pool_coords)
                    cy = sum(p[2] for p in pool_coords)/len(pool_coords)
                    pool_coords.sort(key=lambda k: (k[1]-cx)**2 + (k[2]-cy)**2)
            
            if count > 6 and is_dram_group:
                sorted_all_ids = [x[0] for x in pool_coords]
                mid_x = GRID_WIDTH // 2
                l_ids = [pid for pid in sorted_all_ids if get_pos(pid)[0] < mid_x]
                r_ids = [pid for pid in sorted_all_ids if get_pos(pid)[0] >= mid_x]
                half = count // 2
                selected = l_ids[:half] + r_ids[:count - half]
                if len(selected) < count:
                    remain_set = set(selected)
                    for pid in sorted_all_ids:
                        if pid not in remain_set:
                            selected.append(pid)
                            if len(selected) == count: break
                return selected
        
        else:
            if strategy == 'H': 
                pool_coords.sort(key=lambda k: (k[2], k[1]))
            elif strategy == 'V': 
                pool_coords.sort(key=lambda k: (k[1], k[2]))
            elif strategy == 'S': 
                cx = sum(p[1] for p in pool_coords)/len(pool_coords)
                cy = sum(p[2] for p in pool_coords)/len(pool_coords)
                pool_coords.sort(key=lambda k: (k[1]-cx)**2 + (k[2]-cy)**2)
        
        return [x[0] for x in pool_coords[:count]]

    def get_bbox_str(router_ids):
        if not router_ids: return "None"
        xs = [get_pos(r)[0] for r in router_ids]
        ys = [get_pos(r)[1] for r in router_ids]
        if not xs: return "None"
        return f"{max(xs)-min(xs)+1}x{max(ys)-min(ys)+1}"

    # =========================================================================
    # MAIN OPTIMIZATION LOGIC
    # =========================================================================

    meta = config.setdefault("_simple_dse", {})
    cfg_n1, metrics_n1, link_effs_n1 = load_iter_data(iteration - 1, last_iter_dir)

    # Determine total iterations
    total_iters = float('inf')
    if last_iter_dir:
        try:
            with open(last_iter_dir.parent.parent / "run_meta.json", "rb") as f:
                total_iters = orjson.loads(f.read()).get("iterations", float('inf'))
        except: pass

    is_final_iter = (iteration == total_iters) and (iteration > 1)

    # Track stagnation & re-exploration state
    stagnation_count = meta.get("stagnation_count", 0)
    reexplore_start_iter = meta.get("reexplore_start_iter", None)
    reexplore_count = meta.get("reexplore_count", 0)
    bw_phase_best_time = meta.get("bw_phase_best_time", float('inf'))
    
    # ★ Store Iter1/Iter2 counts for delta-based re-exploration
    iter1_counts = meta.get("iter1_counts", {})  # {group_key: {node: count}}
    iter2_counts = meta.get("iter2_counts", {})  # {group_key: {node: count}}
    
    # Check current performance (BW phase only)
    if iteration >= 9 and last_iter_dir and reexplore_start_iter is None:
        current_time, _ = get_best_performance(iteration - 1, iteration - 1, last_iter_dir)
        if current_time < bw_phase_best_time * 0.995:
            bw_phase_best_time = current_time
            stagnation_count = 0
            meta["bw_phase_best_time"] = bw_phase_best_time
        else:
            stagnation_count += 1
        meta["stagnation_count"] = stagnation_count

    logging.info(f"\n{'='*80}")
    logging.info(f"   ITERATION {iteration} - DSE Configuration")
    if reexplore_start_iter is None:
        logging.info(f"   Stagnation: {stagnation_count}/3, BW Best: {bw_phase_best_time:.0f}ns")
    else:
        reexplore_phase = iteration - reexplore_start_iter
        grad_type = "REVERSE" if (reexplore_count % 2 == 1) else "AGGRESSIVE"
        logging.info(f"   RE-EXPLORATION #{reexplore_count} Phase {reexplore_phase} ({grad_type})")
    logging.info(f"{'='*80}")

    # =====================================================================
    # FINAL ITERATION: BEST-OF-BEST ASSEMBLY
    # =====================================================================
    if is_final_iter:
        logging.info(f"=== [FINAL PHASE] ASSEMBLING BEST CONFIGURATIONS ===")
        mixed_best = get_mixed_best_config(1, iteration - 1, last_iter_dir)
        if mixed_best:
            logging.info(f"   -> Assembled Best-of-Best config from all {iteration-1} iterations")
            for gk, gv in mixed_best.items():
                if allowed_groups and gk.split("_")[-1] not in allowed_groups: continue
                config["partitions"][gk] = gv
            
            if "history" not in meta:
                meta["history"] = []
            meta["history"].append({
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "phase": "FINAL_BEST_ASSEMBLY"
            })
            return config

    # Load best configs from previous phases
    best_partitions_iter14 = {}
    best_combined_iter17 = {}
    best_combined_iter18 = {}

    if iteration >= 5 and last_iter_dir:
        best_partitions_iter14 = get_mixed_best_config(1, 4, last_iter_dir)

    if iteration >= 8 and last_iter_dir:
        best_combined_iter17 = get_mixed_best_config(1, 7, last_iter_dir)

    if iteration >= 9 and last_iter_dir:
        best_combined_iter18 = get_mixed_best_config(1, 8, last_iter_dir)

    for group_key, group in partitions.items():
        g_id = group_key.split("_")[-1] if "_" in group_key else group_key
        if allowed_groups is not None and g_id not in allowed_groups: continue
        
        c_nodes = group.get("compute_nodes", {})
        t_nodes = group.get("tensor_nodes", {})
        
        if t_nodes:
            for t_name, t_node in t_nodes.items(): 
                set_node_config(t_node, DRAM_POOL, t_name, expected_type="dram")
        
        if not c_nodes: continue
        
        node_names = sorted(list(c_nodes.keys()))
        debug_info = {}
        global_used_routers = set()
        target_weights = {}
        target_counts = {}
        base_strat = 'V'
        use_name_heuristic = False
        
        g_metrics = metrics_n1.get(group_key, {})
        g_links = link_effs_n1.get(group_key, {})
        
        # =====================================================================
        # ADAPTIVE RE-EXPLORATION: DELTA-BASED
        # =====================================================================
        if iteration >= 9 and stagnation_count >= 3:
            if reexplore_start_iter is None:
                reexplore_start_iter = iteration
                reexplore_count += 1
                meta["reexplore_start_iter"] = reexplore_start_iter
                meta["reexplore_count"] = reexplore_count
                logging.info(f"   [{group_key}] !!! STAGNATION #{reexplore_count} - DELTA-BASED RE-EXPLORATION !!!")
            
            reexplore_phase = iteration - reexplore_start_iter
            is_reverse = (reexplore_count % 2 == 1)
            
            # Get Iter1/Iter2 counts
            iter1_grp = iter1_counts.get(group_key, {})
            iter2_grp = iter2_counts.get(group_key, {})
            
            if not iter1_grp or not iter2_grp:
                logging.warning(f"   [{group_key}] No baseline counts - fallback to equal")
                for n in node_names: target_counts[n] = len(CORE_POOL) // len(node_names)
            else:
                # ★ Delta-based allocation
                for n in node_names:
                    c1 = iter1_grp.get(n, MIN_CORES_PER_TASK)
                    c2 = iter2_grp.get(n, MIN_CORES_PER_TASK)
                    delta = c2 - c1
                    
                    if is_reverse:
                        # REVERSE: 늘어난 만큼 빼고, 줄어든 만큼 늘림
                        target_counts[n] = max(MIN_CORES_PER_TASK, c1 - delta)
                    else:
                        # AGGRESSIVE: 늘어난 만큼 더 늘리고, 줄어든 만큼 더 줄임
                        target_counts[n] = max(MIN_CORES_PER_TASK, c1 + delta)
                
                # Renormalize to total cores
                total_assigned = sum(target_counts.values())
                total_cores = len(CORE_POOL)
                if total_assigned != total_cores:
                    scale = total_cores / total_assigned if total_assigned > 0 else 1
                    for n in node_names:
                        target_counts[n] = max(MIN_CORES_PER_TASK, int(target_counts[n] * scale))
                    
                    # Adjust remainder
                    diff = total_cores - sum(target_counts.values())
                    if diff > 0:
                        for n in sorted(node_names, key=lambda x: target_counts[x], reverse=True):
                            if diff <= 0: break
                            target_counts[n] += 1
                            diff -= 1
                    elif diff < 0:
                        for n in sorted(node_names, key=lambda x: target_counts[x]):
                            if diff >= 0: break
                            if target_counts[n] > MIN_CORES_PER_TASK:
                                target_counts[n] -= 1
                                diff += 1
                
                grad_label = "REVERSE" if is_reverse else "AGGRESSIVE"
                logging.info(f"   [{group_key}] {grad_label} DELTA:")
                for n in node_names:
                    c1 = iter1_grp.get(n, 0)
                    c2 = iter2_grp.get(n, 0)
                    delta = c2 - c1
                    logging.info(f"      {n}: Iter1={c1}, Iter2={c2}, Δ={delta:+d} → Target={target_counts[n]}")
            
            # After 3 iterations, return to BW
            if reexplore_phase >= 2:
                logging.info(f"   [{group_key}] RE-EXPLORE COMPLETE - RETURNING TO BW")
                
                reexplore_best = get_mixed_best_config(reexplore_start_iter, iteration, last_iter_dir)
                
                stagnation_count = 0
                reexplore_start_iter = None
                meta["stagnation_count"] = 0
                meta["reexplore_start_iter"] = None
                
                source_combined = reexplore_best.get(group_key, {}).get("compute_nodes", {}) if reexplore_best else c_nodes
                
                for n in node_names:
                    best_rids = source_combined.get(n, {}).get("router_ids", [])
                    if best_rids:
                        cnt = len(best_rids)
                        target_counts[n] = cnt
                        set_node_config(c_nodes[n], best_rids, n, expected_type="core")
                        for r in best_rids: global_used_routers.add(r)
                    else:
                        target_counts[n] = MIN_CORES_PER_TASK
                
                all_assigned_cores = list(global_used_routers)
                num_to_swap = max(1, int(len(all_assigned_cores) * 0.1))
                
                router_to_node = {}
                for n in node_names:
                    for r in c_nodes[n].get("router_ids", []):
                        router_to_node[r] = n
                
                router_effs = [(r, g_links.get(r, 0.0)) for r in all_assigned_cores]
                router_effs.sort(key=lambda x: x[1])
                
                worst_routers = [r for r, _ in router_effs[:num_to_swap]]
                best_routers = [r for r, _ in router_effs[-num_to_swap:]]
                
                swapped_count = 0
                for w_router, b_router in zip(worst_routers, best_routers):
                    w_node = router_to_node.get(w_router)
                    b_node = router_to_node.get(b_router)
                    
                    if w_node and b_node and w_node != b_node:
                        w_rids = list(c_nodes[w_node].get("router_ids", []))
                        b_rids = list(c_nodes[b_node].get("router_ids", []))
                        
                        if w_router in w_rids and b_router in b_rids:
                            w_rids[w_rids.index(w_router)] = b_router
                            b_rids[b_rids.index(b_router)] = w_router
                            
                            set_node_config(c_nodes[w_node], w_rids, w_node, expected_type="core")
                            set_node_config(c_nodes[b_node], b_rids, b_node, expected_type="core")
                            swapped_count += 1
                
                logging.info(f"   [BW-SWAP] Swapped {swapped_count} pairs\n")
                continue
        
        # =====================================================================
        # NORMAL PHASES
        # =====================================================================
        elif iteration == 1:
            logging.info(f"   [{group_key}] Phase 1: BASELINE")
            use_name_heuristic = True
            for n in node_names: target_weights[n] = 1.0
            
            nodes_to_alloc = node_names
            available_cores = sorted(list(set(CORE_POOL)))
            total_w = sum(target_weights.values()) or 1
            remaining = len(available_cores) - len(nodes_to_alloc) * MIN_CORES_PER_TASK
            for n in nodes_to_alloc: target_counts[n] = MIN_CORES_PER_TASK
            
            if remaining > 0:
                remainders = []
                for n in nodes_to_alloc:
                    share = (target_weights[n] / total_w) * remaining
                    whole = int(share)
                    target_counts[n] += whole
                    remainders.append((share - whole, n))
                remainders.sort(key=lambda x: x[0], reverse=True)
                extra = remaining - sum(int((target_weights[n]/total_w)*remaining) for n in nodes_to_alloc)
                for i in range(extra):
                    if i < len(remainders):
                        target_counts[remainders[i][1]] += 1
            
            # ★ Save Iter1 counts
            if group_key not in iter1_counts:
                iter1_counts[group_key] = {}
            for n in node_names:
                iter1_counts[group_key][n] = target_counts[n]
            meta["iter1_counts"] = iter1_counts
        
        elif iteration == 2:
            logging.info(f"   [{group_key}] Phase 1: LOAD+COMP WEIGHTED")
            use_name_heuristic = True
            
            for n in node_names:
                load_time = g_metrics.get(n, {}).get("load_t", 0.0)
                comp_time = g_metrics.get(n, {}).get("comp_t", 0.0)
                target_weights[n] = load_time + comp_time
                if target_weights[n] <= 0: target_weights[n] = 1.0
            
            nodes_to_alloc = node_names
            available_cores = sorted(list(set(CORE_POOL)))
            total_w = sum(target_weights.values()) or 1
            remaining = len(available_cores) - len(nodes_to_alloc) * MIN_CORES_PER_TASK
            for n in nodes_to_alloc: target_counts[n] = MIN_CORES_PER_TASK
            
            if remaining > 0:
                remainders = []
                for n in nodes_to_alloc:
                    share = (target_weights[n] / total_w) * remaining
                    whole = int(share)
                    target_counts[n] += whole
                    remainders.append((share - whole, n))
                remainders.sort(key=lambda x: x[0], reverse=True)
                extra = remaining - sum(int((target_weights[n]/total_w)*remaining) for n in nodes_to_alloc)
                for i in range(extra):
                    if i < len(remainders):
                        target_counts[remainders[i][1]] += 1
            
            # matmul & v_proj coupling
            matmul_n = next((n for n in node_names if "matmul" in n.lower()), None)
            v_proj_n = next((n for n in node_names if "v_proj" in n.lower()), None)
            
            if matmul_n and v_proj_n and matmul_n != v_proj_n:
                coupled_cores = max(target_counts[matmul_n], target_counts[v_proj_n])
                deficit = coupled_cores * 2 - (target_counts[matmul_n] + target_counts[v_proj_n])
                
                if deficit > 0:
                    target_counts[matmul_n] = coupled_cores
                    target_counts[v_proj_n] = coupled_cores
                    others = [n for n in node_names if n not in [matmul_n, v_proj_n]]
                    if others:
                        steal_per = deficit // len(others)
                        steal_rem = deficit % len(others)
                        for i, o in enumerate(others):
                            steal = steal_per + (1 if i < steal_rem else 0)
                            target_counts[o] = max(MIN_CORES_PER_TASK, target_counts[o] - steal)
            
            # ★ Save Iter2 counts
            if group_key not in iter2_counts:
                iter2_counts[group_key] = {}
            for n in node_names:
                iter2_counts[group_key][n] = target_counts[n]
            meta["iter2_counts"] = iter2_counts
        
        elif iteration <= 4:
            logging.info(f"   [{group_key}] Phase 2: GRADIENT PARTITIONING")
            
            prev_counts = get_counts_from_cfg(cfg_n1, group_key, node_names)
            for n in node_names:
                target_counts[n] = prev_counts.get(n, MIN_CORES_PER_TASK)
        
        elif iteration <= 7:
            logging.info(f"   [{group_key}] Phase 3: PLACEMENT EXPLORATION")
            
            source_cfg_nodes = best_partitions_iter14.get(group_key, {}).get("compute_nodes", {}) if best_partitions_iter14 else c_nodes
            
            for n in node_names:
                cnt = len(source_cfg_nodes.get(n, {}).get("router_ids", [])) or MIN_CORES_PER_TASK
                target_counts[n] = cnt
            
            strategies = ['V', 'H', 'S']
            base_strat = strategies[(iteration - 5) % len(strategies)]
        
        elif iteration == 8:
            logging.info(f"   [{group_key}] Phase 4a: BEST ASSEMBLY")
            
            source_combined = best_combined_iter17.get(group_key, {}).get("compute_nodes", {}) if best_combined_iter17 else c_nodes
            
            for n in node_names:
                best_rids = source_combined.get(n, {}).get("router_ids", [])
                if best_rids:
                    cnt = len(best_rids)
                    target_counts[n] = cnt
                    set_node_config(c_nodes[n], best_rids, n, expected_type="core")
                    for r in best_rids: global_used_routers.add(r)
                else:
                    target_counts[n] = MIN_CORES_PER_TASK
            continue
        
        else:  # iteration >= 9 (normal BW optimization)
            logging.info(f"   [{group_key}] Phase 4b: BW OPTIMIZATION (10% Swap)")
            
            source_combined = best_combined_iter18.get(group_key, {}).get("compute_nodes", {}) if best_combined_iter18 else c_nodes
            
            for n in node_names:
                best_rids = source_combined.get(n, {}).get("router_ids", [])
                if best_rids:
                    cnt = len(best_rids)
                    target_counts[n] = cnt
                    set_node_config(c_nodes[n], best_rids, n, expected_type="core")
                    for r in best_rids: global_used_routers.add(r)
                else:
                    target_counts[n] = MIN_CORES_PER_TASK
            
            all_assigned_cores = list(global_used_routers)
            num_to_swap = max(1, int(len(all_assigned_cores) * 0.1))
            
            router_to_node = {}
            for n in node_names:
                for r in c_nodes[n].get("router_ids", []):
                    router_to_node[r] = n
            
            router_effs = [(r, g_links.get(r, 0.0)) for r in all_assigned_cores]
            router_effs.sort(key=lambda x: x[1])
            
            worst_routers = [r for r, _ in router_effs[:num_to_swap]]
            best_routers = [r for r, _ in router_effs[-num_to_swap:]]
            
            swapped_count = 0
            for w_router, b_router in zip(worst_routers, best_routers):
                w_node = router_to_node.get(w_router)
                b_node = router_to_node.get(b_router)
                
                if w_node and b_node and w_node != b_node:
                    w_rids = list(c_nodes[w_node].get("router_ids", []))
                    b_rids = list(c_nodes[b_node].get("router_ids", []))
                    
                    if w_router in w_rids and b_router in b_rids:
                        w_rids[w_rids.index(w_router)] = b_router
                        b_rids[b_rids.index(b_router)] = w_router
                        
                        set_node_config(c_nodes[w_node], w_rids, w_node, expected_type="core")
                        set_node_config(c_nodes[b_node], b_rids, b_node, expected_type="core")
                        swapped_count += 1
            
            logging.info(f"   [BW-SWAP] Swapped {swapped_count} pairs\n")
            continue
        
        # Ensure minimum cores
        for n in node_names:
            if target_counts.get(n, 0) < MIN_CORES_PER_TASK:
                target_counts[n] = MIN_CORES_PER_TASK
        
        # =====================================================================
        # ALLOCATION LOOP
        # =====================================================================
        if iteration < 8 or (iteration >= 9 and reexplore_start_iter is not None):
            nodes_to_process = sorted(node_names, key=lambda n: target_counts.get(n, 0), reverse=True)
            current_pool = sorted(list(set(CORE_POOL)))
            
            for n in nodes_to_process:
                cnt = target_counts[n]
                strat = base_strat
                
                valid_pool = sorted(list(set(current_pool) - global_used_routers))
                valid_pool = validate_router_list(valid_pool, "core")
                
                assigned_ids = get_placement(strat, cnt, valid_pool, node_name=n, 
                                            use_name_heuristic=use_name_heuristic)
                
                for r in assigned_ids: global_used_routers.add(r)
                set_node_config(c_nodes[n], assigned_ids, n, expected_type="core")
                
                debug_info[n] = {"cnt": cnt, "strat": strat}
            
            # Logging
            logging.info(f"\n   ┌─ GROUP: {group_key} (Strategy: {base_strat})")
            logging.info(f"   ├─ PARTITIONING:")
            for n in sorted(node_names):
                cnt = target_counts.get(n, 0)
                pct = (cnt / len(CORE_POOL) * 100) if CORE_POOL else 0
                logging.info(f"   │    {n:<25} : {cnt:>2} cores ({pct:>5.1f}%)")
            
            logging.info(f"   ├─ PLACEMENT:")
            for n in sorted(node_names):
                rids = c_nodes[n].get("router_ids", [])
                dims = c_nodes[n].get("partition_dims", [])
                strat = debug_info.get(n, {}).get("strat", "?")
                shape_str = get_bbox_str(rids)
                logging.info(f"   │    {n:<25} | {len(rids):<2} | {shape_str:<5} | {strat:<2} | {str(dims)}")
            
            logging.info(f"   └─ Total: {len(global_used_routers)}/{len(CORE_POOL)}\n")

    logging.info(f"{'='*80}\n")

    
    """ Edit code above """
