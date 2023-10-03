def DH_AL(u, ch, B, T, y):
    # Initialize variables
    N = len(u[0])
    print("N is ", N)
    Nu = [len(ue) for ue in u]
    wu = np.array(Nu) / N
    
    z = np.empty((0, 3))
    uz = [ [] for _ in range(N-1)]
    u_ = u.copy()
    
    pl = [None] * len(u)
    Aul = [None] * len(u)
    Au = np.zeros(len(u) + N)
    eu = np.ones(len(u) + N)
    Lu = [None] * len(u)
    
    P = [0]
    
    for t in range(1, T+1):
        for b in range(1, B+1):
            # Select v from P
            prop = wu[P]
            for i, v in enumerate(P):
                coeff = 1 if len(P) == 1 else 0 if not u_[v] else 1 - max(pl[v][:, -2])
                prop[i] *= coeff
            
            prob = prop / np.sum(prop)
            vi = choices(range(len(prob)), prob)[0]
            
            # Query label
            s = 0
            if len(u_[P[vi]]) != 0:
                s = int(sample(u_[P[vi]], 1)[0])
            else:
                continue
            for ui in u_:
                if s in ui:
                    ui.remove(s)
            
            l = y[s]
            
            sampled_vec = np.array([s, P[vi], l])
            z = np.vstack([z, sampled_vec])
            
            # Update node counts
            u_i = [i for i, ue in enumerate(u) if s in ue]
            for uw in u_i:
                uz[uw].append((s, l))
                nu = len(uz[uw])
                
                cl, c = zip(*Counter([l for s, l in uz[uw]]).items())
                p_l = np.array(c) / nu
                
                delta = 1 / nu + np.sqrt((p_l * (1 - p_l)) / nu)
                lb = np.maximum(p_l - delta, 0)
                ub = np.minimum(p_l + delta, 1)
                
                pl[uw] = np.column_stack((cl, p_l, lb, ub))
                Lu[uw] = cl[np.argmax(p_l)]
                
        # Update admissibilities, error/scores
        u_i = [i for i, ple in enumerate(pl) if ple is not None]
        for uw in u_i:
            beta = 1.5
            p_l = pl[uw]
            if len(uz[uw]) > 1:
                LHS = p_l[:, -2]
                RHS = beta * p_l[:, -1] - 1
                a_l = LHS[:, None] > RHS
                
                a_l[np.diag_indices_from(a_l)] = False
                idx = np.all(a_l, axis=1)
                
                Aul[uw] = p_l[idx, 0]
                Au[uw] = np.any(idx)
                eu[uw] = 1 - np.max(p_l[:, 1]) if np.any(idx) else 1
                
        eu[Au == 0] = 1
        
        # Refine pruning & labelling
        P, L, XL = prune_label(P, u, Au, Lu, ch, z, N, eu, wu)
    
    return XL, np.array(z), P, L