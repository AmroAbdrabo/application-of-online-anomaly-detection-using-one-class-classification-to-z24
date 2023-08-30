def prune_label(P, u, Au, Lu, ch, z, N, eu, wu):
    """
    Function to refine the pruning (P) and labelling (L) for the DH active learner.
    
    Parameters:
    - P : list
        Input pruning, to be refined (array of node/cluster numbers).
    - u : dict
        Contains N-1 arrays. Each array represents a node in the hierarchical tree.
    - Au : list
        Logistic array, indicating the admissibility of each cluster in the current pruning.
    - Lu : dict
        Cell containing the majority label(s), for each node.
    - ch : numpy array
        2x(N-1) matrix containing the 2 children nodes for each cluster.
    - z : numpy array
        Sampled data information array.
    - N : int
        Total number of clusters.
    - eu : list
        Error associated with propagating the majority label to unlabelled instances.
    - wu : list
        Node weights - proportion of the total data in each node.
        
    Returns:
    - P_ : list
        Output (refined) pruning.
    - L : list
        The majority label for each cluster in the refined pruning.
    - XL : numpy array
        The labelled dataset provided by the DH learner.
    """
    
    L = [0] * len(u)  # Initialize admissible cluster label pairs
    P_ = list(P)  # Define working pruning
    print("length of u is ")
    print(len(u))


    for i, v in enumerate(P):  # For each node in the current pruning
        # LABEL parent node in case descendants are not admissible
        Lu[0] = 0  # Arbitrary label of root
        if len(P) != 1 and len(Lu[v]) == 1:
            L[v - 1] = int(Lu[v])

        # Identify first descendants...
        chv = ch[:, v - 1]
        Pv = [int(v)]
        Achv = [int(Au[x]) for x in chv] # x was previously x - 1 but we already decrement indices by 1 in h_cluster

        # While at least one pair of siblings is admissible, refine and label Pv
        while np.sum(int(np.sum(Achv)) == 2) >= 1:
            i_ch = [idx for idx, val in enumerate(Achv) if np.sum(val) == 2]

            for ich in i_ch:
                ep = eu[Pv[ich] - 1]
                ech = (1 / np.sum([wu[x - 1] for x in chv[:, ich]])) * np.sum(
                    [wu[x - 1] * eu[x - 1] for x in chv[:, ich]])
                Lch = [int(Lu[x]) for x in chv[:, ich]]
                Lch_log = [len(x) == 1 for x in Lch]

                if len(Lu[Pv[ich]]) == 1 and ech < ep and np.sum(Lch_log) == 2:
                    Pv[ich] = 0
                    u_ = chv[:, ich]
                    Pv.extend(list(u_))

            Pv = [x for x in Pv if x != 0]

            if len(set(Pv)) > chv.shape[1]:
                chv = ch[:, Pv]
                Achv = [int(Au[x]) for x in chv.flatten()]
            else:
                break

        if len(Pv) > 1:
            print(f"\nTOTAL CLUSTERS {len(P_)}: nodes {Pv} replace node [{v}]")
            P_ = [x for x in P_ if x != v]
            P_.extend(Pv)
            for uw in Pv:
                L[uw - 1] = Lu[uw]

    xl = [0] * N
    for idx, v in enumerate(P):
        if len(P) >= len(set(z[:, -1])):
            xi = u[v]
            xl[xi - 1] = L[v - 1]

    for idx in range(z.shape[0]):
        z_ = int(z[idx, 0])
        zl = z[idx, -1]
        xl[z_ - 1] = int(zl)

    XL = np.array([[idx, x] for idx, x in enumerate(xl) if x != 0])

    return P_, L, XL