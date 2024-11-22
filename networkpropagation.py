import networkx as nx
import numpy as np
import pandas as pd
import os
import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed



def multiple_testing_correction(pvalues, correction_type="FDR"):
    """
    Consistent with R - print
    correct_pvalues_for_multiple_testing([0.0, 0.01, 0.029, 0.03, 0.031, 0.05,
     0.069, 0.07, 0.071, 0.09, 0.1])
    """
    from numpy import array, empty

    #pvalues = array(pvalues)
    sample_size = pvalues.shape[0]
    qvalues = empty(sample_size)
    if correction_type == "Bonferroni":
        #  Bonferroni correction
        qvalues = sample_size * pvalues
    elif correction_type == "Bonferroni-Holm":
        #  Bonferroni-Holm correction
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        for rank, vals in enumerate(values):
            pvalue, i = vals
            qvalues[i] = (sample_size - rank) * pvalue
    elif correction_type == "FDR":
        #  Benjamini-Hochberg, AKA - FDR test
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        values.reverse()
        new_values = []
        for i, vals in enumerate(values):
            rank = sample_size - i
            pvalue, index = vals
            new_values.append((sample_size / rank) * pvalue)
        for i in range(0, int(sample_size) - 1):
            if new_values[i] < new_values[i + 1]:
                new_values[i + 1] = new_values[i]
        for i, vals in enumerate(values):
            pvalue, index = vals
            qvalues[index] = new_values[i]
    return qvalues



def format_string(ppi_path, info_path=None, k=800, outpath=None):
    """
    read string txt file and return networkX object with correct id (gn)
    Args:
        ppi_path: pathway to save network in graphml format
        info_path: name for mapping string id to gene names, not needed
        k: score to fiter interactions, default 600
    Returns:
        formatted ppi with gene name as node name

    """
    ppi = pd.read_csv(ppi_path, sep="\s+", engine="python")
    ppi = ppi[ppi["combined_score"] > k]
    ppi = nx.from_pandas_edgelist(df=ppi, source="protein1", target="protein2")
    ppi.remove_edges_from(nx.selfloop_edges(ppi))
    if info_path:
        info = pd.read_csv(info_path, sep="\t")
        info = dict(zip(info["protein_external_id"], info["preferred_name"]))
        ppi = nx.relabel_nodes(ppi, mapping=info, copy=True)
    if outpath:
        nx.write_graphml(ppi, path="{}.graphml".format(outpath))
    return ppi


def test_s_matrix(s_matrix):
    """
    test s_matrix for normality
    s_matrix sum columns should be 1
    """
    sm = np.sum(s_matrix, axis=0)
    if np.allclose(sm, np.full(shape=sm.shape, fill_value=1, dtype=np.int64), atol=1e-03):
        pass
    else:
        raise ValueError("Smatrix has wrong column values")

def gen_s_matrix(G, outpath, pr=0.3, nm="string_human"):
    """
    Generate S_matrix
    Args:
        G: networkX graph object
        outpath: pathway to save the s_matrix and the nodename
        pr = restart_probability
        nm = name for the saved file
    Returns:
        print s_matrix and nodenames to a file
    
            
    NOTE Order around %*% matters and S includes the pr factor !
    col sums of S = 1
    row sums of inv_denom = 1/pr 
    """

    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc)
    nx.write_graphml(G, path="{}_largest_component".format(nm))
    # Calculate matrix inputs
    nodes = G.nodes()
    A = nx.adjacency_matrix(G, nodelist=nodes)
    D = np.array([x[1] for x in nx.degree(G)])

    # Calculate matrix inputs
    nodes = G.nodes()
    A = nx.adjacency_matrix(G, nodelist=nodes)
    D = np.array([x[1] for x in nx.degree(G)])

    # integrity check: node order matches
    degreeNodes = np.array([x[0] for x in nx.degree(G)])
    assert all(degreeNodes == nodes)

    Dinverse = np.diag(1 / D)
    W = np.matmul(A.todense(), Dinverse)
    I = np.identity(W.shape[0])

    s_matrix = pr * np.linalg.inv(I - (1 - pr) * W)
    # print(np.max(s_matrix), np.min(s_matrix))
    np.save("{}pr{}f.npy".format(outpath, pr), s_matrix)
    # print(s_matrix.shape)
    # node names
    nodes_network = pd.DataFrame(nodes)  # Getting node names(gene names)
    nodes_network.rename(columns={0: "gene_name"}, inplace=True)
    # print(nodes_network.shape)
    nodes_network.to_csv("{}pr{}f.csv".format(outpath, pr), index=False)
    test_s_matrix(s_matrix)
    return s_matrix, nodes_network



def propagate_s_matrix(
    s_matrix,
    s_matrix_names,
    geneheats,
    nperm=2000,
    net_heat_only=True,
    permute_only_obs=False,
):
    """
    Performs heat diffusion
    Args:
        s_matrix: numpy s matrix
        s_matrix_names: array with order of names in s_matrix
        geneheats: pd dataframe with one value per gene
        nperm: number of permutations (int)
        net_heat_only: remove self heat for nodes present in geneheats
        permute_only_obs: use only observed genes
    Returns:

    """
    if geneheats.duplicated(subset=["Genes"]).any():
        geneheats = geneheats.groupby("Genes").max().reset_index()
        
    # add heats to column in gene names only if present
    init_heat = geneheats[geneheats["Genes"].isin(s_matrix_names["gene_name"])]
    # lots of genes absent from s_matrix_names
    
    # now add all the rest of the names with 0 heat
    init_heat = pd.merge(
        init_heat, s_matrix_names, left_on="Genes", right_on="gene_name", how="outer"
    ).drop("Genes", axis=1)
    init_heat.fillna(0, inplace=True)
    init_heat.set_index("gene_name", inplace=True)
    ## not all names in geneheats are in s_matrix_names
    
    
    # essential to keep correct order
    init_heat = init_heat.reindex(index=s_matrix_names["gene_name"])
    # initial matrix multiplication
    prop_heat = np.matmul(s_matrix, init_heat.values)
    # sanity check
    if np.abs(np.sum(prop_heat) - np.sum(init_heat.values)) > 0.001:
        print(np.sum(prop_heat), np.sum(init_heat.values))
        print(np.abs(np.sum(prop_heat) - np.sum(init_heat.values)))
        raise ValueError("wrong diff heat")
    maxheat = prop_heat.flatten()
    if net_heat_only:
        # remove self-heat
        maxheat = maxheat - init_heat.values.flatten() * np.diagonal(s_matrix).flatten()
    perm_matrix = permute_heats(init_heat, nperm, limit_obs=permute_only_obs)
    perm_heat = np.matmul(s_matrix, perm_matrix)
    if net_heat_only:
        # remove self-heat
        perm_heat = perm_heat - perm_heat * np.diagonal(s_matrix).reshape(-1, 1)

    # p value calc
    delta_heat = perm_heat - prop_heat
    # fract of positive values
    p_value = np.count_nonzero(delta_heat > 0, axis=1) / nperm
    results = pd.DataFrame(
        {
            "init_heat": init_heat.values.flatten(),
            "prop_heat": prop_heat.flatten(),
            "p": p_value.flatten(),
            'count': np.count_nonzero(delta_heat > 0, axis=1),
            "self_heat": init_heat.values.flatten() * np.diagonal(s_matrix).flatten(),
        },
        index=init_heat.index,
    ).reset_index()
    results['q'] = multiple_testing_correction(results['p'].values, correction_type='FDR')
    signf_g = results[results["q"] <= 0.05]
    contrb = calc_contribution(s_matrix, init_heat, signf_g, net_heat_only)
    #results = results[results['q']>0]
    return results, contrb


def calc_single_contr(sub_df):
    sub_df["contr"] = 100 * sub_df["heat"].values / np.sum(sub_df["heat"].values)
    sub_df.sort_values("contr", inplace=True)
    sub_df["contr"] = np.cumsum(sub_df["contr"])
    return sub_df


def calc_contribution(s_matrix, init_heat, signf_g, net_heat_only):
    signf_idx = signf_g.index.to_list()
    contrib_heat = pd.DataFrame(
        np.multiply(s_matrix[signf_idx].T, init_heat.values),
        columns=signf_g["gene_name"].values,
    )
    contrib_heat["gene_name"] = init_heat.index
    contrib_heat = pd.melt(contrib_heat, id_vars="gene_name")
    contrib_heat.columns = ["From", "To", "heat"]
    if net_heat_only:
        contrib_heat["heat"] = np.where(
            contrib_heat["From"] != contrib_heat["To"], contrib_heat["heat"], 0
        )
        c_heat = contrib_heat.groupby("To").apply(calc_single_contr)
        c_heat = c_heat[c_heat["contr"] > 0]
        return c_heat


def permute_heats(init_heat, nperm, limit_obs=False):
    """ """
    # create empty array of length of gene names and as many cols as permutation
    perm_matrix = np.zeros(shape=(init_heat.shape[0], nperm))
    toshuffle = init_heat.values.flatten()
    if limit_obs:
        # boolean mask
        idx = toshuffle > 0
        sub = toshuffle[idx]
        for i in range(0, nperm):
            np.random.shuffle(sub)
            toshuffle[idx] = sub
            perm_matrix[:, i] = toshuffle
        # need to get indexes of real values in init_heats
        return perm_matrix
    else:
        for i in range(0, nperm):
            np.random.shuffle(toshuffle)
            perm_matrix[:, i] = toshuffle
    return perm_matrix


def mag_score(log2fc, pvalue, magscale=2):
    """
    Args:
        log2fc: array of log2fc
        pvalue:  array of pvalue
        magscale: magnification

    Returns: combined stats

    """
    mag = magscale * np.abs(log2fc)
    significant = -np.log10(pvalue)
    mask = np.where(significant>mag)
    significant[mask] = mag[mask]
    return np.sqrt(mag * significant)


def main():
    matr_nm = 's_matrix_human_string'

    # G = format_string(ppi_path='meta/string_human.txt',
    #                     info_path='meta/info_human.txt',
    #                     k=600,
    #                     outpath=None)
    
    # ppi = pd.read_csv('meta/PathwayCommons12.All.hgnc.sif', sep='\t', header=None)
    # ppi = nx.from_pandas_edgelist(ppi, source=0, target=2)    
    # print(len(ppi.nodes))
    # ppi.add_edges_from(G.edges())
    # print(len(ppi.nodes))
    # ppi.remove_edges_from(nx.selfloop_edges(ppi))
    pr = 0.2
    
    # s_matrix, s_matrix_names = gen_s_matrix(ppi, outpath='meta/{}'.format(matr_nm), pr=pr)
    s_matrix = np.load("meta/{}pr{}f.npy".format(matr_nm, str(pr)))
    s_matrix_names = pd.read_csv("meta/{}pr{}f.csv".format(matr_nm, str(pr)))
    #data = gen_test_data(s_matrix_names)
    data = pd.read_csv("regulated_genes.csv")
    # data = data[['Log2FC', 'GN', 'q']]
    # data.columns = ['Log2FC', 'GN', 'p']    
    # data = data[data["p"] < 0.05]
    data["heat"] = mag_score(data["logFC.TP_6h"].values, data["adj.P.Val.TP_6h"].values, magscale=0.5)
    data = data[["gn", "heat"]]
    data.columns = ['Genes', 'heat'] 
    results, contrb = propagate_s_matrix(
        s_matrix,
        s_matrix_names,
        data,
        nperm=50000,
        net_heat_only=True,
        permute_only_obs=False,
    )
    #results['entity'] = ['Small molecule' if x.startswith('CHEBI') else 'Protein' for x in results['gene_name']]
    results.to_csv("results_netprop_{}.csv".format(matr_nm))
    # signf_net = ppi.subgraph(results[results['p']<=0.05]['gene_name'].values)
    # nx.write_graphml(signf_net,
    #                  path='signif_net_{}Day{}.graphml'.format(str(day), matr_nm))




if __name__ == '__main__':
    main()
