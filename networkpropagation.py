import networkx as nx
import numpy as np
import pandas as pd
import os
from mypy import biostat


def format_string(ppi_path, info_path=None, k=600, outpath=None):
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
    """
    if np.allclose(np.sum(s_matrix, axis=1), atol=1e-03):
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
    """
    if os.path.isfile("{}_largest_component.graphml".format(nm)):
        G = nx.read_graphml("{}_largest_component.graphml".format(nm))
    else:
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
    print(np.max(s_matrix), np.min(s_matrix))
    np.save("{}pr{}f.npy".format(outpath, pr), s_matrix)
    print(s_matrix.shape)
    # node names
    nodes_network = pd.DataFrame(nodes)  # Getting node names(gene names)
    nodes_network.rename(columns={0: "gene_name"}, inplace=True)
    print(nodes_network.shape)
    nodes_network.to_csv("{}pr{}f.csv".format(outpath, pr), index=False)
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
        permute_only_obs: use only observation
    Returns:

    """

    if geneheats.duplicated(subset=["Genes"]).any():
        geneheats = geneheats.groupby("Genes").max().reset_index()
    # add heats to column in gene names only if present
    init_heat = geneheats[geneheats["Genes"].isin(s_matrix_names["gene_name"])]
    # now add all the rest of the names with 0 heat
    init_heat = pd.merge(
        init_heat, s_matrix_names, left_on="Genes", right_on="gene_name", how="outer"
    ).drop("Genes", axis=1)
    init_heat.fillna(0, inplace=True)
    init_heat.set_index("gene_name", inplace=True)
    # essential to keep correct order
    init_heat = init_heat.reindex(index=s_matrix_names["gene_name"])
    # initial matrix multiplication
    prop_heat = np.matmul(s_matrix, init_heat.values)

    # sanity check
    if abs(np.sum(prop_heat) - np.sum(init_heat.values)) > 0.001:
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
            "p": p_value,
            "q": biostat.multiple_testing_correction(p_value),
            "self_heat": init_heat.values.flatten() * np.diagonal(s_matrix).flatten(),
        },
        index=init_heat.index,
    ).reset_index()
    signf_g = results[results["q"] <= 0.05]
    contrb = calc_contribution(s_matrix, init_heat, signf_g, net_heat_only)
    results = results[results['q']>0]
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
    ppi = format_string(ppi_path='meta/string_human.txt',
                        info_path='meta/info_human.txt',
                        k=600,
                        outpath=None)
    #s_matrix, s_matrix_names = gen_s_matrix(ppi, outpath='meta/s_matrix_human', pr=0.5)
    s_matrix = np.load("meta/s_matrix_humanpr0.3f.npy")
    s_matrix_names = pd.read_csv("meta/s_matrix_humanpr0.3f.csv")
    data = pd.read_csv("malaria_stats_data.csv")
    data = data[data["p"] > 0]
    data["heat"] = mag_score(data["FC_HMT_LMT"].values, data["p"].values)
    day = 0
    data_heat = data[data["Day"] == day]
    data_heat = data_heat[["Genes", "heat"]]
    results, contrb = propagate_s_matrix(
        s_matrix,
        s_matrix_names,
        data_heat,
        nperm=20000,
        net_heat_only=True,
        permute_only_obs=False,
    )
    results.to_csv("results_netprop{}Day.csv".format(str(day)))
    signf_net = ppi.subgraph(results[results['q']<=0.05]['gene_name'].values)
    nx.write_graphml(ppi, path='signif_net_{}Day.graphml'.format(str(day)))




if __name__ == '__main__':
    main()
