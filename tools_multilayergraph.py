import matplotlib.pyplot as plt
import numpy as np


def convert_tensor2supra_adjacency(tensorgraph):
    # ToDo: this code can be optimized using np.transpose and np.reshape.
    nnode, nlayers, _, _ = tensorgraph.shape
    supra_adj1 = np.zeros((nnode*nlayers, nnode*nlayers))
    for l1 in range(1, nlayers+1):
        ind1_start = (l1 - 1) * nnode
        ind1_end = l1 * nnode
        for l2 in range(1, nlayers+1):
            ind2_start = (l2 - 1) * nnode
            ind2_end = l2 * nnode
            supra_adj1[ind1_start:ind1_end, ind2_start:ind2_end] = tensorgraph[:, l1-1, :, l2-1]
    return supra_adj1


class MultiLayerGraph:
    """
    A tensor-formulation of a multilayer graph
    """
    # ToDo: we should decide if we wanna do it with NetworkX or pure python
    # ToDo: time is not included in this implementation
    layers = {}  # keeps the layers --> layers[n] contains the adjacency matrix of layer n
    interlayers = {}  # keeps the inter-layer connections -->
                      # interlayers['n-m'] contains the adjacency matrix of the connections between layers n and m

    def __init__(self):
        self.layers = self.layers
        self.interlayers = self.interlayers
        self.tensor = None
        self.supra_adj = None
        self.n_nodes = 0
        self.n_layers = 0
        self.binary = False
    # def check_up_layers(self):
    #     current_layers = list(self.layers.keys())

    def update(self, layers, interlayers):
        # ToDo: check up the layers to have the valid structure. for now WE should pass valid structures to the class:D
        # ToDo: check that the layers have the same number of nodes
        self.layers = layers
        self.interlayers = interlayers
        self.n_nodes = layers[1].shape[0]
        self.n_layers = len(list(layers.keys()))

    # def delete_node():

    # def add_node():

    # def delete_layer():

    def _interlayer_inds(self, key):
        ind = key.index('-')
        l1 = int(key[:ind])
        l2 = int(key[ind + 1:])
        return l1, l2

    # def _checkup_layers(self):

    def _checkup_interlayers(self):
        # self._checkup_layers()
        keys_layers = np.asarray(list(self.layers.keys()))
        for l1 in keys_layers:
            for l2 in keys_layers:
                if l1 == l2:
                    continue
                keys_interlayers = np.asarray(list(self.interlayers.keys()))
                key_1 = str(l1) + '-' + str(l2)
                ind = np.where(keys_interlayers == key_1)[0]
                if not len(ind):
                    key_2 = str(l2) + '-' + str(l1)
                    if np.isin(key_2, keys_interlayers):
                        self.interlayers[key_1] = self.interlayers[key_2].T
                    else:  # if l1-l2 and l2-l1 are not specified
                        self.interlayers[key_1] = np.eye(self.n_nodes)
                        self.interlayers[key_2] = np.eye(self.n_nodes)

    def add_layer(self, key, graph):
        try:
            key1 = int(key)
            self.layers[key1] = graph
            self.n_layers += 1
            if self.n_nodes == 0:
                self.n_nodes = graph.shape[0]
            else:
                try:
                    assert(self.n_nodes == graph.shape[0])
                except AssertionError:
                    raise Exception("the number of nodes in the given layer does not match with the previous layers")
        except ValueError:
            self.interlayers[key] = graph

    def get_layer(self, key):
        try:
            key1 = int(key)
            return self.layers[key1]
        except ValueError:
            return self.interlayers[key]

    def convert2supra_adjacency(self):
        self._checkup_interlayers()
        self.supra_adj = np.zeros((self.n_nodes*self.n_layers, self.n_nodes*self.n_layers))
        keys_layers = np.asarray(list(self.layers.keys()))
        for l1 in keys_layers:
            ind1_start = (l1 - 1) * self.n_nodes
            ind1_end = l1 * self.n_nodes
            for l2 in keys_layers:
                ind2_start = (l2 - 1) * self.n_nodes
                ind2_end = l2 * self.n_nodes
                self.supra_adj[ind1_start:ind1_end, ind2_start:ind2_end] = self.tensor[:, l1-1, :, l2-1]

    def compute_transition_tensor(self):
        tensor_normalize = np.zeros(self.tensor.shape)
        _, delta_nl = self.overlapping_degree()  # node-layer overlapping degree node-layer
        delta_nl[delta_nl == 0] = 1  # if it is zero, it does not affect the devision
        for ni in range(self.n_nodes):
            for li in range(self.n_layers):
                tensor_normalize[ni, li, :, :] = self.tensor[ni, li] / delta_nl
        return tensor_normalize

    def supra_eigenvector(self):
        """
        from De Domenico (2015). equation (1) --> eq. (20) of supp. material
        :return:
        """
        from tools_general import eig_power_iteration
        self.convert2supra_adjacency()
        _, cent = eig_power_iteration(self.supra_adj)
        cent_node_layer = cent.reshape((self.n_nodes, self.n_layers), order='F')
        cent_node = np.mean(cent_node_layer, axis=1)
        return cent_node, cent_node_layer

    def supra_randwalk(self):
        """
        from De Domenico (2015). eq. (10) of supp. material and explanations in page 31 of supp material
        :return:
        """
        from tools_general import eig_power_iteration

        tensor_normalize = self.compute_transition_tensor()
        supra_adj1 = convert_tensor2supra_adjacency(tensor_normalize)
        _, cent = eig_power_iteration(supra_adj1)
        cent_node_layer = cent.reshape((self.n_nodes, self.n_layers), order='F')
        cent_node = np.mean(cent_node_layer, axis=1)
        return cent_node, cent_node_layer

    def supra_pagerank(self, rho=0.85):
        """
        from De Domenico (2015). eq. (11) of supp. material and explanations in page 31 & 32 of supp material
        :return:
        """
        from tools_general import eig_power_iteration
        trans_tensor = self.compute_transition_tensor()
        trans_tensor = rho * trans_tensor + (1 - rho) / self.n_nodes / self.n_layers
        supra_adj1 = convert_tensor2supra_adjacency(trans_tensor)
        _, cent = eig_power_iteration(supra_adj1)
        cent_node_layer = cent.reshape((self.n_nodes, self.n_layers), order='F')
        cent_node = np.mean(cent_node_layer, axis=1)
        return cent_node, cent_node_layer

    def convert2tensor(self):
        # ToDo: should do it only if it is not None?
        self._checkup_interlayers()
        self.tensor = np.empty((self.n_nodes, self.n_layers, self.n_nodes, self.n_layers))
        keys_layers = np.asarray(list(self.layers.keys()))
        for l1 in keys_layers:
            self.tensor[:, l1 - 1, :, l1 - 1] = self.layers[l1]
        keys_interlayers = np.asarray(list(self.interlayers.keys()))
        for key in keys_interlayers:
            l1, l2 = self._interlayer_inds(key)
            self.tensor[:, l1 - 1, :, l2 - 1] = self.interlayers[key]

    def overlapping_degree(self):
        """
        Computes overlapping degree of a node
        according to Wang et al 2019, equation 4
        :return:
        delta_node: numpy.ndarray with shape (self.n_nodes,)
                    overlapping degree
        delta_node_layer: numpy.ndarray with shape (self.n_nodes, self.n_layers)
                          node-layer overlapping degree
        """
        self.convert2tensor()
        delta_node_layer = np.sum(np.sum(self.tensor, axis=-1), axis=-1)
        delta_node = np.sum(delta_node_layer, axis=-1)
        return delta_node, delta_node_layer

    def tm_eigenvector(self, epsilon=0.001, max_iter=500, convcurve=False):
        """
        (c) original code by Alina Studenova - 2021
        compute TM ev centrality based on Wang, Yu, & Zou, (2019), equation (12) and (17)
        :param epsilon: stopping threshold for the error
        :param convcurve: if True, the error array is returned
        :return:
        tm_theta: numpy.ndarray with shape (self.n_nodes,)
                  tm_eigenvector centrality
        ek: numpy.ndarray with shape (self.n_nodes, self.n_layers)
            node-layer tm_eigenvector centrality
        error_values: list with length of number of iterations (is returned only if convergence_curve=True)
                      error values of the iterations

        """
        self.convert2tensor()
        if convcurve:
            error_values = []
        ek = np.ones((self.n_nodes, self.n_layers))
        n_iter = 0
        while n_iter <= max_iter:
            n_iter += 1
            etilda = np.empty(ek.shape)
            for ni in range(self.n_nodes):
                for li in range(self.n_layers):
                    etilda[ni, li] = np.sum(self.tensor[ni, li] * ek)
            lambda_k = np.linalg.norm(etilda, ord='fro')  # Frobenius norm
            e_kplus1 = etilda / lambda_k
            error = np.linalg.norm(ek - e_kplus1)
            ek = e_kplus1
            if convcurve:
                error_values.append(error)
            if error < epsilon:
                break
        # Compute centrality of each node from Ek - eq. 17
        tm_theta = np.mean(ek, axis=1)

        if convcurve:
            return tm_theta, ek, error_values
        return tm_theta, ek

    def tm_pagerank(self, rho=0.85, epsilon=0.001, max_iter=500, convcurve=False):
        """
        (c) original code by Alina Studenova (2021)
        compute TM Page-Rank centrality based on Wang, Yu, & Zou, (2019), equation (18) and (24)
        :param rho: teleportation parameter. The default value is taken from the classical PageRank algorithm
        :param epsilon: stopping criterion threshold
        :param convcurve: if True, the error array is returned
        :return:
        """
        if convcurve:
            error_values = []

        pk = np.ones((self.n_nodes, self.n_layers))
        tensor_normalize = self.compute_transition_tensor()
        n_iter = 0
        while n_iter <= max_iter:
            n_iter += 1
            p_kplus1 = np.empty(pk.shape)
            for ni in range(self.n_nodes):
                for li in range(self.n_layers):
                    p_kplus1[ni, li] = rho * np.sum(tensor_normalize[ni, li] * pk) + (1 - rho) / np.product(pk.shape)
            error = np.linalg.norm(pk - p_kplus1)
            pk = p_kplus1
            if convcurve:
                error_values.append(error)
            if error < epsilon:
                break
        # Compute centrality of each node from Pk, equation 24
        phi_pagerank = np.mean(pk, axis=1)

        if convcurve:
            return phi_pagerank, pk, error_values
        return phi_pagerank, pk

    def global_pagerank(self, rho=0.85, f=None, epsilon=0.001, max_iter=500):
        """
        (c) original code by Alina Studenova (2021) -adopted from original MATLAB code of Wu et al (2019)
        compute Global Page-Rank centrality based on Wu et al. (2019)
        :param rho: teleportation parameter
        :param f: default is the L1 norm
        :param epsilon: stopping threshold
        :param max_iter: maximum number of iterations allowed
        :return:
        node centrality
        node-layer centrality
        """

        from tools_general import khatri_rao_, eig_power_iteration
        if f is None:
            f = lambda x: np.linalg.norm(x, ord=1, axis=0, keepdims=True)

        self.convert2supra_adjacency()
        phi_bar = np.random.rand(self.n_nodes * self.n_layers, 1)
        phi_bar = phi_bar / sum(phi_bar)
        phi = phi_bar.reshape((self.n_nodes, self.n_layers), order='F')

        trans_tensor = self.compute_transition_tensor()
        trans_tensor = rho * trans_tensor + (1 - rho) / self.n_nodes / self.n_layers
        trans_tensor_supra_adj = convert_tensor2supra_adjacency(trans_tensor)
        n_iter = 0
        while n_iter <= max_iter:
            n_iter += 1
            influence = f(phi).T / f(phi)
            h_mat = khatri_rao_(trans_tensor_supra_adj, influence)
            _, phi_new = eig_power_iteration(h_mat)

            phi_new = phi_new / np.sum(phi_new)  # do we need this?
            phi_new = phi_new.reshape((self.n_nodes, self.n_layers), order='F')

            error = np.linalg.norm(phi - phi_new)
            phi = phi_new.reshape((self.n_nodes, self.n_layers), order='F')
            if error < epsilon:
                break

        gprphi = np.mean(phi, axis=1)

        return gprphi, phi

