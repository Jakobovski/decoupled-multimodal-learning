from __future__ import division
from cdzproject.db.basic_table import BasicTable
from cdzproject.db.one_to_many_table import OneToManyTable


class Database:

    def __init__(self):
        """This is a wrapper for a data structure that is used to
        maintain the relationships between clusters and nodes, and node/node_manager.
;        """
        self.nodes = BasicTable('nodes')
        self.clusters = BasicTable('clusters')

        self.nodes_to_clusters = OneToManyTable('nodes_to_clusters')
        self.clusters_to_nodes = OneToManyTable('clusters_to_nodes')
        self.node_manager_to_nodes = OneToManyTable('node_manager_to_nodes')

    def add_node(self, node, cluster, initial=False):
        """

        :param node:
        :param cluster:
        :param initial:
        :return:
        """
        print ">> adding node:", node.name, "(initial)" if initial else ""
        self.nodes.add(node)
        self.clusters.add(cluster)

        # We are always making a new clusters and a new node
        self.nodes_to_clusters.add(node, [cluster], [1])
        self.clusters_to_nodes.add(cluster, [node], [1])

        # Add node to node_manager_to_nodes
        self.node_manager_to_nodes.add_related_item(node.cortex.node_manager, node)

    def add_cluster(self, cluster):
        """

        :param cluster:
        :return:
        """
        raise Exception('This should never be called. Clusters are automatically added with new nodes.')

    def delete_node(self, node):
        """

        :param node:
        :return:
        """
        print ">> removing node:", node.name
        self.nodes.remove(node)

        clusters = self.get_nodes_clusters(node)
        for cluster in clusters:
            self.clusters_to_nodes.remove_related_item(cluster, node)

        self.nodes_to_clusters.remove(node)
        self.node_manager_to_nodes.remove_related_item(node.cortex.node_manager, node)

    def _delete_cluster(self, cluster, force=False):
        """

        :param cluster:
        :param force:
        :return:
        """
        print ">> removing cluster:", cluster.name

        if force:
            nodes = self.get_clusters_nodes(cluster)
            for node in nodes[:]:
                self.nodes_to_clusters.remove_related_item(node, cluster)
                self.clusters_to_nodes.remove_related_item(cluster, node)

                # if the node only had this as the cluster then lets remove the node
                if len(self.get_nodes_clusters(node)) == 0:
                    self.delete_node(node)

        # Check the cluster has no nodes.
        assert len(self.get_clusters_nodes(cluster)) == 0
        self.clusters.remove(cluster)
        self.clusters_to_nodes.remove(cluster)

        # Remove cluster from CDZ
        ce = cluster.cdz
        ce.remove_cluster(cluster)

    def get_clusters_nodes(self, cluster, include_strengths=False):
        """

        :param cluster:
        :param include_strengths:
        :return:
        """
        data = self.clusters_to_nodes.get(cluster)

        if include_strengths:
            return data['list'], data['strengths']
        else:
            return data['list']

    def get_nodes_clusters(self, node, include_strengths=False, include_all=False):
        """

        :param node:
        :param include_strengths:
        :return:
        """
        data = self.nodes_to_clusters.get(node)

        if include_all:
            return data['list'], data['strengths'], data['position'], data['count']
        elif include_strengths:
            return data['list'], data['strengths']
        else:
            return data['list']

    def get_node_managers_nodes(self, node_manager):
        """

        :param node_manager:
        :return:
        """
        return self.node_manager_to_nodes.get(node_manager)['list']

    def adjust_node_to_cluster_strength(self, node, cluster, amount, last_encoding):
        """ Amount is a quantity that is added not multiplied.

        :param node:
        :param cluster:
        :param amount:
        :return:
        """
        is_node_related = self.nodes_to_clusters.is_related(node, cluster)
        is_cluster_related = self.nodes_to_clusters.is_related(node, cluster)

        assert is_node_related == is_cluster_related

        # If this is a new relationship
        if not is_node_related:
            # These should be of strength 0, not strength 1 as it is a new item
            self.nodes_to_clusters.add_related_item(node, cluster, amount, last_encoding)
            self.clusters_to_nodes.add_related_item(cluster, node, amount)
        else:
            # increase the relationship strength
            self.nodes_to_clusters.increase_relationship_strength(node, cluster, amount, last_encoding)

    def adjust_cluster_to_node_strength(self, cluster, node, amount):
        """

        :param cluster:
        :param node:
        :param amount:
        :return:
        """
        self.clusters_to_nodes.increase_relationship_strength(cluster, node, amount)

    def cleanup(self):
        """Performs maintenance on the system.
            - Deletes clusters that are underutilized

        :return:
        """
        # Deletes clusters that are underutilized
        clusters_to_delete = []
        for cluster_data in self.clusters_to_nodes.data.values():
            cluster = cluster_data['obj']
            if cluster.is_underutilized():
                clusters_to_delete.append(cluster)

        for cluster in clusters_to_delete:
            self._delete_cluster(cluster, force=True)

    def verify_data_integrity(self):
        """Verifies the consistency of data in the DB.

        :return:
        """
        self.nodes.verify_data_integrity()
        self.clusters.verify_data_integrity()
        self.nodes_to_clusters.verify_data_integrity()
        self.clusters_to_nodes.verify_data_integrity()
        self.node_manager_to_nodes.verify_data_integrity()

        def _cross_table_validation(table1, table2):
            """Validation on the relationships between tables."""
            for item_name, data in table1.data.iteritems():
                related_items = data['list']
                item = data['obj']

                # Verify that each relationship is stored in the corresponding table
                for related_item in related_items:
                    # Verify every cluster that is referenced by a node exists in the cluster list
                    assert table2.get(related_item)
                    # verify the node is in the cluster's node_list
                    assert table2.is_related(related_item, item)

        # Run cross-table validation
        _cross_table_validation(self.nodes_to_clusters, self.clusters_to_nodes)
        _cross_table_validation(self.clusters_to_nodes, self.nodes_to_clusters)
