from __future__ import division


class OneToManyTable:

    def __init__(self, name):
        """

        :param name:
        :return:
        """
        self.name = name
        self.data = {}

    def add(self, item, related_items, strengths, position=None):
        """Accepts the item and one of its relationships

        :param item:
        :param related_items:
        :param strengths:
        :return:
        """
        if item.name not in self.data:
            assert isinstance(related_items, list)
            assert isinstance(strengths, list)
            self.data[item.name] = {'obj': item,
                                    'list': related_items,
                                    'strengths': strengths,
                                    'position': [position],
                                    'count': [1]}
        else:
            raise Exception('Item already in table')

    def get(self, item):
        return self.data[item.name]

    def remove(self, item):
        del self.data[item.name]

    def add_related_item(self, item, related_item, strength=1, position=None):
        """

        :param item:
        :param related_item:
        :param strength:
        :return:
        """
        if not self.data.get(item.name):
            self.add(item, [related_item], [1], position=position)
        else:
            if related_item not in self.data[item.name]['list']:
                self.data[item.name]['list'].append(related_item)
                self.data[item.name]['strengths'].append(strength)
                self.data[item.name]['position'].append(position)
                self.data[item.name]['count'].append(1)

            else:
                raise Exception('The item already is related.')

        self._normalize_item(item)

    def remove_related_item(self, item, related_item):
        """

        :param item:
        :param related_item:
        :return:
        """
        idx = self.data[item.name]['list'].index(related_item)
        self.data[item.name]['list'].remove(related_item)
        self.data[item.name]['strengths'].pop(idx)
        self.data[item.name]['position'].pop(idx)
        self.data[item.name]['count'].pop(idx)

        # If the list now has 0 items in it then we should not normalize.
        # Because normalizing will intentionally throw an exception
        if len(self.data[item.name]['list']) > 0:
            self._normalize_item(item)

    def verify_data_integrity(self):
        """

        :return:
        """
        for item_name, val in self.data.iteritems():
            item_list = val['list']

            # Verify that the length of the node_list and and strength are the same
            assert len(item_list) == len(val['strengths'])

            if len(item_list) > 0:
                assert .99 < sum(val['strengths']) <= 1.01

    def _normalize_item(self, item):
        self.data[item.name]['strengths'][:] = self._normalize(self.data[item.name]['strengths'])

    @staticmethod
    def _normalize(list_to_n):
        """Normalizes in place"""
        total = sum(list_to_n)

        if total <= 0.0:
            raise Exception()

        list_to_n[:] = [item / total for item in list_to_n]
        return list_to_n

    def is_related(self, item, related_item):
        """

        :param item:
        :param related_item:
        :return:
        """
        if not self.data.get(item.name):
            return False
        else:
            return bool(related_item in self.data[item.name]['list'])

    def increase_relationship_strength(self, item, related_item, amount, position=None):
        """ Amount is a quantity that is added not multiplied.

        :param item:
        :param related_item:
        :param amount:
        :return:
        """
        assert self.is_related(item, related_item)

        idx = self.data[item.name]['list'].index(related_item)
        self.data[item.name]['strengths'][idx] += amount
        self.data[item.name]['count'][idx] += 1

        # Calculate the moving average of position
        if self.data[item.name]['position'][idx] is not None:
            old_pos = self.data[item.name]['position'][idx]
            new_pos = old_pos + (position - old_pos) / self.data[item.name]['count'][idx]
            # print " delta: ", np.linalg.norm(np.subtract(position, old_pos)), 'count: ', self.data[item.name]['count'][idx]
            self.data[item.name]['position'][idx] = new_pos
        else:
            self.data[item.name]['position'][idx] = position

        self._normalize_item(item)

    def count(self):
        return len(self.data)

    def get_items_without_related_items(self):
        """gets all the items that are not related to anything"""
        return [data['obj'] for data in self.data.values() if len(data['list']) == 0]
