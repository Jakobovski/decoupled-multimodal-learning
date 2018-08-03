class BasicTable:

    def __init__(self, name):
        """

        :param name:
        :return:
        """
        self.name = name
        self.data = {}

    def add(self, item):
        """

        :param item:
        :return:
        """
        self.data[item.name] = item

    def remove(self, item):
        """

        :param item:
        :return:
        """
        del self.data[item.name]

    def get(self, item_name):
        """

        :param item_name:
        :return:
        """
        return self.data[item_name]

    def verify_data_integrity(self):
        pass
