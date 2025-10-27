class Store:
    """
    Represents a retail store with location and inventory information.
    """

    def __init__(self, id, name, city, latitude, longitude):
        self.id = id
        self.name = name
        self.city = city
        self.latitude = latitude
        self.longitude = longitude

    def __repr__(self):
        return f"Store(id={self.id}, name='{self.name}', city='{self.city}')"


class Product:
    """
    Represents a product with attributes and sales velocity information.
    """

    def __init__(self, id, name, category, price, cost):
        self.id = id
        self.name = name
        self.category = category
        self.price = price
        self.cost = cost

    def __repr__(self):
        return f"Product(id={self.id}, name='{self.name}', category='{self.category}')"
