class Registry:
    def __init__(self):
        self.next_id = 0
        self.active_ids = set()

    def get_new_id(self):
        id = self.next_id
        self.next_id += 1
        self.active_ids.add(id)
        return id

    def remove_id(self, id):
        self.active_ids.discard(id)

    def is_active(self, id):
        return id in self.active_ids

    def get_active_count(self):
        return len(self.active_ids)