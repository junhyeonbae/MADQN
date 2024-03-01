


shared_shape = (args.map_size + (predator1_view_range-2)*2, args.map_size + (predator1_view_range-2)*2, 3)

def from_guestbook(self):
    x_start = self.pos[1] + args.predator1_view_range - 2
    y_start = self.pos[0] + args.predator1_view_range - 2

    x_range = int(self.view_range)
    y_range = int(self.view_range)
    z_range = self.shared_shape[2]

    extracted_area = self.shared[x_start - (x_range - 1):x_start + (x_range + 1),
                     y_start - (y_range - 1): y_start + (y_range + 1),
                     :z_range]

    return extracted_area


def to_guestbook(self, info):
    x_start = self.pos[1] + args.predator1_view_range - 2
    y_start = self.pos[0] + args.predator1_view_range - 2

    x_range = int(self.view_range)
    y_range = int(self.view_range)

    self.shared[x_start - (x_range - 1):x_start + (x_range + 1), y_start - (y_range - 1): y_start + (y_range + 1),
    :] += info
