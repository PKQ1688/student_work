import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Warehouse:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))

    def add_robot(self, x, y):
        self.grid[y, x] = 1  # 假设1代表有机器人

    def move_robot(self, old_x, old_y, new_x, new_y):
        if 0 <= new_x < self.width and 0 <= new_y < self.height:
            self.grid[old_y, old_x] = 0  # 移除旧位置的机器人
            self.grid[new_y, new_x] = 1  # 添加新位置的机器人


def update(frame, warehouse, img):
    if frame == 0:
        warehouse.add_robot(0, 0)
    elif frame == 1:
        warehouse.move_robot(0, 0, 1, 0)
    elif frame == 2:
        warehouse.move_robot(1, 0, 1, 1)

    img.set_data(warehouse.grid)
    return [img]


# 创建仓库对象
warehouse = Warehouse(5, 5)

# 设置动画
fig, ax = plt.subplots()
img = ax.imshow(warehouse.grid, cmap='viridis', vmin=0, vmax=1)
ani = animation.FuncAnimation(fig, update, frames=3, fargs=(warehouse, img))

plt.show()
