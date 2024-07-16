import pygame
import random

# 初始化pygame
pygame.init()

# 定义颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# 设置屏幕宽高和每个格子的大小
screen_width = 640
screen_height = 480
grid_size = 20

# 创建屏幕对象
screen = pygame.display.set_mode([screen_width, screen_height])

# 设置窗口标题
pygame.display.set_caption("贪吃蛇")

# 控制游戏循环的变量
done = False

# 控制刷新速度的时钟对象
clock = pygame.time.Clock()

# 初始化贪吃蛇头部的位置和初始长度
snake_head = [100, 50]
snake_body = [[100, 50], [90, 50], [80, 50]]

# 初始化食物的位置
food_pos = [random.randrange(1, (screen_width // grid_size)) * grid_size,
            random.randrange(1, (screen_height // grid_size)) * grid_size]

# 初始化移动方向
direction = "RIGHT"
change_to = direction

# 定义函数用于显示分数
def show_score(score):
    font = pygame.font.SysFont('Calibri', 20)
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, [10, 10])

# 游戏循环
while not done:
    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and direction != "DOWN":
                change_to = "UP"
            elif event.key == pygame.K_DOWN and direction != "UP":
                change_to = "DOWN"
            elif event.key == pygame.K_LEFT and direction != "RIGHT":
                change_to = "LEFT"
            elif event.key == pygame.K_RIGHT and direction != "LEFT":
                change_to = "RIGHT"

    # 更新方向
    direction = change_to

    # 根据方向移动贪吃蛇头部的位置
    if direction == "UP":
        snake_head[1] -= grid_size
    elif direction == "DOWN":
        snake_head[1] += grid_size
    elif direction == "LEFT":
        snake_head[0] -= grid_size
    elif direction == "RIGHT":
        snake_head[0] += grid_size

    # 更新贪吃蛇的身体
    snake_body.insert(0, list(snake_head))

    # 检查是否吃到了食物
    if snake_head == food_pos:
        # 生成新的食物位置
        food_pos = [random.randrange(1, (screen_width // grid_size)) * grid_size,
                    random.randrange(1, (screen_height // grid_size)) * grid_size]
    else:
        # 若没有吃到食物，则删除最后一个方块
        snake_body.pop()

    # 绘制背景
    screen.fill(BLACK)

    # 绘制贪吃蛇身体
    for pos in snake_body:
        pygame.draw.rect(screen, GREEN, pygame.Rect(pos[0], pos[1], grid_size, grid_size))

    # 绘制食物
    pygame.draw.rect(screen, RED, pygame.Rect(food_pos[0], food_pos[1], grid_size, grid_size))

    # 边界检测
    if snake_head[0] < 0 or snake_head[0] > screen_width - grid_size:
        done = True
    elif snake_head[1] < 0 or snake_head[1] > screen_height - grid_size:
        done = True

    # 身体碰撞检测
    if snake_head in snake_body[1:]:
        done = True

    # 显示分数
    show_score(len(snake_body) - 3)

    # 更新屏幕
    pygame.display.update()

    # 控制游戏刷新速度
    clock.tick(10)

# 退出pygame
pygame.quit()