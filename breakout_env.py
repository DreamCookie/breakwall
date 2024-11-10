import pygame
import numpy as np

class BreakoutEnv:
    def __init__(self, width=400, height=300):
        pygame.init()
        self.width = width
        self.height = height
        self.paddle_width = 60
        self.paddle_height = 10
        self.ball_radius = 5
        self.reset()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Breakout Environment')

    def reset(self):
        # Инициализация платформы
        self.paddle_x = (self.width - self.paddle_width) / 2
        # Инициализация мяча
        self.ball_x = self.width / 2
        self.ball_y = self.height / 2
        self.ball_speed_x = 3 * np.random.choice([-1, 1])
        self.ball_speed_y = -3  # Начинаем движение вверх
        # Инициализация кирпичей
        self.bricks = []
        brick_rows = 5
        brick_cols = 8
        brick_width = self.width / brick_cols
        brick_height = 15
        for row in range(brick_rows):
            for col in range(brick_cols):
                brick_x = col * brick_width
                brick_y = row * brick_height + 30
                self.bricks.append(pygame.Rect(brick_x, brick_y, brick_width, brick_height))
        return self.get_state()

    def step(self, action):
        reward = 0  # Инициализация переменной reward

        # Обработка действия агента
        if action == 1:
            self.paddle_x -= 5
        elif action == 2:
            self.paddle_x += 5
        self.paddle_x = np.clip(self.paddle_x, 0, self.width - self.paddle_width)

        # Обновление позиции мяча
        self.ball_x += self.ball_speed_x
        self.ball_y += self.ball_speed_y

        # Проверка столкновений с границами окна
        if self.ball_x <= 0 or self.ball_x >= self.width:
            self.ball_speed_x *= -1
        if self.ball_y <= 0:
            self.ball_speed_y *= -1

        # Столкновение с платформой
        paddle_rect = pygame.Rect(self.paddle_x, self.height - self.paddle_height,
                                  self.paddle_width, self.paddle_height)
        ball_rect = pygame.Rect(self.ball_x - self.ball_radius, self.ball_y - self.ball_radius,
                                self.ball_radius * 2, self.ball_radius * 2)
        if ball_rect.colliderect(paddle_rect):
            self.ball_speed_y *= -1
            self.ball_y = self.height - self.paddle_height - self.ball_radius
            reward += 0.5  # Награда за успешное отражение мяча

        # Столкновение с кирпичами
        hit_index = ball_rect.collidelist(self.bricks)
        if hit_index != -1:
            hit_brick = self.bricks.pop(hit_index)
            if abs(self.ball_x - (hit_brick.x + hit_brick.width / 2)) < hit_brick.width / 2:
                self.ball_speed_y *= -1
            else:
                self.ball_speed_x *= -1
            reward += 1  # Добавляем награду за уничтожение кирпича

        # Небольшая награда за каждый шаг
        reward += 0.01

        # Проверка окончания игры
        done = False
        if self.ball_y >= self.height:
            done = True
            reward += -5  # штраф за потерю мяча
        elif len(self.bricks) == 0:
            done = True
            reward += 10  # Награда за уничтожение всех кирпичей

        return self.get_state(), reward, done, {}

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self.screen.fill((0, 0, 0))
        # Отрисовка платформы
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (self.paddle_x, self.height - self.paddle_height,
                          self.paddle_width, self.paddle_height))
        # Отрисовка мяча
        pygame.draw.circle(self.screen, (255, 255, 255),
                           (int(self.ball_x), int(self.ball_y)), self.ball_radius)
        # Отрисовка кирпичей
        for brick in self.bricks:
            pygame.draw.rect(self.screen, (255, 0, 0), brick)
        pygame.display.flip()

    def get_state(self):
        # Текущее состояние как  numpy массив
        state = np.array([
            self.paddle_x / (self.width - self.paddle_width),
            self.ball_x / self.width,
            self.ball_y / self.height,
            self.ball_speed_x / 5,
            self.ball_speed_y / 5
        ])
        return state
