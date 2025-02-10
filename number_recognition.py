#распознава́ния чи́сел
import numpy as np
import pygame  # Используется для реализации интерфейса рисования
import torch  #Фреймво́рк для рабо́ты с нейро́нными сетя́ми
import torch.nn as nn  
import torch.optim as optim 
from torchvision import datasets, transforms  # Для загрузки датасетов и обработки изображе́ний
from PIL import Image  
import cv2  

# Определение архитектуры нейронной сети CNN 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Первый сверточный слой: вход 1 канал (черно-белое изображение), выход 32 канала
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # Второй сверточный слой: вход 32 канала, выход 64 канала
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Третий сверточный слой: вход 64 канала, выход 128 каналов
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # Полносвязные слои для классификации
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # Первый полносвязный слой
        self.fc2 = nn.Linear(256, 10)  # Выходной слой: 10 классов (цифры от 0 до 9)

    def forward(self, x):
        # Прямое распространение через слои сети 
        x = torch.relu(self.conv1(x))  # Активация ReLU после первого слоя

        x = torch.max_pool2d(x, 2)  
        x = torch.relu(self.conv2(x))  # Активация ReLU после второго слоя
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))  # Активация ReLU после третьего слоя
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x)) 
        x = self.fc2(x) 
        return x

# Функция для обучения модели
def train_model():
    # Подгото́вка да́нных для обуче́ния с примене́нием аугмента́ции
    transform = transforms.Compose([
        transforms.RandomRotation(10),  
        transforms.RandomAffine(0, translate=(0.1, 0.1)), 
        transforms.ToTensor(),  
        transforms.Normalize((0.5,), (0.5,))  # Нормализа́ция
    ])
    # Загрузка датасета MNIST
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = CNN()  
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Оптимизатор Adam

    # Обучение модели
    epochs = 10  #Коли́чество эпо́х обуче́ния 
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()  
            output = model(images) 
            loss = criterion(output, labels)  
            loss.backward()  
            optimizer.step()  
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # Сохранение обученной модели
    torch.save(model.state_dict(), 'cnn_model.pth')
    return model

# Разделе́ние числа́ на отде́льные ци́фры
def divide_number_into_digits(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Загру́зка изображе́ния в града́циях се́рого
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # Бинариза́ция
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # По́иск ко́нтуров
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])  # Сортиро́вка по координа́те X
    digits = []
    for i, ctr in enumerate(contours):
        x, y, w, h = cv2.boundingRect(ctr)  # Выделе́ние ограни́чивающего прямоуго́льника
        if w * h > 100:  # Игнори́рование ме́лкого шу́ма
            digit = thresh[y:y+h, x:x+w]  # Извлече́ние о́бласти изображе́ния с ци́фрой

            padded_digit = cv2.copyMakeBorder(digit, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
            padded_digit = cv2.resize(padded_digit, (28, 28), interpolation=cv2.INTER_AREA)
            digits.append(padded_digit)  
            cv2.imwrite(f'digit_{i}.png', padded_digit)  
    return digits

# Распознава́ние цифр
def recognize_digits(model, digits):
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    recognized = []
    for digit in digits:
        digit_img = Image.fromarray(digit) 
        digit_tensor = transform(digit_img).unsqueeze(0)  #Добавле́ние разме́рности для паке́та
        with torch.no_grad(): 
            output = model(digit_tensor)  
            pred = torch.argmax(output, dim=1).item()  # Получе́ние предсказа́ния
            recognized.append(pred)  
    return recognized

# ИИнтерфе́йс рисова́ния с испо́льзованием Pygame
def drawing_interface(model):
    pygame.init()
    screen = pygame.display.set_mode((600, 400))  
    screen.fill('white')  
    pygame.display.set_caption("Draw Numbers")  
    running = True
    drawing = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: 
                drawing = True
            if event.type == pygame.MOUSEBUTTONUP:  
                drawing = False
            if event.type == pygame.KEYDOWN: 
                if event.key == pygame.K_RETURN: 
                    pygame.image.save(screen, "num.jpg")  
                    digits = divide_number_into_digits("num.jpg")  
                    recognized = recognize_digits(model, digits)  
                    print("Recognized Number:", "".join(map(str, recognized)))
                if event.key == pygame.K_ESCAPE: 
                    screen.fill('white') 
            if drawing:
                pygame.draw.circle(screen, 'black', event.pos, 7)  
        pygame.display.flip()  
    pygame.quit()

if __name__ == "__main__":
    model = CNN()  
    try:
        model.load_state_dict(torch.load('cnn_model.pth')) 
        model.eval() 
        print("Model loaded successfully.")
    except FileNotFoundError:  
        print("Training new model...")
        model = train_model()
    drawing_interface(model)  