import numpy as np
from tensorflow.keras.models import load_model
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from keras.utils import load_img, img_to_array

classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']

model = load_model('fashion_mnist_dense.h5')
# Токен вашего бота
API_TOKEN = '6032690563:AAE4uyXyRhlprfK79C9p70X9VYugO98FLlA'

# Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)



@dp.message_handler(content_types=['photo'])
async def handle_photo(message: types.Message):
    photo = message.photo[-1]

    photo_file = await bot.download_file_by_id(photo.file_id)
    with open(f"{photo.file_id}.jpg", "wb") as f:
        f.write(photo_file.read())
    img = load_img(f"{photo.file_id}.jpg", target_size=(28, 28), color_mode="grayscale")

    x = img_to_array(img)
    x = x.reshape(1, 784)
    x = 255 - x
    x /= 255

    prediction = model.predict(x)
    prediction = np.argmax(prediction)

    # print("Номер класса: ", prediction)
    # print("Название класса: ", classes[prediction])
    await bot.send_message(message.chat.id, text=f"{classes[prediction]}")



# Запуск бота
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)