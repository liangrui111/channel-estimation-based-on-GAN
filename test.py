import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import h5py
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
path=r'C:/Users/梁睿/Desktop/GAN_estimation1/new_data/new_data_all.mat'

with h5py.File(path, 'r') as file:
    train_images = np.transpose(np.array(file['train_images']))#10000个信道矩阵
    #print(train_images.shape)
    H = np.transpose(np.array(file['H'])).astype('float32')#用于求nmmse的100个信道矩阵
    #print(H.shape)
    yd5 = np.transpose(np.array(file['yd5'])).astype('float32')
    yd2 = np.transpose(np.array(file['yd2'])).astype('float32')
    y0 = np.transpose(np.array(file['y0'])).astype('float32')#接收信号
    #print(y0.shape)
    y2 = np.transpose(np.array(file['y2'])).astype('float32')
    y4 = np.transpose(np.array(file['y4'])).astype('float32')
    y6 = np.transpose(np.array(file['y6'])).astype('float32')
    y8 = np.transpose(np.array(file['y8'])).astype('float32')
    y10 = np.transpose(np.array(file['y10'])).astype('float32')
    y12 = np.transpose(np.array(file['y12'])).astype('float32')
    y15 = np.transpose(np.array(file['y15'])).astype('float32')
    p = np.transpose(np.array(file['p'])).astype('float32') #导频信号
# (train_images , train_labels),(_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 32, 32, 2).astype('float32')  # reshape增加一个channel维度
# (train_images - 127.5) / 127.5
# 批大小
BTATH_SIZE = 256
BUFFER_SIZE = 100000

datasets = tf.data.Dataset.from_tensor_slices(train_images)
datasets = datasets.shuffle(BUFFER_SIZE).batch(BTATH_SIZE)


# 生成器
def generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(2048, input_shape=(100,), use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((32, 32, 2)))
    model.add(layers.Conv2D(128, 5, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(128, 3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(64, 3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(32, 3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(32, 3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(16, 3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(2, 3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    return model  ###


# 辨别器
def discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, 3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(128, 3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(64, 3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(32, 3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(16, 3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(2, 3, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Flatten())
    model.add(layers.Dense(256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(1))

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # 计算真实标签和预测标签之间的交叉熵损失#最后一层没有激活所以要加from_logits=True


# 辨别器损失函数
def discriminator_loss(real_out, fake_out):
    real_loss = cross_entropy(tf.ones_like(real_out), real_out)  # tf.ones_like创建一个所有元素都设为1的张量。给定一个张量(张量)，
    # 这个操作返回一个与所有元素都设为1的张量类型和形状相同的张量。
    fake_loss = cross_entropy(tf.zeros_like(fake_out), fake_out)
    return real_loss + fake_loss


# 生成器损失函数
def generator_loss(fake_out):
    return cross_entropy(tf.ones_like(fake_out), fake_out)  # 我们希望生成的fake_out被判定为1#这里都是判别器的输出


# 因为G和D是两个模型，所以我们创建两个优化器
generator_opt = tf.keras.optimizers.Adam(1e-4)
discriminator_opt = tf.keras.optimizers.Adam(1e-4)
# 训练参数
EPOCHS = 100
noise_dim = 100

num_exp_to_generate = 16  # 每个epoch生成16个样本，以观察方向是否正确
seed = tf.random.normal([num_exp_to_generate, noise_dim])  # 生成正态分布随机数

generator = generator_model()  # 返回model
discriminator = discriminator_model()


# 训练步骤函数
def train_step(images):  # 接收一个批次的图片对一个批次的图片进行训练
    noise = tf.random.normal([BTATH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:  # 两个模型的梯度
        real_out = discriminator(images, training=True)  # 可训练的

        gen_image = generator(noise, training=True)
        fake_out = discriminator(gen_image, training=True)  # 假图片的输出

        gen_loss = generator_loss(fake_out)
        disc_loss = discriminator_loss(real_out, fake_out)

    gradient_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)  # 生成器的变量
    gradient_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_opt.apply_gradients(zip(gradient_gen, generator.trainable_variables))  # 使用定义好的优化器 #输入参数是通过什么对什么进行优化
    discriminator_opt.apply_gradients(zip(gradient_disc, discriminator.trainable_variables))


# 绘制图片
def generate_plot_image(gen_model, test_noise):  # 接收生成模型和测试噪声
    pre_images = gen_model(test_noise, training=False)  # 这里不需训练设置为false
    fig = plt.figure(figsize=(4, 4))  # 初始化画布
    for i in range(pre_images.shape[0]):  # pre_image的第一维度是个数即16
        plt.subplot(4, 4, i + 1)  # 从1开始
        plt.imshow((pre_images[i, :, :, 0] + 1) / 2, cmap='gray')  # 生成器用来tanh范围是[-1,1],所以将其变到[0,1] #设置使用黑白颜色
        plt.axis('off')  # 不显示坐标系
    plt.show()  # 一起显示出来


# 训练函数
def train(datasets, epochs):
    for epoch in range(epochs):  # 循环多少epoch
        for _ in range(3):
            for image_batch in datasets:  # 循环数据集中所有批次
                train_step(image_batch)
                print('.', end='')
        # generate_plot_image(generator,seed)
        plot_nmmse()



#new_model = tf.keras.models.load_model('my_model_20.h5', compile=False)
noise_channel_ini = np.array(tf.random.normal([20, noise_dim]))#生成正态分布随机数#放在外面保证起始点相同
def nmmse(y):  #定义返回nmmse的函数
    noise_channel = tf.Variable(noise_channel_ini, name='noise_channel')
    print(noise_channel[0,1])
    #定义优化器
    channel_opt = tf.keras.optimizers.Adam(0.01)
    for _ in range(300):
        #对输入噪声的梯度
        with tf.GradientTape() as chan_tape:
        #使用2范数代价函数衡量信道真实情况
            #h = generator(noise_channel, training=False)

            h = generator(noise_channel, training=False)
            loss_channel = 0
            for i in range(0, 20):
                a = tf.reshape(y[i, :, :, 0], (32, 16))
                b = tf.reshape(h[i, :, :, 0], (32, 32))
                c = tf.reshape(p[0, :, :, 0], (32, 16))
                a1 = tf.reshape(y[i, :, :, 1], (32, 16))
                b1 = tf.reshape(h[i, :, :, 1], (32, 32))
                c1 = tf.reshape(p[0, :, :, 1], (32, 16))
                y_ = tf.matmul(b, c)
                y1_ = tf.matmul(b1, c1)
                m = tf.keras.losses.MeanSquaredError()
                loss_channel_temp = 0.5*(m(a, y_) + m(a1, y1_))
                loss_channel = loss_channel + loss_channel_temp
            loss_channel = 0.1*loss_channel
            #loss_channel = 0.5*(m(a, y_) + m(a1, y1_))
        gradient_chan = chan_tape.gradient(loss_channel, noise_channel)
        #使用优化器
        channel_opt.apply_gradients([(gradient_chan, noise_channel)])

    print(noise_channel[0, 1])
    #计算最终的信道
    H_e=generator(noise_channel,training=False)
    #计算归一化均方误差
    #NMMSE=0.01*sum(sum((H-He_p1).^2))/(sum(sum((H.*H))));
    NMMSE = 0
    for j in range(20):
        temp=(np.linalg.norm((np.reshape(H[j, :, :, :],(32,64))-np.reshape(H_e[j,:,:,:], (32, 64))), ord=2)/np.linalg.norm(np.reshape(H[j, :, :, :], (32,64)),ord=2))
        NMMSE = NMMSE+temp
    NMMSE = 0.5*0.1*NMMSE
    NMMSE=10*math.log(NMMSE,10)
    return NMMSE




#定义画nmse-snr图的函数
def plot_nmmse():
    # 求不同信噪比时的nmmse
    NMSEd5 = nmmse(yd5)
    NMSEd2 = nmmse(yd2)
    NMSE0 = nmmse(y0)
    NMSE2 = nmmse(y2)
    NMSE4 = nmmse(y4)
    NMSE6 = nmmse(y6)
    NMSE8 = nmmse(y8)
    NMSE10 = nmmse(y10)
    NMSE12 = nmmse(y12)
    NMSE15 = nmmse(y15)
    Nmse = [NMSEd5, NMSEd2, NMSE0, NMSE2, NMSE4, NMSE6, NMSE8, NMSE10, NMSE12, NMSE15]
    SNR = [-5, -2, 0, 2, 4, 6, 8, 10, 12, 15]
    fig = plt.figure(figsize=(1,1)) #初始化画布
    plt.plot(SNR,Nmse)
    plt.show() #一起显示出来


train(datasets, EPOCHS)




